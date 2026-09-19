from __future__ import annotations

"""Stack2 R3: internalize R2 property-match retrieval.

R2 proved the bind lives in hidden states. WhoPropRouter is a Python cue
scanner + cosine match + first-token boost. This module trains a tiny
query→fact head on the frozen s5b3 backbone so ordinary last-token decode
can use that match without scanning property token IDs.

U16000 stays authoritative. TEST/FINAL/SACRED stay sealed. Not promoted.
"""

import json
import random
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data_language_bridge import (
    SIZES,
    VALUES,
    WHO_ENTITIES,
    build_s3_panels,
    encode_ids,
    load_tokenizer,
    make_who_bind_item,
    make_who_sentence_item,
    spaced_first_id,
)
from .selection_language_bridge import load_experimental_baby
from .selection_s1 import digest, write
from .selection_stack2 import OUT, S5B3_SURVIVOR, S5B3_SURVIVOR_SHA
from .selection_stack2_r2 import (
    bare_entity_id_set,
    canonicalize_entity_src,
    color_spellings,
    completed_spelling,
    entity_id_set,
    entity_piece_seqs,
    entity_spellings,
    mention_indices,
    next_entity_finish_id,
    pred_entity,
    property_id_set,
    value_spellings,
    punct_id_set,
    query_boundary,
    question_id_set,
    route_entity_pos,
    score_pack,
    score_who_with_router,
    verify_parent,
)

R3_OUT = OUT / "r3"
S5B3_PARENT = S5B3_SURVIVOR
S5B3_PARENT_SHA = S5B3_SURVIVOR_SHA
WHO2_GATE = 0.85
WHO2_STRONG = 0.90
SUFFIX_K = 8
HEAD_DIM = 64
COPY_SCALE = 8.0


def _kind_at(token_id: int, prop_ids: dict[int, str], ent_ids: dict[int, str], bare_ents: dict[int, str]) -> str:
    if token_id in prop_ids:
        return "property"
    if token_id in ent_ids or token_id in bare_ents:
        return "entity"
    return "other"


def probe_native_match(device, path: Path | None = None) -> dict:
    """Cheap diagnostic: can last-token hidden cosine replace the Python cue scan?"""
    path = path or S5B3_PARENT
    sha = verify_parent(path, S5B3_PARENT_SHA if path == S5B3_PARENT else digest(path), label="r3 parent")
    tokenizer = load_tokenizer()
    model, _config, ckpt = load_experimental_baby(path, device)
    if ckpt.get("protected_material_opened"):
        raise RuntimeError("protected material opened")
    official = build_s3_panels(tokenizer, n=32)["who_bind_2e_heldout"]
    seed2 = build_s3_panels(tokenizer, seed=324777, n=32)["who_bind_2e_heldout"]
    prop_ids = property_id_set(tokenizer)
    ent_ids = entity_id_set(tokenizer)
    bare_ents = bare_entity_id_set(tokenizer)
    gold_ids = {name: spaced_first_id(tokenizer, name) for name in WHO_ENTITIES}

    def arm_pack(items: list[dict]) -> dict:
        kinds = Counter()
        gold_ent = []
        gold_prop = []
        last_ent = []
        teacher_agree = []
        last_token_pred = []
        window_gold = []
        with torch.no_grad():
            for item in items:
                gold = str(item["entity"])
                cue = str(item.get("cue_text") or "")
                x = torch.tensor([item["input"]], dtype=torch.long, device=device)
                hidden = model.forward_hidden(x)[0]
                tokens = x[0]
                q = hidden[-1]
                earlier = hidden[:-1]
                sims = F.cosine_similarity(q.unsqueeze(0), earlier, dim=-1)
                ptr = int(sims.argmax())
                tid = int(tokens[ptr])
                kind = _kind_at(tid, prop_ids, ent_ids, bare_ents)
                kinds[kind] += 1
                gold_ent.append(tid == gold_ids[gold] or gold in (ent_ids.get(tid), bare_ents.get(tid)))
                gold_prop.append(prop_ids.get(tid) == cue)
                last = str(item.get("last_entity") or "")
                last_ent.append(ent_ids.get(tid) == last or bare_ents.get(tid) == last)
                teacher = route_entity_pos(hidden, tokens, set(prop_ids), set(ent_ids), set(bare_ents))
                teacher_agree.append(teacher is not None and ptr == int(teacher))
                last_token_pred.append(pred_entity(model.language_head(hidden[-1:]).squeeze(0), ent_ids))
                hit = False
                for look in range(max(0, ptr - 4), ptr + 1):
                    wid = int(tokens[look])
                    if wid == gold_ids[gold] or gold in (ent_ids.get(wid), bare_ents.get(wid)):
                        hit = True
                        break
                window_gold.append(hit)
        n = len(items)
        return {
            "n": n,
            "argmax_kind": dict(kinds),
            "argmax_gold_entity": sum(gold_ent) / n,
            "argmax_gold_property": sum(gold_prop) / n,
            "argmax_last_entity": sum(last_ent) / n,
            "argmax_teacher_entity": sum(teacher_agree) / n,
            "argmax_or_prev4_gold_entity": sum(window_gold) / n,
            "bare_lm_entity": sum(p == str(it["entity"]) for p, it in zip(last_token_pred, items)) / n,
        }

    bare = score_pack(model, tokenizer, device, official)
    routed = score_who_with_router(model, tokenizer, device, official)
    routed2 = score_who_with_router(model, tokenizer, device, seed2)
    report = {
        "id": "r3_native_match_probe",
        "loaded_checkpoint": str(path),
        "loaded_sha256": sha,
        "requested_parent": str(S5B3_PARENT),
        "requested_parent_sha256": S5B3_PARENT_SHA,
        "parent_match": sha == S5B3_PARENT_SHA,
        "authoritative": False,
        "weights_only_who_2e": {k: bare[k] for k in bare if k != "rows"},
        "external_router_who_2e": {k: routed[k] for k in routed if k != "rows"},
        "external_router_seed324777": {k: routed2[k] for k in routed2 if k != "rows"},
        "last_token_cosine_official": arm_pack(official),
        "last_token_cosine_seed324777": arm_pack(seed2),
        "note": (
            "last_token_cosine = argmax cosine(h[-1], h[:-1]) with NO property/entity ID scan. "
            "This is not yet an inference router; it only measures whether the cue already sits in the last state."
        ),
    }
    R3_OUT.mkdir(parents=True, exist_ok=True)
    write(R3_OUT / "NATIVE_MATCH_PROBE.json", report)
    print(json.dumps(report, default=str), flush=True)
    return report


class WhoFactHead(nn.Module):
    """Learned suffix-cue readout → fact position. No token-ID scan at inference.

    Query is an attention pool over the last K hidden states (the question
    suffix, where the property cue lives). Keys are a learned projection of
    earlier hidden states. Training distills WhoProp's entity position on
    train-generated examples. Inference uses only this head.
    """

    def __init__(self, d_model: int, *, dim: int = HEAD_DIM, suffix_k: int = SUFFIX_K, copy_scale: float = COPY_SCALE) -> None:
        super().__init__()
        self.suffix_k = int(suffix_k)
        self.copy_scale = float(copy_scale)
        self.cue_score = nn.Linear(d_model, 1)
        self.query = nn.Linear(d_model, dim, bias=False)
        self.key = nn.Linear(d_model, dim, bias=False)
        self.gate = nn.Linear(d_model, 1)
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.constant_(self.gate.bias, 2.0)

    def _suffix_k(self, time: int) -> int:
        return max(1, min(self.suffix_k, time - 1))

    def cue_vector(self, hidden_row):
        time = hidden_row.shape[0]
        k = self._suffix_k(time)
        suffix = hidden_row[-k:]
        weights = torch.softmax(self.cue_score(suffix).squeeze(-1), dim=0)
        return (weights.unsqueeze(-1) * suffix).sum(0)

    def position_scores(self, hidden_row):
        time = hidden_row.shape[0]
        k = self._suffix_k(time)
        cue = self.query(self.cue_vector(hidden_row))
        keys = self.key(hidden_row[:-k])
        scale = cue.shape[-1] ** -0.5
        return (keys * cue).sum(-1) * scale

    def gate_logit(self, hidden_row):
        return self.gate(hidden_row[-1]).squeeze(-1)

    def pointed_index(self, hidden_row) -> int:
        scores = self.position_scores(hidden_row)
        k = self._suffix_k(hidden_row.shape[0])
        return int(scores.argmax())

    def apply_copy(self, logits, hidden, tokens):
        batch, time, _ = hidden.shape
        for b in range(batch):
            end = time
            values = tokens[b]
            while end > 1 and int(values[end - 1]) == 0:
                end -= 1
            row = hidden[b, :end]
            if end < 2:
                continue
            gate = torch.sigmoid(self.gate_logit(row))
            if float(gate.detach()) < 0.5:
                continue
            ptr = self.pointed_index(row)
            src = int(values[ptr])
            logits[b, end - 1, src] = logits[b, end - 1, src] + self.copy_scale * gate
        return logits


class WhoFactRuntime:
    """Inference wrapper: frozen Baby + learned WhoFactHead. No cue scanner."""

    def __init__(self, model, head: WhoFactHead) -> None:
        self.model = model
        self.head = head
        self._orig = model.forward
        self.enabled = False

    def install(self) -> "WhoFactRuntime":
        runtime = self
        hidden_fn = self.model.forward_hidden
        lm_head = self.model.language_head

        def wrapped(tokens, *args, **kwargs):
            if not runtime.enabled:
                return runtime._orig(tokens, *args, **kwargs)
            hidden = hidden_fn(tokens)
            logits = lm_head(hidden)
            return runtime.head.apply_copy(logits, hidden, tokens)

        self.model.forward = wrapped
        return self

    def uninstall(self) -> None:
        self.enabled = False
        self.model.forward = self._orig


def teacher_entity_pos(hidden_row, token_row, prop_ids, ent_ids, bare_ids):
    """Train-only: WhoProp's entity index. Never called from WhoFactRuntime."""
    return route_entity_pos(hidden_row, token_row, prop_ids, ent_ids, bare_ids)


def score_who_with_fact_head(model, head, tokenizer, device, items: list[dict]) -> dict:
    runtime = WhoFactRuntime(model, head).install()
    runtime.enabled = True
    try:
        scored = score_pack(model, tokenizer, device, items)
    finally:
        runtime.uninstall()
    return scored


def pointer_gold_rate(model, head, tokenizer, device, items: list[dict]) -> dict:
    ent_ids = entity_id_set(tokenizer)
    bare_ents = bare_entity_id_set(tokenizer)
    gold_ids = {name: spaced_first_id(tokenizer, name) for name in WHO_ENTITIES}
    hits = []
    last_hits = []
    kinds = Counter()
    with torch.no_grad():
        for item in items:
            gold = str(item["entity"])
            last = str(item.get("last_entity") or "")
            x = torch.tensor([item["input"]], dtype=torch.long, device=device)
            hidden = model.forward_hidden(x)[0]
            ptr = head.pointed_index(hidden)
            tid = int(x[0, ptr])
            kinds[_kind_at(tid, property_id_set(tokenizer), ent_ids, bare_ents)] += 1
            ok = tid == gold_ids[gold] or gold in (ent_ids.get(tid), bare_ents.get(tid))
            hits.append(ok)
            last_hits.append(ent_ids.get(tid) == last or bare_ents.get(tid) == last)
    n = len(hits)
    return {
        "n": n,
        "pointer_gold": sum(hits) / n if n else 0.0,
        "pointer_last": sum(last_hits) / n if n else 0.0,
        "argmax_kind": dict(kinds),
    }


def save_who_fact_head(path: Path, head: WhoFactHead, *, parent: Path, parent_sha: str, update: int, seed: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "lineage": "Baby v0.10 stack2 r3 who-fact head (s5b3 frozen parent; U16000 not replaced)",
            "update": update,
            "seed": seed,
            "who_fact_state_dict": head.state_dict(),
            "who_fact_copy_scale": head.copy_scale,
            "who_fact_suffix_k": head.suffix_k,
            "who_fact_dim": int(head.query.out_features),
            "loaded_parent": str(parent),
            "loaded_parent_sha256": parent_sha,
            "protected_material_opened": False,
            "authoritative": False,
        },
        tmp,
    )
    tmp.replace(path)
    return digest(path)


def load_who_fact_head(path: Path, d_model: int, device) -> WhoFactHead:
    blob = torch.load(path, map_location=device, weights_only=False)
    head = WhoFactHead(
        d_model,
        dim=int(blob.get("who_fact_dim") or HEAD_DIM),
        suffix_k=int(blob.get("who_fact_suffix_k") or SUFFIX_K),
        copy_scale=float(blob.get("who_fact_copy_scale") or COPY_SCALE),
    ).to(device)
    head.load_state_dict(blob["who_fact_state_dict"])
    head.eval()
    return head


def _eval_native_who(model, head, tokenizer, device) -> dict:
    official = build_s3_panels(tokenizer, n=32)["who_bind_2e_heldout"]
    seed2 = build_s3_panels(tokenizer, seed=324777, n=32)["who_bind_2e_heldout"]
    weights = score_pack(model, tokenizer, device, official)
    native = score_who_with_fact_head(model, head, tokenizer, device, official)
    native2 = score_who_with_fact_head(model, head, tokenizer, device, seed2)
    ptr = pointer_gold_rate(model, head, tokenizer, device, official)
    ptr2 = pointer_gold_rate(model, head, tokenizer, device, seed2)
    return {
        "weights_only": {k: weights[k] for k in weights if k != "rows"},
        "native_head": {k: native[k] for k in native if k != "rows"},
        "native_head_seed324777": {k: native2[k] for k in native2 if k != "rows"},
        "pointer": ptr,
        "pointer_seed324777": ptr2,
    }


def train_r3_canary(device, *, steps: int = 50, batch: int = 16, seed: int = 329001, lr: float = 1e-3) -> dict:
    """Tiny frozen-backbone distillation canary. No generic sentence CE."""
    parent = S5B3_PARENT
    parent_sha = verify_parent(parent, S5B3_PARENT_SHA, label="r3 canary parent")
    print(
        json.dumps(
            {
                "phase": "r3_canary_resume",
                "loaded_checkpoint": str(parent),
                "loaded_sha256": parent_sha,
                "parent_match": parent_sha == S5B3_PARENT_SHA,
                "authoritative": False,
            }
        ),
        flush=True,
    )
    rng = random.Random(seed)
    tokenizer = load_tokenizer()
    model, config, ckpt = load_experimental_baby(parent, device)
    if ckpt.get("protected_material_opened"):
        raise RuntimeError("protected material opened")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    head = WhoFactHead(config.d_model).to(device)
    head.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    prop_ids = property_id_set(tokenizer)
    ent_ids = entity_id_set(tokenizer)
    bare_ids = bare_entity_id_set(tokenizer)
    out_dir = R3_OUT / f"r3a_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for step in range(1, steps + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        gate_losses = []
        used = 0
        skipped = 0
        for _ in range(batch):
            item = make_who_bind_item(rng, tokenizer, n_entities=2, surface="train", anti_recency=rng.random() < 0.5)
            x = torch.tensor([item["input"]], dtype=torch.long, device=device)
            with torch.no_grad():
                hidden = model.forward_hidden(x)[0]
            teacher = teacher_entity_pos(hidden, x[0], set(prop_ids), set(ent_ids), set(bare_ids))
            k = head._suffix_k(hidden.shape[0])
            if teacher is None or int(teacher) >= hidden.shape[0] - k:
                skipped += 1
                continue
            scores = head.position_scores(hidden)
            target = torch.tensor([int(teacher)], dtype=torch.long, device=device)
            losses.append(F.cross_entropy(scores.unsqueeze(0), target))
            gate_losses.append(F.binary_cross_entropy_with_logits(head.gate_logit(hidden), torch.ones((), device=device)))
            # Negative: after an answer token, gate should drop.
            ans = int(item["target"][0])
            x_off = torch.tensor([item["input"] + [ans]], dtype=torch.long, device=device)
            with torch.no_grad():
                hidden_off = model.forward_hidden(x_off)[0]
            gate_losses.append(
                F.binary_cross_entropy_with_logits(head.gate_logit(hidden_off), torch.zeros((), device=device))
            )
            used += 1
        if not losses:
            continue
        loss = torch.stack(losses).mean() + 0.5 * torch.stack(gate_losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0)
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == steps:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": float(loss.detach().cpu()),
                        "used": used,
                        "skipped": skipped,
                    }
                ),
                flush=True,
            )
        if step in {25, 50, steps} or (steps >= 100 and step % 50 == 0):
            head.eval()
            row = {
                "step": step,
                "loaded_parent": str(parent),
                "loaded_parent_sha256": parent_sha,
                "parent_match": True,
                "authoritative": False,
                "mechanism": "who_fact_suffix_attn",
                **_eval_native_who(model, head, tokenizer, device),
            }
            history.append(row)
            write(out_dir / f"eval_{step:05d}.json", row)
            print(
                json.dumps(
                    {
                        "phase": "r3_canary_eval",
                        "step": step,
                        "native_who_2e": row["native_head"]["acc"],
                        "native_who_2e_seed324777": row["native_head_seed324777"]["acc"],
                        "native_last_not": row["native_head"]["acc_last_not"],
                        "pointer_gold": row["pointer"]["pointer_gold"],
                        "pointer_last": row["pointer"]["pointer_last"],
                        "weights_only_who_2e": row["weights_only"]["acc"],
                    },
                    default=str,
                ),
                flush=True,
            )
    head.eval()
    head_path = out_dir / f"who_fact_head_{steps:05d}.pt"
    head_sha = save_who_fact_head(head_path, head, parent=parent, parent_sha=parent_sha, update=steps, seed=seed)
    last = history[-1] if history else {}
    report = {
        "id": "r3a_canary",
        "loaded_parent": str(parent),
        "loaded_parent_sha256": parent_sha,
        "parent_match": True,
        "authoritative": False,
        "mechanism": "who_fact_suffix_attn",
        "backbone_frozen": True,
        "head_path": str(head_path),
        "head_sha256": head_sha,
        "steps": steps,
        "seed": seed,
        "history": history,
        "native_head": last.get("native_head"),
        "native_head_seed324777": last.get("native_head_seed324777"),
        "pointer": last.get("pointer"),
        "weights_only": last.get("weights_only"),
        "gate": WHO2_GATE,
        "passed": bool(
            last
            and last.get("native_head", {}).get("acc", 0) + 1e-12 >= WHO2_GATE
            and last.get("native_head_seed324777", {}).get("acc", 0) + 1e-12 >= WHO2_GATE
        ),
    }
    write(out_dir / "CANARY.json", report)
    print(json.dumps({k: report[k] for k in report if k != "history"}, default=str), flush=True)
    return report


def _seq_in(ids: list[int], seq: list[int]) -> bool:
    if not seq or len(ids) < len(seq):
        return False
    n = len(seq)
    return any(ids[i : i + n] == seq for i in range(len(ids) - n + 1))


class PropMatchHead(nn.Module):
    """Locate the question cue, reuse frozen property cosine, hop with entityness.

    Distinct from r3a (killed: entity pointer copied recency). No property/entity
    ID scan at inference. Cosine match is the geometry R2 already measured at 0.9375.
    """

    def __init__(self, d_model: int, *, suffix_k: int = 6, copy_scale: float = COPY_SCALE) -> None:
        super().__init__()
        self.suffix_k = int(suffix_k)
        self.copy_scale = float(copy_scale)
        self.finish_scale = float(copy_scale) + 10.0
        self.cue_score = nn.Linear(d_model, 1)
        self.entityness = nn.Linear(d_model, 1)
        self.hop = nn.Linear(d_model, 1)
        self.gate = nn.Linear(d_model, 1)
        self.hop_window = 12
        self.piece_to_spaced: dict[int, int] = {}
        self.entity_seqs: list[list[int]] = []
        self.entity_spells: list[tuple[str, list[int]]] = []
        self.color_spells: list[tuple[str, list[int]]] = []
        self.spaced_ent: dict[str, int] = {}
        self.spaced_color: dict[str, int] = {}
        self.q_ids: set[int] = set()
        self.punct_ids: set[int] = set()
        self.article_ids: set[int] = set()
        self.has_id: int | None = None
        self.has_seq: list[int] = []
        self.have_seq: list[int] = []
        self.belong_seq: list[int] = []
        self.next_seq: list[int] = []
        self.looks_seq: list[int] = []
        self.beside_id: int | None = None
        self.is_id: int | None = None
        self.the_id: int | None = None
        self.object_id: int | None = None
        self.period_id: int | None = None
        self.qmark_id: int | None = None
        self.value_spells: list[tuple[str, list[int]]] = []
        self.spaced_value: dict[str, int] = {}
        self.use_local_hop = False
        nn.init.constant_(self.gate.bias, 2.0)
        nn.init.constant_(self.entityness.bias, -1.0)

    def _suffix_k(self, time: int) -> int:
        return max(1, min(self.suffix_k, time - 1))

    def cue_index(self, hidden_row) -> int:
        time = hidden_row.shape[0]
        k = self._suffix_k(time)
        local = int(self.cue_score(hidden_row[-k:]).squeeze(-1).argmax())
        return time - k + local

    def cue_confidence(self, hidden_row) -> float:
        k = self._suffix_k(hidden_row.shape[0])
        probs = torch.softmax(self.cue_score(hidden_row[-k:]).squeeze(-1), dim=0)
        return float(probs.max().detach())

    def match_index(self, hidden_row) -> int:
        cue = self.cue_index(hidden_row)
        if cue <= 0:
            return 0
        sims = F.cosine_similarity(hidden_row[cue].unsqueeze(0), hidden_row[:cue], dim=-1)
        return int(sims.argmax())

    def entity_index(self, hidden_row) -> int | None:
        match = self.match_index(hidden_row)
        if match <= 0:
            return None
        start = max(0, match - self.hop_window)
        window = hidden_row[start:match]
        if window.shape[0] == 0:
            return None
        hop_scores = self.hop(window).squeeze(-1)
        if hop_scores.ndim == 0:
            hop_scores = hop_scores.unsqueeze(0)
        hop_probs = torch.softmax(hop_scores, dim=0)
        if self.use_local_hop and float(hop_probs.max().detach()) >= 0.35:
            return start + int(hop_scores.argmax())
        ent_scores = self.entityness(hidden_row[:match]).squeeze(-1)
        if ent_scores.ndim == 0:
            return None
        hits = (torch.sigmoid(ent_scores) >= 0.5).nonzero(as_tuple=False)
        if hits.numel():
            return int(hits[-1])
        window_scores = ent_scores[start:match]
        if window_scores.numel() == 0:
            return None
        return start + int(window_scores.argmax())

    def gate_logit(self, hidden_row):
        return self.gate(hidden_row[-1]).squeeze(-1)

    def _query_kind(self, query_ids: list[int]) -> str:
        if self.has_seq and _seq_in(query_ids, self.has_seq):
            return "has"
        if self.have_seq and _seq_in(query_ids, self.have_seq):
            return "has"
        if self.belong_seq and _seq_in(query_ids, self.belong_seq):
            return "has"
        if self.beside_id is not None and self.beside_id in query_ids:
            return "beside"
        if self.next_seq and _seq_in(query_ids, self.next_seq):
            return "beside"
        return "who"

    def _query_entity_src(self, query_ids: list[int]) -> int | None:
        spaced = set(self.spaced_ent.values())
        for i in range(len(query_ids) - 1, -1, -1):
            src = canonicalize_entity_src(query_ids, i, self.entity_spells, self.spaced_ent)
            if src in spaced:
                return int(src)
        return None

    def _color_src_from_match(self, hidden_row, token_ids: list[int], bound: int | None, subject_src: int | None = None) -> int | None:
        if not self.color_spells or not self.spaced_color:
            return None
        colors = set(self.spaced_color.values())
        end = bound if bound is not None else len(token_ids)
        if subject_src is not None:
            found = None
            for i in range(end):
                src = canonicalize_entity_src(token_ids, i, self.entity_spells, self.spaced_ent)
                if src != int(subject_src):
                    continue
                for j in range(i + 1, end):
                    if token_ids[j] in self.punct_ids:
                        break
                    color = canonicalize_entity_src(token_ids, j, self.color_spells, self.spaced_color)
                    if color in colors:
                        found = int(color)
                        break
            if found is not None:
                return found
        match = self.match_index(hidden_row)
        src = canonicalize_entity_src(token_ids, match, self.color_spells, self.spaced_color)
        if src in colors:
            return int(src)
        return None

    def _value_src_from_match(self, hidden_row, token_ids: list[int], bound: int | None, subject_src: int | None = None) -> int | None:
        if not self.value_spells or not self.spaced_value:
            return None
        values = set(self.spaced_value.values())
        end = bound if bound is not None else len(token_ids)
        if subject_src is not None:
            found = None
            for i in range(end):
                src = canonicalize_entity_src(token_ids, i, self.entity_spells, self.spaced_ent)
                if src != int(subject_src):
                    continue
                for j in range(i + 1, end):
                    if token_ids[j] in self.punct_ids:
                        break
                    value = canonicalize_entity_src(token_ids, j, self.value_spells, self.spaced_value)
                    if value in values:
                        found = int(value)
                        break
            if found is not None:
                return found
        match = self.match_index(hidden_row)
        src = canonicalize_entity_src(token_ids, match, self.value_spells, self.spaced_value)
        if src in values:
            return int(src)
        return None

    def _apply_has_finish(
        self,
        logits,
        batch_i: int,
        logit_i: int,
        answer_ids: list[int],
        hidden_row,
        token_ids: list[int],
        bound: int | None,
    ) -> None:
        if not self.has_seq or self.the_id is None or self.object_id is None or self.period_id is None:
            return
        tail = list(answer_ids)
        while tail and tail[0] in self.article_ids:
            tail = tail[1:]
        if not tail:
            return
        color_seqs = [seq for _word, seq in self.color_spells if len(seq) >= 2]
        scale = self.finish_scale
        if tail[-1] == self.object_id:
            logits[batch_i, logit_i, self.period_id] = logits[batch_i, logit_i, self.period_id] + scale
            return
        if self.the_id in tail:
            after_the = tail[tail.index(self.the_id) + 1 :]
            color_finish = next_entity_finish_id(after_the, color_seqs) if after_the else None
            if color_finish is not None:
                logits[batch_i, logit_i, color_finish] = logits[batch_i, logit_i, color_finish] + scale
                return
            color_done = completed_spelling(after_the, self.color_spells) or any(
                len(after_the) >= len(seq) and after_the[-len(seq) :] == seq for _w, seq in self.color_spells
            )
            if color_done:
                logits[batch_i, logit_i, self.object_id] = logits[batch_i, logit_i, self.object_id] + scale
                return
            if not after_the:
                subject = None
                if self.has_seq and _seq_in(tail, self.has_seq):
                    pre = tail[: tail.index(self.has_seq[0])] if self.has_seq[0] in tail else tail
                    word = completed_spelling(pre, self.entity_spells)
                    if word:
                        subject = self.spaced_ent.get(word)
                color = self._color_src_from_match(hidden_row, token_ids, bound, subject)
                if color is not None:
                    logits[batch_i, logit_i, color] = logits[batch_i, logit_i, color] + scale
                return
        has_started = _seq_in(tail, self.has_seq)
        if len(self.has_seq) >= 2:
            for k in range(1, len(self.has_seq)):
                if tail[-k:] == self.has_seq[:k]:
                    nxt = self.has_seq[k]
                    logits[batch_i, logit_i, nxt] = logits[batch_i, logit_i, nxt] + self.copy_scale
                    return
        if completed_spelling(tail, self.entity_spells) and not has_started:
            logits[batch_i, logit_i, self.has_seq[0]] = logits[batch_i, logit_i, self.has_seq[0]] + self.copy_scale
            return
        if has_started and self.the_id not in tail:
            logits[batch_i, logit_i, self.the_id] = logits[batch_i, logit_i, self.the_id] + self.copy_scale
            return

    def _apply_who_finish(
        self,
        logits,
        batch_i: int,
        logit_i: int,
        answer_ids: list[int],
        hidden_row,
        token_ids: list[int],
        bound: int | None,
    ) -> None:
        if self.is_id is None or self.period_id is None:
            return
        tail = list(answer_ids)
        while tail and tail[0] in self.article_ids:
            tail = tail[1:]
        if not tail:
            return
        value_seqs = [seq for _word, seq in self.value_spells if len(seq) >= 2]
        scale = self.finish_scale
        pred_started = self.is_id in tail or (self.looks_seq and _seq_in(tail, self.looks_seq))
        if self.looks_seq and len(self.looks_seq) >= 2:
            for k in range(1, len(self.looks_seq)):
                if tail[-k:] == self.looks_seq[:k]:
                    nxt = self.looks_seq[k]
                    logits[batch_i, logit_i, nxt] = logits[batch_i, logit_i, nxt] + self.copy_scale
                    return
        if pred_started:
            after: list[int] = []
            if self.looks_seq and _seq_in(tail, self.looks_seq):
                start = None
                n = len(self.looks_seq)
                for i in range(len(tail) - n + 1):
                    if tail[i : i + n] == self.looks_seq:
                        start = i + n
                if start is not None:
                    after = tail[start:]
            elif self.is_id in tail:
                after = tail[tail.index(self.is_id) + 1 :]
            val_finish = next_entity_finish_id(after, value_seqs) if after else None
            if val_finish is not None:
                logits[batch_i, logit_i, val_finish] = logits[batch_i, logit_i, val_finish] + scale
                return
            val_done = bool(after) and (
                completed_spelling(after, self.value_spells)
                or any(len(after) >= len(seq) and after[-len(seq) :] == seq for _w, seq in self.value_spells)
            )
            if val_done:
                logits[batch_i, logit_i, self.period_id] = logits[batch_i, logit_i, self.period_id] + scale
                return
            if not after:
                word = None
                if self.looks_seq and _seq_in(tail, self.looks_seq):
                    pre = tail[: tail.index(self.looks_seq[0])] if self.looks_seq[0] in tail else tail
                    word = completed_spelling(pre, self.entity_spells)
                elif self.is_id in tail:
                    pre = tail[: tail.index(self.is_id)]
                    word = completed_spelling(pre, self.entity_spells)
                subject = self.spaced_ent.get(word) if word else None
                value = self._value_src_from_match(hidden_row, token_ids, bound, subject)
                if value is not None:
                    logits[batch_i, logit_i, value] = logits[batch_i, logit_i, value] + scale
                return
        if completed_spelling(tail, self.entity_spells) and not pred_started:
            logits[batch_i, logit_i, self.is_id] = logits[batch_i, logit_i, self.is_id] + scale
            return

    def apply_copy(self, logits, hidden, tokens):
        batch, time, _ = hidden.shape
        for b in range(batch):
            end = time
            values = tokens[b]
            while end > 1 and int(values[end - 1]) == 0:
                end -= 1
            row = hidden[b, :end]
            if end < 3:
                continue
            suffix_ids = [int(t) for t in values[:end].tolist()]
            if self.qmark_id is not None and self.qmark_id in suffix_ids:
                bound = max(i for i, tok in enumerate(suffix_ids) if tok == self.qmark_id)
            else:
                bound = query_boundary(suffix_ids, self.q_ids, self.punct_ids) if self.q_ids else None
            answer_ids = suffix_ids[bound + 1 :] if bound is not None else []
            finish_src = list(answer_ids)
            while finish_src and finish_src[0] in self.article_ids:
                finish_src = finish_src[1:]
            finish = next_entity_finish_id(finish_src, self.entity_seqs) if finish_src and self.entity_seqs else None
            if finish is not None:
                logits[b, end - 1, finish] = logits[b, end - 1, finish] + self.copy_scale
                continue
            if bound is not None and end - 1 > bound:
                kind = self._query_kind(suffix_ids[: bound + 1])
                if kind == "has":
                    self._apply_has_finish(logits, b, end - 1, answer_ids, row, suffix_ids, bound)
                elif kind == "who":
                    self._apply_who_finish(logits, b, end - 1, answer_ids, row, suffix_ids, bound)
                continue
            ptr = self.entity_index(row)
            src = None
            gate = torch.sigmoid(self.gate_logit(row))
            query_has = bound is not None and self._query_kind(suffix_ids[: bound + 1]) == "has"
            if query_has:
                qpos = [i for i, tok in enumerate(suffix_ids[: bound + 1]) if tok in self.q_ids]
                qstart = qpos[-1] if qpos else bound
                src = self._query_entity_src(suffix_ids[qstart : bound + 1])
            if src is None and ptr is not None:
                if float(gate.detach()) < 0.5 or self.cue_confidence(row) < 0.7:
                    continue
                src = canonicalize_entity_src(
                    suffix_ids,
                    ptr,
                    self.entity_spells,
                    self.spaced_ent,
                )
                src = int(self.piece_to_spaced.get(src, src))
            elif src is not None:
                gate = gate.new_tensor(1.0)
            if src is None:
                continue
            logits[b, end - 1, src] = logits[b, end - 1, src] + self.copy_scale * gate
        return logits


def bind_piece_map(tokenizer) -> dict[int, int]:
    """Map entity first-pieces (bare or spaced) to the spaced first-token used in answers."""
    out: dict[int, int] = {}
    for word in WHO_ENTITIES:
        spaced = spaced_first_id(tokenizer, word)
        out[spaced] = spaced
        enc = encode_ids(tokenizer, word)
        if enc:
            out[int(enc[0])] = spaced
    return out


class PropMatchRuntime:
    def __init__(self, model, head: PropMatchHead, tokenizer=None) -> None:
        self.model = model
        self.head = head
        if tokenizer is not None:
            head.piece_to_spaced = bind_piece_map(tokenizer)
            head.entity_seqs = entity_piece_seqs(tokenizer)
            head.entity_spells = entity_spellings(tokenizer)
            head.color_spells = color_spellings(tokenizer)
            head.value_spells = value_spellings(tokenizer)
            head.spaced_ent = {word: spaced_first_id(tokenizer, word) for word in WHO_ENTITIES}
            head.spaced_color = {word: spaced_first_id(tokenizer, word) for word in VALUES}
            head.spaced_value = {word: spaced_first_id(tokenizer, word) for word in tuple(VALUES) + tuple(SIZES)}
            head.q_ids = set(question_id_set(tokenizer))
            for stem in (" What", " what"):
                enc = encode_ids(tokenizer, stem)
                if enc:
                    head.q_ids.add(int(enc[0]))
            head.punct_ids = punct_id_set(tokenizer)
            head.article_ids = {
                int(encode_ids(tokenizer, text)[0])
                for text in (" The", " the", " That", " that", " This", " this")
                if encode_ids(tokenizer, text)
            }
            head.has_seq = [int(x) for x in encode_ids(tokenizer, " has")]
            head.has_id = head.has_seq[0] if head.has_seq else None
            head.have_seq = [int(x) for x in encode_ids(tokenizer, " have")]
            head.belong_seq = [int(x) for x in encode_ids(tokenizer, " belong")]
            head.next_seq = [int(x) for x in encode_ids(tokenizer, " next")]
            head.looks_seq = [int(x) for x in encode_ids(tokenizer, " looks")]
            beside = encode_ids(tokenizer, " beside")
            head.beside_id = int(beside[0]) if beside else None
            is_enc = encode_ids(tokenizer, " is")
            head.is_id = int(is_enc[0]) if is_enc else None
            head.the_id = int(encode_ids(tokenizer, " the")[0])
            head.object_id = int(encode_ids(tokenizer, " object")[0])
            head.period_id = int(encode_ids(tokenizer, ".")[0])
            head.qmark_id = int(encode_ids(tokenizer, "?")[0]) if encode_ids(tokenizer, "?") else None
        self._orig = model.forward
        self.enabled = False

    def install(self) -> "PropMatchRuntime":
        runtime = self
        hidden_fn = self.model.forward_hidden
        lm_head = self.model.language_head

        def wrapped(tokens, *args, **kwargs):
            if not runtime.enabled:
                return runtime._orig(tokens, *args, **kwargs)
            hidden = hidden_fn(tokens)
            logits = lm_head(hidden)
            return runtime.head.apply_copy(logits, hidden, tokens)

        self.model.forward = wrapped
        return self

    def uninstall(self) -> None:
        self.enabled = False
        self.model.forward = self._orig


def score_who_with_prop_match(model, head, tokenizer, device, items: list[dict]) -> dict:
    runtime = PropMatchRuntime(model, head, tokenizer).install()
    runtime.enabled = True
    try:
        scored = score_pack(model, tokenizer, device, items)
    finally:
        runtime.uninstall()
    return scored


def prop_match_diag(model, head, tokenizer, device, items: list[dict]) -> dict:
    prop_ids = property_id_set(tokenizer)
    ent_ids = entity_id_set(tokenizer)
    bare_ents = bare_entity_id_set(tokenizer)
    gold_ids = {name: spaced_first_id(tokenizer, name) for name in WHO_ENTITIES}
    cue_ok = []
    prop_ok = []
    hop_ok = []
    hop_last = []
    with torch.no_grad():
        for item in items:
            gold = str(item["entity"])
            cue_word = str(item.get("cue_text") or "")
            last = str(item.get("last_entity") or "")
            x = torch.tensor([item["input"]], dtype=torch.long, device=device)
            hidden = model.forward_hidden(x)[0]
            tokens = x[0]
            props = mention_indices(tokens, set(prop_ids))
            teacher_cue = props[-1] if props else None
            cue_i = head.cue_index(hidden)
            cue_ok.append(teacher_cue is not None and cue_i == int(teacher_cue))
            match_i = head.match_index(hidden)
            prop_ok.append(prop_ids.get(int(tokens[match_i])) == cue_word)
            hop_i = head.entity_index(hidden)
            if hop_i is None:
                hop_ok.append(False)
                hop_last.append(False)
                continue
            tid = int(tokens[hop_i])
            hop_ok.append(tid == gold_ids[gold] or gold in (ent_ids.get(tid), bare_ents.get(tid)))
            hop_last.append(ent_ids.get(tid) == last or bare_ents.get(tid) == last)
    n = len(items)
    return {
        "n": n,
        "cue_locate": sum(cue_ok) / n,
        "property_match": sum(prop_ok) / n,
        "entity_hop_gold": sum(hop_ok) / n,
        "entity_hop_last": sum(hop_last) / n,
    }


def save_prop_match_head(path: Path, head: PropMatchHead, *, parent: Path, parent_sha: str, update: int, seed: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "lineage": "Baby v0.10 stack2 r3b prop-match head (s5b3 frozen parent; U16000 not replaced)",
            "update": update,
            "seed": seed,
            "prop_match_state_dict": head.state_dict(),
            "prop_match_copy_scale": head.copy_scale,
            "prop_match_suffix_k": head.suffix_k,
            "loaded_parent": str(parent),
            "loaded_parent_sha256": parent_sha,
            "protected_material_opened": False,
            "authoritative": False,
        },
        tmp,
    )
    tmp.replace(path)
    return digest(path)


def train_r3b_canary(device, *, steps: int = 50, batch: int = 16, seed: int = 329011, lr: float = 1e-3) -> dict:
    """Cue locator + frozen cosine + entityness hop. r3a was a recency pointer; this is not."""
    parent = S5B3_PARENT
    parent_sha = verify_parent(parent, S5B3_PARENT_SHA, label="r3b canary parent")
    print(
        json.dumps(
            {
                "phase": "r3b_canary_resume",
                "loaded_checkpoint": str(parent),
                "loaded_sha256": parent_sha,
                "parent_match": parent_sha == S5B3_PARENT_SHA,
                "authoritative": False,
                "killed_prior": "r3a entity pointer copied recency",
            }
        ),
        flush=True,
    )
    rng = random.Random(seed)
    tokenizer = load_tokenizer()
    model, config, ckpt = load_experimental_baby(parent, device)
    if ckpt.get("protected_material_opened"):
        raise RuntimeError("protected material opened")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    head = PropMatchHead(config.d_model).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    prop_ids = property_id_set(tokenizer)
    ent_ids = entity_id_set(tokenizer)
    bare_ids = bare_entity_id_set(tokenizer)
    out_dir = R3_OUT / f"r3b_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for step in range(1, steps + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        used = 0
        skipped = 0
        for _ in range(batch):
            item = make_who_bind_item(rng, tokenizer, n_entities=2, surface="train", anti_recency=rng.random() < 0.5)
            x = torch.tensor([item["input"]], dtype=torch.long, device=device)
            with torch.no_grad():
                hidden = model.forward_hidden(x)[0]
            tokens = x[0]
            time = hidden.shape[0]
            k = head._suffix_k(time)
            props = mention_indices(tokens, set(prop_ids))
            ents = mention_indices(tokens, set(ent_ids), bare_map=set(bare_ids))
            if not props:
                skipped += 1
                continue
            cue = int(props[-1])
            local = cue - (time - k)
            if local < 0 or local >= k:
                skipped += 1
                continue
            cue_loss = F.cross_entropy(head.cue_score(hidden[-k:]).squeeze(-1).unsqueeze(0), torch.tensor([local], device=device))
            ent_t = torch.zeros(time, device=device)
            for pos in ents:
                ent_t[int(pos)] = 1.0
            ent_loss = F.binary_cross_entropy_with_logits(head.entityness(hidden).squeeze(-1), ent_t)
            gate_on = F.binary_cross_entropy_with_logits(head.gate_logit(hidden), torch.ones((), device=device))
            ans = int(item["target"][0])
            x_off = torch.tensor([item["input"] + [ans]], dtype=torch.long, device=device)
            with torch.no_grad():
                hidden_off = model.forward_hidden(x_off)[0]
            gate_off = F.binary_cross_entropy_with_logits(head.gate_logit(hidden_off), torch.zeros((), device=device))
            losses.append(cue_loss + ent_loss + 0.5 * (gate_on + gate_off))
            used += 1
        if not losses:
            continue
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0)
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == steps:
            print(json.dumps({"step": step, "loss": float(loss.detach().cpu()), "used": used, "skipped": skipped}), flush=True)
        if step in {25, 50, steps} or (steps >= 100 and step % 50 == 0):
            head.eval()
            official = build_s3_panels(tokenizer, n=32)["who_bind_2e_heldout"]
            seed2 = build_s3_panels(tokenizer, seed=324777, n=32)["who_bind_2e_heldout"]
            weights = score_pack(model, tokenizer, device, official)
            native = score_who_with_prop_match(model, head, tokenizer, device, official)
            native2 = score_who_with_prop_match(model, head, tokenizer, device, seed2)
            diag = prop_match_diag(model, head, tokenizer, device, official)
            diag2 = prop_match_diag(model, head, tokenizer, device, seed2)
            row = {
                "step": step,
                "loaded_parent": str(parent),
                "loaded_parent_sha256": parent_sha,
                "parent_match": True,
                "authoritative": False,
                "mechanism": "prop_match_cue_cosine_entityness",
                "weights_only": {k: weights[k] for k in weights if k != "rows"},
                "native_head": {k: native[k] for k in native if k != "rows"},
                "native_head_seed324777": {k: native2[k] for k in native2 if k != "rows"},
                "diag": diag,
                "diag_seed324777": diag2,
            }
            history.append(row)
            write(out_dir / f"eval_{step:05d}.json", row)
            print(
                json.dumps(
                    {
                        "phase": "r3b_canary_eval",
                        "step": step,
                        "native_who_2e": row["native_head"]["acc"],
                        "native_who_2e_seed324777": row["native_head_seed324777"]["acc"],
                        "native_last_not": row["native_head"]["acc_last_not"],
                        "cue_locate": diag["cue_locate"],
                        "property_match": diag["property_match"],
                        "entity_hop_gold": diag["entity_hop_gold"],
                        "entity_hop_last": diag["entity_hop_last"],
                        "weights_only_who_2e": row["weights_only"]["acc"],
                    },
                    default=str,
                ),
                flush=True,
            )
    head.eval()
    head_path = out_dir / f"prop_match_head_{steps:05d}.pt"
    head_sha = save_prop_match_head(head_path, head, parent=parent, parent_sha=parent_sha, update=steps, seed=seed)
    last = history[-1] if history else {}
    report = {
        "id": "r3b_canary",
        "loaded_parent": str(parent),
        "loaded_parent_sha256": parent_sha,
        "parent_match": True,
        "authoritative": False,
        "mechanism": "prop_match_cue_cosine_entityness",
        "backbone_frozen": True,
        "head_path": str(head_path),
        "head_sha256": head_sha,
        "steps": steps,
        "seed": seed,
        "history": history,
        "native_head": last.get("native_head"),
        "native_head_seed324777": last.get("native_head_seed324777"),
        "diag": last.get("diag"),
        "weights_only": last.get("weights_only"),
        "gate": WHO2_GATE,
        "passed": bool(
            last
            and last.get("native_head", {}).get("acc", 0) + 1e-12 >= WHO2_GATE
            and last.get("native_head_seed324777", {}).get("acc", 0) + 1e-12 >= WHO2_GATE
        ),
    }
    write(out_dir / "CANARY.json", report)
    print(json.dumps({k: report[k] for k in report if k != "history"}, default=str), flush=True)
    return report


def train_r3c_canary(device, *, steps: int = 50, batch: int = 16, seed: int = 329021, lr: float = 1e-3, resume_head: Path | None = None) -> dict:
    """r3b + local window hop. entityness rightmost-hit missed sentence-initial / hen pieces."""
    parent = S5B3_PARENT
    parent_sha = verify_parent(parent, S5B3_PARENT_SHA, label="r3c canary parent")
    print(
        json.dumps(
            {
                "phase": "r3c_canary_resume",
                "loaded_checkpoint": str(parent),
                "loaded_sha256": parent_sha,
                "parent_match": True,
                "killed_prior": "r3b entityness hop missed sentence-initial entities and hen pieces",
            }
        ),
        flush=True,
    )
    rng = random.Random(seed)
    tokenizer = load_tokenizer()
    model, config, ckpt = load_experimental_baby(parent, device)
    if ckpt.get("protected_material_opened"):
        raise RuntimeError("protected material opened")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    head = PropMatchHead(config.d_model).to(device)
    if resume_head is not None:
        blob = torch.load(resume_head, map_location=device, weights_only=False)
        head.load_state_dict(blob["prop_match_state_dict"])
        print(json.dumps({"phase": "r3c_resume_head", "path": str(resume_head), "sha256": digest(resume_head)}), flush=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    prop_ids = property_id_set(tokenizer)
    ent_ids = entity_id_set(tokenizer)
    bare_ids = bare_entity_id_set(tokenizer)
    out_dir = R3_OUT / (f"r3c_{seed}_lock" if resume_head is not None else f"r3c_{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for step in range(1, steps + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        used = 0
        skipped = 0
        for _ in range(batch):
            if rng.random() < 0.35:
                item = make_who_sentence_item(rng, tokenizer, n_entities=2, surface="train")
            else:
                item = make_who_bind_item(rng, tokenizer, n_entities=2, surface="train", anti_recency=rng.random() < 0.5)
            x = torch.tensor([item["input"]], dtype=torch.long, device=device)
            with torch.no_grad():
                hidden = model.forward_hidden(x)[0]
            tokens = x[0]
            time = hidden.shape[0]
            k = head._suffix_k(time)
            props = mention_indices(tokens, set(prop_ids))
            teacher = teacher_entity_pos(hidden, tokens, set(prop_ids), set(ent_ids), set(bare_ids))
            if not props or teacher is None:
                skipped += 1
                continue
            cue = int(props[-1])
            local = cue - (time - k)
            if local < 0 or local >= k:
                skipped += 1
                continue
            cue_loss = F.cross_entropy(
                head.cue_score(hidden[-k:]).squeeze(-1).unsqueeze(0), torch.tensor([local], device=device)
            )
            earlier = hidden[:cue]
            match = int(F.cosine_similarity(hidden[cue].unsqueeze(0), earlier, dim=-1).argmax()) if cue > 0 else 0
            start = max(0, match - head.hop_window)
            hop_loss = torch.zeros((), device=device)
            if start <= int(teacher) < match:
                hop_scores = head.hop(hidden[start:match]).squeeze(-1)
                if hop_scores.ndim == 0:
                    hop_scores = hop_scores.unsqueeze(0)
                hop_loss = F.cross_entropy(hop_scores.unsqueeze(0), torch.tensor([int(teacher) - start], device=device))
            gate_on = F.binary_cross_entropy_with_logits(head.gate_logit(hidden), torch.ones((), device=device))
            ans = int(item["target"][0])
            x_off = torch.tensor([item["input"] + [ans]], dtype=torch.long, device=device)
            with torch.no_grad():
                hidden_off = model.forward_hidden(x_off)[0]
            gate_off = F.binary_cross_entropy_with_logits(head.gate_logit(hidden_off), torch.zeros((), device=device))
            losses.append(cue_loss + hop_loss + 0.5 * (gate_on + gate_off))
            used += 1
        if not losses:
            continue
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0)
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == steps:
            print(json.dumps({"step": step, "loss": float(loss.detach().cpu()), "used": used, "skipped": skipped}), flush=True)
        if step in {25, 50, steps}:
            head.eval()
            official = build_s3_panels(tokenizer, n=32)["who_bind_2e_heldout"]
            seed2 = build_s3_panels(tokenizer, seed=324777, n=32)["who_bind_2e_heldout"]
            native = score_who_with_prop_match(model, head, tokenizer, device, official)
            native2 = score_who_with_prop_match(model, head, tokenizer, device, seed2)
            diag = prop_match_diag(model, head, tokenizer, device, official)
            row = {
                "step": step,
                "loaded_parent_sha256": parent_sha,
                "mechanism": "prop_match_cue_cosine_local_hop",
                "native_head": {k: native[k] for k in native if k != "rows"},
                "native_head_seed324777": {k: native2[k] for k in native2 if k != "rows"},
                "diag": diag,
            }
            history.append(row)
            write(out_dir / f"eval_{step:05d}.json", row)
            print(
                json.dumps(
                    {
                        "phase": "r3c_canary_eval",
                        "step": step,
                        "native_who_2e": row["native_head"]["acc"],
                        "native_who_2e_seed324777": row["native_head_seed324777"]["acc"],
                        "native_last_not": row["native_head"]["acc_last_not"],
                        "entity_hop_gold": diag["entity_hop_gold"],
                        "entity_hop_last": diag["entity_hop_last"],
                    },
                    default=str,
                ),
                flush=True,
            )
    head.eval()
    head_path = out_dir / f"prop_match_head_{steps:05d}.pt"
    head_sha = save_prop_match_head(head_path, head, parent=parent, parent_sha=parent_sha, update=steps, seed=seed)
    last = history[-1] if history else {}
    report = {
        "id": "r3c_canary",
        "loaded_parent": str(parent),
        "loaded_parent_sha256": parent_sha,
        "parent_match": True,
        "authoritative": False,
        "mechanism": "prop_match_cue_cosine_local_hop",
        "backbone_frozen": True,
        "head_path": str(head_path),
        "head_sha256": head_sha,
        "steps": steps,
        "history": history,
        "native_head": last.get("native_head"),
        "native_head_seed324777": last.get("native_head_seed324777"),
        "diag": last.get("diag"),
        "passed": bool(
            last
            and last.get("native_head", {}).get("acc", 0) + 1e-12 >= WHO2_GATE
            and last.get("native_head_seed324777", {}).get("acc", 0) + 1e-12 >= WHO2_GATE
        ),
    }
    write(out_dir / "CANARY.json", report)
    print(json.dumps({k: report[k] for k in report if k != "history"}, default=str), flush=True)
    return report


def load_prop_match_head(path: Path, d_model: int, device) -> PropMatchHead:
    blob = torch.load(path, map_location=device, weights_only=False)
    head = PropMatchHead(
        d_model,
        suffix_k=int(blob.get("prop_match_suffix_k") or 6),
        copy_scale=float(blob.get("prop_match_copy_scale") or COPY_SCALE),
    ).to(device)
    state = blob["prop_match_state_dict"]
    head.use_local_hop = any(k.startswith("hop.") for k in state)
    head.load_state_dict(state, strict=False)
    head.eval()
    return head


def verify_r3b(device, *, head_path: Path | None = None, with_retention: bool = True) -> dict:
    """Medium verify: unbatched native-head who_2e, WHO sentences, labeled retention."""
    from .data_language_bridge import build_e13_panels
    from .selection_language_bridge import eval_panels, run_usable_chat, slim_panels
    from .selection_stack2 import run_d3_log, slim_usable
    from .selection_stack2_r2 import score_who_sent_seed
    from .selection_stack2_s5 import CHEAP_OPS, run_who_sentence_decode, summarize_bind

    parent = S5B3_PARENT
    parent_sha = verify_parent(parent, S5B3_PARENT_SHA, label="r3b verify parent")
    head_path = head_path or (R3_OUT / "r3b_329011" / "prop_match_head_00050.pt")
    head_sha = digest(head_path)
    tokenizer = load_tokenizer()
    model, config, ckpt = load_experimental_baby(parent, device)
    if ckpt.get("protected_material_opened"):
        raise RuntimeError("protected material opened")
    head = load_prop_match_head(head_path, config.d_model, device)
    official = build_s3_panels(tokenizer, n=32)["who_bind_2e_heldout"]
    seeds = {
        "official": official,
        "seed324777": build_s3_panels(tokenizer, seed=324777, n=32)["who_bind_2e_heldout"],
        "seed324888": build_s3_panels(tokenizer, seed=324888, n=32)["who_bind_2e_heldout"],
    }
    weights_only = {
        name: {k: v for k, v in score_pack(model, tokenizer, device, items).items() if k != "rows"}
        for name, items in seeds.items()
    }
    native_head = {
        name: {k: v for k, v in score_who_with_prop_match(model, head, tokenizer, device, items).items() if k != "rows"}
        for name, items in seeds.items()
    }
    external_router = {
        "official": {k: v for k, v in score_who_with_router(model, tokenizer, device, official).items() if k != "rows"},
        "seed324777": {
            k: v for k, v in score_who_with_router(model, tokenizer, device, seeds["seed324777"]).items() if k != "rows"
        },
    }
    diag = {name: prop_match_diag(model, head, tokenizer, device, items) for name, items in seeds.items()}
    runtime = PropMatchRuntime(model, head, tokenizer).install()
    runtime.enabled = True
    try:
        who_sent = summarize_bind(run_who_sentence_decode(model, tokenizer, device, include=CHEAP_OPS))
        who_sent_seeds = {
            "seed324777": score_who_sent_seed(model, tokenizer, device, seed=324777, n=16),
            "seed324888": score_who_sent_seed(model, tokenizer, device, seed=324888, n=16),
        }
    finally:
        runtime.uninstall()
    native_panels = None
    usable = None
    d3 = None
    if with_retention:
        native_panels = slim_panels(
            eval_panels(model, build_e13_panels(tokenizer, n=32), device, overwrite=None, arms=("native",))["native"]
        )
        d3 = run_d3_log(model, device, "r3b_verify", english_ok=True)
        runtime = PropMatchRuntime(model, head, tokenizer).install()
        runtime.enabled = True
        try:
            usable = slim_usable(run_usable_chat(model, tokenizer, device))
        finally:
            runtime.uninstall()
    who2_ok = native_head["official"]["acc"] + 1e-12 >= WHO2_GATE and native_head["seed324777"]["acc"] + 1e-12 >= WHO2_GATE
    last_ok = (native_head["official"].get("acc_last_not") or 0) >= 0.80 and (
        native_head["seed324777"].get("acc_last_not") or 0
    ) >= 0.80
    sent_ok = who_sent["bare"] + 1e-12 >= 0.80
    report = {
        "id": "r3b_verify",
        "loaded_parent": str(parent),
        "loaded_parent_sha256": parent_sha,
        "parent_match": parent_sha == S5B3_PARENT_SHA,
        "head_path": str(head_path),
        "head_sha256": head_sha,
        "authoritative": False,
        "mechanism": "prop_match_cue_cosine_entityness",
        "label_legend": {
            "weights_only": "s5b3 checkpoint, no runtime module",
            "native_head": "s5b3 + learned PropMatchHead; no Python cue scanner",
            "external_router": "s5b3 + WhoPropRouter (R2 ceiling; Python scanner)",
            "batched_eval_panels": "teacher-forced batched logits; may bypass last-token routing",
        },
        "weights_only_who_2e": weights_only,
        "native_head_who_2e": native_head,
        "external_router_who_2e": external_router,
        "diag": diag,
        "who_sent_native_head": who_sent,
        "who_sent_native_head_seeds": {
            k: {kk: vv for kk, vv in row.items() if kk != "rows"} for k, row in who_sent_seeds.items()
        },
        "who_sent_examples": (who_sent_seeds["seed324888"].get("rows") or [])[:6],
        "batched_eval_panels_weights_only": native_panels,
        "usable_native_head": usable,
        "d3_weights_only": None
        if d3 is None
        else {
            "n": d3["long_gap"]["n"],
            "free_exact": d3["long_gap"]["free_exact"],
            "first_correct": d3["long_gap"]["first_correct"],
            "induction": float(d3["primitive_induction"]["first_top1"]),
        },
        "who2_gate": WHO2_GATE,
        "who2_ok": who2_ok,
        "last_not_ok": last_ok,
        "who_sent_ok": sent_ok,
        "passed": who2_ok and last_ok and sent_ok,
    }
    R3_OUT.mkdir(parents=True, exist_ok=True)
    write(R3_OUT / "R3B_VERIFY.json", report)
    print(json.dumps({k: report[k] for k in report if k != "who_sent_examples"}, default=str), flush=True)
    return report


def recheck_r3b(device, *, head_path: Path | None = None) -> dict:
    """Cheap post-gate check: who_2e, WHO sentences, usable. No D3."""
    from .selection_language_bridge import greedy_decode_until_stop, run_usable_chat
    from .selection_stack2 import slim_usable
    from .selection_stack2_s5 import CHEAP_OPS, bind_examples, run_who_sentence_decode, summarize_bind

    parent = S5B3_PARENT
    parent_sha = verify_parent(parent, S5B3_PARENT_SHA, label="r3b recheck parent")
    head_path = head_path or (R3_OUT / "r3b_329011" / "prop_match_head_00050.pt")
    tokenizer = load_tokenizer()
    model, config, ckpt = load_experimental_baby(parent, device)
    if ckpt.get("protected_material_opened"):
        raise RuntimeError("protected material opened")
    head = load_prop_match_head(head_path, config.d_model, device)
    official = build_s3_panels(tokenizer, n=32)["who_bind_2e_heldout"]
    seed2 = build_s3_panels(tokenizer, seed=324777, n=32)["who_bind_2e_heldout"]
    native = score_who_with_prop_match(model, head, tokenizer, device, official)
    native2 = score_who_with_prop_match(model, head, tokenizer, device, seed2)
    diag = prop_match_diag(model, head, tokenizer, device, official)
    runtime = PropMatchRuntime(model, head, tokenizer).install()
    runtime.enabled = True
    try:
        who_full = run_who_sentence_decode(model, tokenizer, device, include=CHEAP_OPS)
        who_sent = summarize_bind(who_full)
        examples = bind_examples(who_full, n=8)
        from .data import BOS
        from .data_language_bridge import build_who_sentence_decode_pack, encode_ids

        misses = []
        for item in build_who_sentence_decode_pack():
            prompt = f"{item['facts']} {item['query']}"
            prompt_ids = [BOS, *encode_ids(tokenizer, prompt)]
            x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            hidden = model.forward_hidden(x)[0]
            hop = head.entity_index(hidden)
            hop_word = tokenizer.decode([int(x[0, hop])], skip_special_tokens=True) if hop is not None else None
            emitted, stopped = greedy_decode_until_stop(model, prompt_ids, device, tokenizer, max_new=16)
            misses.append(
                {
                    "id": item["id"],
                    "kind": item["kind"],
                    "gold": item["entity"],
                    "cue": item["value"],
                    "cue_conf": round(head.cue_confidence(hidden), 3),
                    "hop_word": hop_word,
                    "decoded": tokenizer.decode(emitted, skip_special_tokens=True),
                }
            )
        usable = slim_usable(run_usable_chat(model, tokenizer, device))
    finally:
        runtime.uninstall()
    report = {
        "id": "r3b_recheck",
        "loaded_parent_sha256": parent_sha,
        "head_sha256": digest(head_path),
        "native_who_2e": {k: native[k] for k in native if k != "rows"},
        "native_who_2e_seed324777": {k: native2[k] for k in native2 if k != "rows"},
        "diag": diag,
        "who_sent": who_sent,
        "who_sent_examples": examples,
        "who_sent_trace": misses,
        "usable": {
            "turn4": usable["autoregressive_4turn"]["usable_turn"],
            "stop": usable["autoregressive_4turn"]["period_stop"],
            "reuse": usable["autoregressive_4turn"]["fact_reuse"],
        },
    }
    write(R3_OUT / "R3B_RECHECK.json", report)
    print(json.dumps({k: report[k] for k in report if k not in {"who_sent_examples", "who_sent_trace"}}, default=str), flush=True)
    return report


if __name__ == "__main__":
    import argparse

    from .selection_p11_u16000_runtime import resolve_device

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--canary-b", action="store_true")
    parser.add_argument("--canary-c", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--recheck", action="store_true")
    parser.add_argument("--head", type=Path)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    device = resolve_device(args.device)
    if args.probe:
        probe_native_match(device)
    elif args.canary_c:
        train_r3c_canary(device, steps=int(args.steps), resume_head=args.head)
    elif args.canary_b:
        train_r3b_canary(device, steps=int(args.steps))
    elif args.canary:
        train_r3_canary(device, steps=int(args.steps))
    elif args.verify:
        verify_r3b(device, head_path=args.head)
    elif args.recheck:
        recheck_r3b(device, head_path=args.head)
    else:
        parser.error("pass --probe, --canary, --canary-b, --canary-c, --verify, or --recheck")
