"""Read-only DEV audit of T31 source-copy logits.

This script deliberately refuses the sealed TEST path.  It loads only the two
terminal T31 checkpoints and the authorized corrected T3 DEV records, then
records aggregate source selection, ordered copy alignment, and final-logit
competition at the actual free-running decision states.
"""
from __future__ import annotations

import hashlib, json, sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(r"C:\DaveLM-CADAVER")
BUNDLE = ROOT / "phase2a_t31_native_source_copy_bridge_v1"
T3_DATA = ROOT / "phase2a_t3_rebuilt_study_v1" / "data"
ARCH = ROOT / "baby_vnext_60m_design_v1"
PARENT_SHA = "c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1"
TEST_PATH = T3_DATA / "qa_test.jsonl"
SEEDS = (880001, 880002)
EOS = 3
PERIOD = 18

sys.path[:0] = [str(BUNDLE), str(ARCH)]
from T31_COPY import T31CopyModel  # noqa: E402
from baby_vnext.config import BabyVNextConfig  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    # Deliberate protected-data guard before any file access.
    if path.resolve() == TEST_PATH.resolve():
        raise RuntimeError("protected TEST is forbidden")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def spans(row: dict) -> list[list[int]]:
    prompt = [2] + [int(x) for x in row["prompt_token_ids"]]
    out = []
    for mention in row["mention_span_bos"]:
        start, end = int(mention[0]), int(mention[-1])
        for i in range(end, len(prompt)):
            if prompt[i] == PERIOD:
                end = i
                break
        else:
            raise RuntimeError(f"missing period: {row['id']}")
        out.append(list(range(start, end + 1)))
    return out


def forks(rows: list[dict]) -> dict[str, int | None]:
    names = sorted({r["correct_name"] for r in rows})
    seq = {}
    for r in rows:
        ci = int(r["correct_index"])
        seq[r["correct_name"]] = [int(x) for x in r["candidate_token_ids"][ci]]
    ans = {}
    for name, toks in seq.items():
        found = None
        for k in range(1, len(toks)):
            pref = tuple(toks[:k])
            prior = [n for n, tt in seq.items() if tuple(tt[:k]) == pref]
            after = [n for n, tt in seq.items() if tuple(tt[: k + 1]) == tuple(toks[: k + 1])]
            if len(prior) >= 2 and len(after) == 1 and after[0] == name:
                found = k
                break
        ans[name] = found
    return ans


def rank_of(logits: torch.Tensor, token: int) -> int:
    return int((logits > logits[int(token)]).sum().item()) + 1


def load(seed: int, device: torch.device):
    ckpt = BUNDLE / f"run_seed{seed}" / "checkpoints" / "checkpoint_1000.pt"
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = T31CopyModel(BabyVNextConfig.load(ARCH / "BABY_VNEXT_CONFIG.json"))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval().to(device)
    return model, sha256(ckpt)


def agg(records: list[dict]) -> dict:
    n = len(records)
    if not n:
        return {"n": 0}
    mean = lambda key: sum(float(x[key]) for x in records) / n
    rate = lambda key: sum(bool(x[key]) for x in records) / n
    return {
        "n": n,
        "final_target_top1_rate": rate("final_target_top1"),
        "lm_target_top1_rate": rate("lm_target_top1"),
        "final_mean_target_rank": mean("final_target_rank"),
        "lm_mean_target_rank": mean("lm_target_rank"),
        "pointer_correct_rate": rate("pointer_correct"),
        "copy_target_top_rate": rate("copy_target_top"),
        "mean_pointer_correct_weight": mean("pointer_correct_weight"),
        "mean_copy_target_mass": mean("copy_target_mass"),
        "mean_copy_target_delta": mean("copy_target_delta"),
        "mean_copy_argmax_delta": mean("copy_argmax_delta"),
        "mean_final_target_logit": mean("final_target_logit"),
        "mean_final_argmax_logit": mean("final_argmax_logit"),
        "mean_lm_target_logit": mean("lm_target_logit"),
        "mean_lm_argmax_logit": mean("lm_argmax_logit"),
        "mean_final_target_vs_argmax_margin": mean("final_target_vs_argmax_margin"),
        "mean_lm_target_vs_argmax_margin": mean("lm_target_vs_argmax_margin"),
        "generated_target_rate": rate("generated_target"),
    }


@torch.no_grad()
def run_seed(seed: int, rows: list[dict], fork: dict, device: torch.device):
    model, cksha = load(seed, device)
    gate = [float(x) for x in model.copy_gate.detach().cpu().tolist()]
    all_steps, fork_steps, fork_wrong, fork_correct, first_diverge = [], [], [], [], []
    exact = 0
    for row in rows:
        prompt = [2] + [int(x) for x in row["prompt_token_ids"]]
        plen = len(prompt)
        ci = int(row["correct_index"])
        target = [int(x) for x in row["candidate_token_ids"][ci]] + [EOS]
        fidx = fork[row["correct_name"]]
        generated, mismatch = [], None
        for step in range(min(len(target) + 2, 8)):
            inp = torch.tensor([prompt + generated], dtype=torch.long, device=device)
            hidden = model.base_model.forward_hidden(inp)
            _, weights, _ = model.pointer_route(hidden, spans(row), plen - 1)
            cp = model.copy_probs_for_row(row, weights, device, model.base_model.config.vocab_size)
            lm = model.base_model.language_head(hidden)[0, -1]
            final = model.logits_with_copy(hidden, cp, plen - 1)[0, -1]
            emitted = int(final.argmax().item())
            if step >= len(target):
                break
            gold = int(target[step])
            cp_row = cp[step] if step < cp.shape[0] else torch.zeros_like(final)
            # Additive path contribution; exact zero after the four copy steps.
            delta = final - lm
            rec = {
                "id": row["id"], "name": row["correct_name"], "step": step,
                "fork_step": fidx, "shared_prefix": fidx is not None,
                "target": gold, "emitted": emitted,
                "generated_target": emitted == gold,
                "final_target_top1": int(final.argmax()) == gold,
                "lm_target_top1": int(lm.argmax()) == gold,
                "final_target_rank": rank_of(final, gold),
                "lm_target_rank": rank_of(lm, gold),
                "pointer_correct": int(weights.argmax()) == ci,
                "pointer_correct_weight": float(weights[ci]),
                "copy_target_mass": float(cp_row[gold]),
                "copy_target_top": int(cp_row.argmax()) == gold if float(cp_row.sum()) else False,
                "copy_target_delta": float(delta[gold]),
                "copy_argmax_delta": float(delta[emitted]),
                "final_target_logit": float(final[gold]),
                "final_argmax_logit": float(final[emitted]),
                "lm_target_logit": float(lm[gold]),
                "lm_argmax_logit": float(lm[emitted]),
                "final_target_vs_argmax_margin": float(final[gold] - final[emitted]),
                "lm_target_vs_argmax_margin": float(lm[gold] - lm[emitted]),
                "gate": gate[step] if step < len(gate) else 0.0,
            }
            all_steps.append(rec)
            if fidx is not None and step == fidx:
                fork_steps.append(rec)
                (fork_correct if emitted == gold else fork_wrong).append(rec)
            if mismatch is None and emitted != gold:
                first_diverge.append(rec)
                mismatch = step
            generated.append(emitted)
            if emitted == EOS:
                break
        if generated == target:
            exact += 1
    def modes(recs):
        return dict(sorted(Counter((r["pointer_correct"], r["copy_target_top"], r["final_target_top1"]) for r in recs).items()))
    return {
        "checkpoint_sha256": cksha, "copy_gate": gate, "exact": exact,
        "all_answer_steps": agg(all_steps),
        "shared_fork_all": agg(fork_steps),
        "shared_fork_emitted_correct": agg(fork_correct),
        "shared_fork_emitted_wrong": agg(fork_wrong),
        "first_divergence": agg(first_diverge),
        "shared_fork_joint_pointer_copy_final": {str(k): v for k, v in modes(fork_steps).items()},
    }


def main():
    if TEST_PATH.exists() is False:
        # Absence is irrelevant, but this prevents accidentally treating a missing seal as permission.
        pass
    rows = load_jsonl(T3_DATA / "qa_dev.jsonl")
    if len(rows) != 128:
        raise RuntimeError(f"DEV count mismatch: {len(rows)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fork = forks(rows)
    if sum(v is not None for v in fork.values()) != 4:
        raise RuntimeError("frozen shared-prefix population mismatch")
    payload = {
        "study": "POST_T31_DEV_ONLY_LOGIT_COMPETITION_AUDIT_V1",
        "read_only": True,
        "allowed": ["qa_dev.jsonl", "T31 terminal checkpoints", "architecture/config", "T31 copy source"],
        "forbidden": ["qa_test.jsonl", "T3 TEST", "FINAL", "sacred"],
        "test_loaded": False, "device": str(device), "dev_rows": len(rows),
        "shared_prefix_names": sorted(n for n, v in fork.items() if v is not None),
        "results": {str(seed): run_seed(seed, rows, fork, device) for seed in SEEDS},
    }
    print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
