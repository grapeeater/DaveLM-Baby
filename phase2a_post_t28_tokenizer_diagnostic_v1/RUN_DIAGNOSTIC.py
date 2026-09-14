"""DEV-only post-T28 diagnostics. Never opens or references protected panels."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from array import array
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(r"C:\DaveLM-CADAVER")
ARCH = ROOT / "baby_vnext_60m_design_v1"
sys.path.insert(0, str(ARCH))
DEV_PATH = ROOT / "phase2a_t3_rebuilt_study_v1" / "data" / "qa_dev.jsonl"
TOK_PATH = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")
TOK_SHA = "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"
PARENT_SHA = "c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1"
SEEDS = (850001, 850002, 850003)
EOS, CTX = 3, 256


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x]


def write(p: Path, obj: object) -> None:
    p.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def load_model(seed: int, device: torch.device):
    from baby_vnext.config import BabyVNextConfig
    from baby_vnext.binding import BabyVNextWithBinding
    ckpt = ROOT / f"phase2a_t28_fast_v2_run_seed{seed}" / "rolling_restart.pt"
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = BabyVNextWithBinding(BabyVNextConfig.load(ARCH / "BABY_VNEXT_CONFIG.json"))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, ckpt


def logits(model, ids: list[int], device: torch.device) -> torch.Tensor:
    out = model(torch.tensor([ids[-CTX:]], device=device))
    return out[0] if isinstance(out, tuple) else out


def greedy(model, row: dict, device: torch.device) -> tuple[list[int], list[int]]:
    prompt = [2] + list(map(int, row["prompt_token_ids"]))
    target = list(map(int, row["candidate_token_ids"][int(row["correct_index"])])) + [EOS]
    running, generated = list(prompt), []
    for _ in range(max(16, len(target) + 4)):
        nxt = int(logits(model, running, device)[-1].argmax())
        running.append(nxt); generated.append(nxt)
        if nxt == EOS:
            break
    return generated, target


@torch.no_grad()
def greedy_all(model, rows: list[dict], device: torch.device) -> dict[str, tuple[list[int], list[int]]]:
    """Same greedy rule as greedy(), batched only across equal-length prefixes."""
    states = {}
    for r in rows:
        prompt = [2] + list(map(int, r["prompt_token_ids"]))
        target = list(map(int, r["candidate_token_ids"][int(r["correct_index"])])) + [EOS]
        states[r["id"]] = {"run": prompt, "gen": [], "target": target,
                           "limit": max(16, len(target) + 4), "done": False}
    while any(not s["done"] for s in states.values()):
        groups = defaultdict(list)
        for rid, s in states.items():
            if not s["done"]:
                groups[len(s["run"][-CTX:])].append((rid, s))
        for _, group in groups.items():
            batch = torch.tensor([s["run"][-CTX:] for _, s in group], device=device)
            out = model(batch); out = out[0] if isinstance(out, tuple) else out
            nxt = out[:, -1].argmax(dim=-1).tolist()
            for (rid, s), token in zip(group, nxt):
                token = int(token); s["run"].append(token); s["gen"].append(token)
                if token == EOS or len(s["gen"]) >= s["limit"]: s["done"] = True
    return {rid: (s["gen"], s["target"]) for rid, s in states.items()}


def classify(generated: list[int], target: list[int], wrongs: list[list[int]]) -> str:
    if generated == target:
        return "exact"
    if not generated or generated[0] == EOS:
        return "immediate_eos"
    if generated[0] == target[0]:
        return "first_token_correct_then_diverge"
    if any(generated == w for w in wrongs):
        return "wrong_candidate"
    if EOS in generated:
        return "other_then_eos"
    return "unterminated_other"


def language_counts() -> Counter:
    p = ROOT / "baby_vnext_phase1g_language_v1" / "data" / "LANGUAGE_TRAIN_STREAM.u16"
    raw = array("H")
    with p.open("rb") as f:
        raw.fromfile(f, p.stat().st_size // 2)
    return Counter(raw)


def prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return n


def odds(a_yes: int, a_no: int, b_yes: int, b_no: int) -> float | None:
    if min(a_yes, a_no, b_yes, b_no) == 0: return None
    return (a_yes * b_no) / (a_no * b_yes)


def tokenizer_diag(out: Path) -> None:
    tok = Tokenizer.from_file(str(TOK_PATH))
    dev = load_jsonl(DEV_PATH)
    if sha(TOK_PATH) != TOK_SHA: raise RuntimeError("tokenizer hash mismatch")
    counts = language_counts()
    all_names = sorted({n for r in dev for n in r["candidates"]})
    name_ids = {n: tok.encode(" " + n).ids for n in all_names}
    items: list[dict] = []
    device = torch.device("cuda")
    for seed in SEEDS:
        model, checkpoint = load_model(seed, device)
        generated = greedy_all(model, dev, device)
        for r in dev:
            gen, target = generated[r["id"]]
            ci = int(r["correct_index"])
            wrongs = [list(map(int, c))+[EOS] for i,c in enumerate(r["candidate_token_ids"]) if i != ci]
            name = r["correct_name"]; ids = list(map(int, r["candidate_token_ids"][ci]))
            peers = [x for x in all_names if x != name]
            overlap = max([prefix(ids, name_ids[x]) for x in peers], default=0)
            short_prefix = any(name_ids[x] == ids[:len(name_ids[x])] and len(name_ids[x]) < len(ids) for x in peers)
            mode = classify(gen, target, wrongs)
            items.append({"seed":seed,"id":r["id"],"name":name,"name_token_ids":name_ids[name],"target_ids":target,
                          "token_pieces":[tok.decode([t]) for t in ids],"answer_text":tok.decode(ids),
                          "generated_ids":gen,"generated_text":tok.decode(gen),"name_tokens":len(name_ids[name]),"answer_tokens_with_period":len(ids),
                          "answer_tokens_with_eos":len(target),"max_shared_prefix_tokens":overlap,"shorter_name_prefix":short_prefix,
                          "mean_train_frequency":sum(counts[t] for t in ids)/len(ids),"min_train_frequency":min(counts[t] for t in ids),
                          "exact":mode=="exact","first_token_correct":bool(gen) and gen[0]==target[0],"mode":mode,
                          "checkpoint_sha256":sha(checkpoint)})
        del model; torch.cuda.empty_cache()
    def rate(rows, key="exact"):
        return sum(bool(x[key]) for x in rows)/len(rows) if rows else None
    summary = {"n":len(items),"by_seed":{},"by_name_token_count":{},"by_shared_prefix":{},"modes":{},"effect_sizes":{}}
    for s in SEEDS:
        rows=[x for x in items if x["seed"]==s]; summary["by_seed"][str(s)]={"n":len(rows),"exact_rate":rate(rows),"first_correct_rate":rate(rows,"first_token_correct"),"modes":dict(Counter(x["mode"] for x in rows))}
    for k in sorted({x["name_tokens"] for x in items}):
        rows=[x for x in items if x["name_tokens"]==k]; summary["by_name_token_count"][str(k)]={"n":len(rows),"exact_rate":rate(rows),"diverge_rate":sum(x["mode"]=="first_token_correct_then_diverge" for x in rows)/len(rows)}
    for flag in (False,True):
        rows=[x for x in items if (x["max_shared_prefix_tokens"]>0)==flag]; summary["by_shared_prefix"][str(flag)]={"n":len(rows),"exact_rate":rate(rows),"diverge_rate":sum(x["mode"]=="first_token_correct_then_diverge" for x in rows)/len(rows)}
    summary["modes"]=dict(Counter(x["mode"] for x in items))
    shared=[x for x in items if x["max_shared_prefix_tokens"]>0]; unshared=[x for x in items if x["max_shared_prefix_tokens"]==0]
    summary["effect_sizes"]["shared_prefix_exact_odds_ratio"] = odds(sum(x["exact"] for x in shared),sum(not x["exact"] for x in shared),sum(x["exact"] for x in unshared),sum(not x["exact"] for x in unshared))
    label="TOKENIZER_ASSOCIATION_WEAK_OR_MIXED"
    if shared and unshared and abs(summary["by_shared_prefix"]["True"]["exact_rate"]-summary["by_shared_prefix"]["False"]["exact_rate"])>=0.10: label="TOKENIZER_ASSOCIATION_SUPPORTED"
    if not shared or not unshared: label="TOKENIZER_ASSOCIATION_NOT_SUPPORTED"
    summary["classification"]=label
    write(out/"ITEMS.json",items); write(out/"SUMMARY.json",summary); write(out/"STATUS.json",{"status":"COMPLETE","classification":label,"test_loaded":False})


def layer_logits(base, ids: list[int], device: torch.device) -> list[torch.Tensor]:
    x=torch.tensor([ids[-CTX:]],device=device); t=x.shape[1]; pos=torch.arange(t,device=device)
    h=base.embedding_dropout(base.token_embedding(x)+base.position_embedding(pos)); out=[h]
    for b in base.blocks: h=b(h); out.append(h)
    # Apply Baby's actual final normalization before the frozen unembedding at
    # every logit-lens site. The final entry is therefore the native final path.
    return [base.language_head(base.final_norm(z)[0,-1]).detach().cpu() for z in out]


def suffix_trace(out: Path) -> None:
    dev=load_jsonl(DEV_PATH); device=torch.device("cuda"); records=[]; totals=defaultdict(lambda: {"n":0,"top1":0,"mean_rank":0.0})
    for seed in SEEDS:
        model, checkpoint=load_model(seed,device); base=model.base_model
        generated=greedy_all(model,dev,device)
        for r in dev:
            gen,target=generated[r["id"]]
            if not (len(gen)>1 and gen[0]==target[0] and gen!=target): continue
            prompt=[2]+list(map(int,r["prompt_token_ids"]))
            for step in range(1,min(len(target),len(gen))):
                want=target[step]
                for mode,prefix in (("teacher",prompt+target[:step]),("free",prompt+gen[:step])):
                    layers=layer_logits(base,prefix,device); rows=[]
                    for li,z in enumerate(layers):
                        rank=int((z > z[want]).sum().item() + 1)
                        top=int(z.argmax().item()); rows.append({"layer":li,"target_token":want,"target_logit":float(z[want]),"target_rank":rank,"target_top1":top==want,"top_token":top,"top_logit":float(z[top]),"margin_vs_top":float(z[want]-z[top])})
                        key=f"{mode}:layer{li}"; totals[key]["n"]+=1; totals[key]["top1"]+=int(top==want); totals[key]["mean_rank"]+=rank
                    records.append({"seed":seed,"id":r["id"],"step":step,"target_token":want,"generated_prefix":gen[:step],"target_prefix":target[:step],"mode":mode,"layers":rows,"checkpoint_sha256":sha(checkpoint)})
        del model; torch.cuda.empty_cache()
    summary={k:{"n":v["n"],"top1_rate":v["top1"]/v["n"],"mean_rank":v["mean_rank"]/v["n"]} for k,v in totals.items()}
    write(out/"TRACE.json",records); write(out/"SUMMARY.json",summary); write(out/"STATUS.json",{"status":"COMPLETE","test_loaded":False,"records":len(records)})


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("mode",choices=("tokenizer","suffix")); ap.add_argument("out",type=Path); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    if a.mode=="tokenizer": tokenizer_diag(a.out)
    else: suffix_trace(a.out)

if __name__=="__main__": main()
