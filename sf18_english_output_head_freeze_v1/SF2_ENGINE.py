"""SF2 frozen single-fact acquisition controller with KL-to-parent retention.

Derived from the authoritative SF1 ENGINE.py (single_fact_acquisition_sf1_seed87011)
with the ONLY scientific change in the English-loss branch (add lambda*KL) plus the
frozen M1-M10 additions (teacher, retention pool, D3 diagnostic, additive update-100
acquisition guard). All SF1 mechanics (schedule, scope, optimizer, commit, eval,
guards, transfer gating, finalize) are otherwise unchanged. No sampling during
execution. No head-freeze fallback exists. lambda is fixed at 1.0.
"""
import argparse, json, hashlib, sys, os, random, math, tempfile
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "sources"))
import hr3_block3_runtime as rt
from PINNED_MASKING import set_scope, prepare_example, pad_batch

LAMBDA_KL = 1.0
KL_POS_PER_UPDATE = 160

SF2_PROTOCOL = HERE / "SF2_PROTOCOL.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def verify(b):
    assert sha(b / "RECEIPT.json") == (b / "RECEIPT.sha256").read_text().split()[0]
    assert sha(b / "SHA256SUMS.txt") == read(b / "RECEIPT.json")["manifest_sha256"]
    for line in (b / "SHA256SUMS.txt").read_text().splitlines():
        h, n = line.split("  ", 1)
        assert sha(b / n) == h, n
    p = read(b / "PROTOCOL.json")
    for path, h in p["external_hashes"].items():
        assert sha(path) == h, path
    assert sha(p["parent"]) == p["parent_sha256"]
    assert sha(p["tokenizer"]) == p["tokenizer_sha256"]
    return p


def scope_audit(m):
    out = {}
    for binding in [False, True]:
        set_scope(m, binding)
        on, off = [], []
        for n, x in m.named_parameters():
            expected = binding or (n.startswith("base_model.") and not any(n.startswith(f"base_model.blocks.{i}.") for i in range(4)))
            assert x.requires_grad == expected and x.grad is None
            (on if expected else off).append(n)
        out["binding" if binding else "english"] = {"active": on, "frozen": off}
    return out


def build_teacher(parent: Path, device):
    """M1: separate frozen DaveLMV082 teacher from parent base_model.* keys."""
    from v0_8_2.model import build_model as _build
    teacher = _build("untied").to(device)
    raw = torch.load(parent, map_location="cpu", weights_only=False)
    sd = {k[len("base_model."):]: v for k, v in raw["model_state_dict"].items() if k.startswith("base_model.")}
    missing, unexpected = teacher.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def assert_teacher_disjoint(student_base, teacher):
    pairs = [("token_embedding.weight", student_base.token_embedding.weight, teacher.token_embedding.weight),
             ("language_head.weight", student_base.language_head.weight, teacher.language_head.weight),
             ("language_head.bias", student_base.language_head.bias, teacher.language_head.bias)]
    for name, a, b in pairs:
        assert a is not b, name
        assert a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr(), name


def load_kl_pool():
    pool_path = HERE / "KL_POOL.json"
    man_path = HERE / "KL_POOL_MANIFEST.json"
    proto = read(SF2_PROTOCOL)
    assert sha(pool_path) == proto["kl"]["pool_sha256"], "KL_POOL.json hash mismatch vs SF2_PROTOCOL"
    assert sha(man_path) == proto["kl"]["pool_manifest_sha256"], "KL_POOL_MANIFEST.json hash mismatch"
    pool = read(pool_path)
    assert pool["count"] == len(pool["entries"]) and len(pool["rows"]) > 0
    assert pool["count"] >= KL_POS_PER_UPDATE * 180
    return pool, proto


def pick_kl_entries(entries, english_index):
    """SF13: broad-coverage selection from the prospectively frozen SF13_KL_SCHEDULE.json.

    For English update u in 1..180, return the 160 frozen [row, col] entries mapped
    through the actual KL_POOL entries list. Replaces SF11/SF12 contiguous
    sequential 160-entry slicing (retention position coverage / sampling geometry
    is the single SF13 scientific variable). JSON object keys are strings.
    """
    n = len(entries)
    schedule_data = read(HERE / 'SF13_KL_SCHEDULE.json')
    sf13_schedule = schedule_data['schedule']
    key = str(english_index)
    if english_index in sf13_schedule:
        sel = sf13_schedule[english_index]
    elif key in sf13_schedule:
        sel = sf13_schedule[key]
    else:
        raise KeyError(f'English update {english_index} not in frozen SF13 schedule')
    entry_map = {(e[0], e[1]): i for i, e in enumerate(entries)}
    out = [entries[entry_map[tuple(e)]] for e in sel]
    assert len(out) == KL_POS_PER_UPDATE
    return out


def compute_kl(m, teacher, device, pool_rows, sel):
    """D_KL(P_student || P_teacher) over the selected ordinary positions (M3/M4)."""
    # Group by pool row index, preserving order of first appearance in sel.
    order = []
    byrow = {}
    for ri, col in sel:
        if ri not in byrow:
            byrow[ri] = []
            order.append(ri)
        byrow[ri].append(col)
    ref = {ri: b for b, ri in enumerate(order)}
    sub_rows = [{"token_ids": pool_rows[ri]} for ri in order]
    x, _ = rt.aligned_tensors(sub_rows, device)
    ls = m.base_model(x)                      # grad enabled, eval mode (dropout off)
    with torch.no_grad():
        lt = teacher(x)
    rows_idx = torch.tensor([ref[ri] for (ri, _) in sel], device=device, dtype=torch.long)
    cols = torch.tensor([c for (_, c) in sel], device=device, dtype=torch.long)
    ls_sel = ls[rows_idx, cols].double()      # [160,1024]
    lt_sel = lt[rows_idx, cols].double()
    ps = ls_sel.softmax(-1)
    kl_pos = (ps * (ls_sel.log_softmax(-1) - lt_sel.log_softmax(-1))).sum(-1)
    return kl_pos.mean().float()


def ranks_from_logits(logits, ids):
    return {int(i): int((logits > logits[int(i)]).sum().item()) + 1 for i in ids}


def measure_d3(m, selection, tok, device):
    """Exact D3 scorer from sf1_readout_selection_forensic_v1 (score_d3)."""
    ids = [x["token_id"] for x in selection["controls"]]
    name_ids = [int(v[0]) for v in selection["name_tokenizations"].values()]
    all_ids = list(dict.fromkeys(name_ids + ids))
    m.base_model.eval()
    rows = []
    with torch.inference_mode():
        for item in selection["positions"]:
            x = torch.tensor([[2] + item["prefix_token_ids"]], dtype=torch.long, device=device)
            logits = m.base_model(x)[0, -1].float()
            logp = logits.log_softmax(-1)
            p = logp.exp()
            ranks = ranks_from_logits(logits, all_ids + [3, item["target_token_id"]])
            nm = {}
            for name, seq in selection["name_tokenizations"].items():
                tid = int(seq[0])
                nm[name] = {"token_id": tid, "text": tok.decode([tid]), "logit": float(logits[tid]),
                            "probability": float(p[tid]), "rank": ranks[tid]}
            rows.append({
                "position_id": item["position_id"],
                "names": nm,
                "combined_name_probability": float(sum(p[int(seq[0])] for seq in selection["name_tokenizations"].values())),
                "eos_probability": float(p[3]),
                "true_next_probability": float(p[int(item["target_token_id"])]),
                "entropy": float(-(p * logp).sum()),
            })
    return rows


def d3_summary(rows):
    n = len(rows)
    return {
        "n": n,
        "mean_combined_name_probability": sum(r["combined_name_probability"] for r in rows) / n,
        "mean_eos_probability": sum(r["eos_probability"] for r in rows) / n,
        "mean_true_next_probability": sum(r["true_next_probability"] for r in rows) / n,
        "mean_entropy": sum(r["entropy"] for r in rows) / n,
        "per_name_mean_probability": {
            name: sum(r["names"][name]["probability"] for r in rows) / n for name in rows[0]["names"]
        },
    }


@torch.no_grad()
def score(m, r, tok, device):
    prefix = [2] + r["prompt_token_ids"]
    scores, details, first = [], [], {}
    ci = r["correct_index"]
    for cand in r["candidate_token_ids"]:
        z = m.base_model(torch.tensor([prefix + cand + [3]], device=device))[0]
        lp = z.log_softmax(-1)
        k = len(prefix) - 1
        vals = [float(lp[k + j, t]) for j, t in enumerate(cand + [3])]
        scores.append(sum(vals[:-1]))
        details.append(vals)
        if not first:
            fids = [c[0] for c in r["candidate_token_ids"]]
            first = {"eos_probability": float(lp[k, 3].exp()), "top1": int(z[k].argmax()),
                     "first_candidate_mass": float(lp[k, fids].exp().sum())}
    gen = rt.greedy_ids(m, r["prompt_token_ids"], device, 32)[len(prefix):]
    target = r["candidate_token_ids"][ci]
    margin = scores[ci] - scores[1 - ci]
    return {"id": r["id"], "family_id": r["family_id"], "pair_id": r["pair_id"], "subgroup": r["subgroup"],
            "prompt": r["prompt"], "candidates": r["candidates"], "correct_index": ci, "scores": scores,
            "token_scores_including_eos": details, "margin": margin, "correct": margin > 0,
            "mass": sum(math.exp(s) for s in scores), "exact": gen == target + [3], "generated_ids": gen,
            "generated_text": tok.decode(gen, skip_special_tokens=True), **first}


def summarize(rows):
    f, p = defaultdict(list), defaultdict(list)
    for r in rows:
        f[r["family_id"]].append(r)
        p[r["pair_id"]].append(r)
    assert all(len(v) == 2 for v in p.values())
    return {"n": len(rows), "correct": sum(r["correct"] for r in rows), "ties": sum(r["margin"] == 0 for r in rows),
            "exact": sum(r["exact"] for r in rows),
            "reversals": sum(all(r["correct"] for r in v) for v in p.values()), "pairs": len(p),
            "families": sum(all(r["correct"] for r in v) for v in f.values()), "family_count": len(f),
            "mean_margin": sum(r["margin"] for r in rows) / len(rows), "min_margin": min(r["margin"] for r in rows),
            "mean_mass": sum(r["mass"] for r in rows) / len(rows),
            "immediate_eos": sum(r["generated_ids"] == [3] for r in rows)}


def panel(m, b, out, label, items, tok, device):
    path = out / (label + "_RAW.jsonl")
    done = {}
    if path.exists():
        for l in path.read_text(encoding="utf-8").splitlines():
            v = json.loads(l)
            assert v["id"] not in done
            done[v["id"]] = v
    m.eval()
    for r in items:
        if r["id"] in done:
            continue
        v = score(m, r, tok, device)
        with path.open("a", encoding="utf-8", newline="\n") as h:
            h.write(json.dumps(v, ensure_ascii=False) + "\n")
            h.flush()
            os.fsync(h.fileno())
        done[r["id"]] = v
    vals = list(done.values())
    assert set(done) == {r["id"] for r in items}
    agg = summarize(vals)
    gg = defaultdict(list)
    for r in vals:
        gg[r["subgroup"]].append(r)
    agg["subgroups"] = {n: summarize(v) for n, v in gg.items()}
    rt.atomic_json(agg, out / (label + "_RESULT.json"))
    return agg


def checks(m, b, device):
    m.eval()
    result = {}
    pin = rt.pinned_binding(b)
    with torch.no_grad():
        for n in ["pilot0", "pilot1"]:
            qs = read(b / f"data/binding_dev_{n}.json")["quartets"]
            raw = pin.binding_eval(m, [d for q in qs for d in q["docs"]], device)
            s = rt.binding_summary(raw)
            result[n] = {"summary": s, "gate": rt.binding_gate(s), "raw": raw}
        dev = [json.loads(l) for l in (b / "data/ENGLISH_DEV.jsonl").read_text(encoding="utf-8").splitlines() if l]
        lang = rt.aligned_dev_loss(m, dev, device)
    return {"binding": result, "language": lang}


def acquire(s):
    return s["correct"] == 16 and s["exact"] == 16 and s["reversals"] == 8 and s["families"] == 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=["preload", "train"], required=True)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    b = a.bundle
    p = verify(b)
    rt.configure_runtime(p["seed"])
    assert os.environ.get("PYTHONHASHSEED") == str(p["seed"])
    import platform, tokenizers
    assert platform.python_version() == "3.12.14" and tokenizers.__version__ == "0.23.1"
    schedule = read(b / "SCHEDULE.json")
    train = read(b / "TRAIN.json")
    idx = {r["id"]: r for r in train}
    pool = read(b / "data/binding_rehearsal.json")["quartets"]
    for u in schedule:
        if u["kind"] == "english":
            assert all(i in idx for i in u["ids"])
        else:
            rt.binding_docs_for_batch(pool, u["quartets"], u["documents"])
    kl_pool, proto = load_kl_pool()
    KL_ROWS = kl_pool["rows"]
    KL_ENTRIES = kl_pool["entries"]

    if a.mode == "preload":
        print("SF2_PRE_PARENT_LOAD_PASS")
        return

    out = a.out
    out.mkdir(exist_ok=True)
    device = torch.device("cuda")
    m = rt.load_model(Path(p["parent"]), device, b)
    scope = scope_audit(m)
    rt.atomic_json(scope, out / "PARAMETER_SCOPE.json")
    tok = Tokenizer.from_file(p["tokenizer"])
    provenance = {"parent": p["parent_sha256"], "protocol": sha(b / "PROTOCOL.json"),
                  "schedule": sha(b / "SCHEDULE.json"), "receipt": sha(b / "RECEIPT.json"),
                  "sf2_protocol": sha(SF2_PROTOCOL), "kl_pool": proto["kl"]["pool_sha256"],
                  "kl_manifest": proto["kl"]["pool_manifest_sha256"], "lambda_kl": LAMBDA_KL}
    if (out / "PROVENANCE.json").exists():
        assert read(out / "PROVENANCE.json") == provenance
    else:
        rt.atomic_json(provenance, out / "PROVENANCE.json")

    teacher = build_teacher(Path(p["parent"]), device)
    assert_teacher_disjoint(m.base_model, teacher)

    restart = out / "restart.pt"
    completed = 0
    if a.resume:
        assert restart.exists()
        state = torch.load(restart, map_location=device, weights_only=False)
        assert state["provenance"] == provenance
        completed = state["completed"]
        assert 0 <= completed <= 200
        m.load_state_dict(state["model"])
    else:
        assert not restart.exists()

    if completed == 0:
        baseline = panel(m, b, out, "update0_acquisition", train, tok, device)
        if not (out / "update0_checks.json").exists():
            rt.atomic_json(checks(m, b, device), out / "update0_checks.json")
        # Baseline reproduction (M9): acquisition.
        assert baseline["correct"] == 9 and baseline["exact"] == 0, baseline
        basecheck = read(out / "update0_checks.json")
        assert all(v["gate"] for v in basecheck["binding"].values())
        ce0 = basecheck["language"]["loss"]
        assert abs(ce0 - 3.3907) / 3.3907 < 1e-4, ce0
        # D3 baseline.
        d3sel = read(Path(proto["d3"]["selection_manifest"]))
        d3r0 = measure_d3(m, d3sel, tok, device)
        rt.atomic_json(d3_summary(d3r0), out / "d3_update0_summary.json")
        with (out / "d3_update0_RAW.jsonl").open("w", encoding="utf-8", newline="\n") as h:
            for r in d3r0:
                h.write(json.dumps(r, ensure_ascii=False) + "\n")
        d3m0 = d3_summary(d3r0)["mean_combined_name_probability"]
        assert abs(d3m0 - proto["d3"]["baseline_mean_four_name_mass"]) < proto["d3"]["tolerance_abs"], d3m0
        # KL == 0 at baseline (student == teacher), using the first English-update selection.
        sel1 = pick_kl_entries(KL_ENTRIES, 1)
        m.eval()
        kl0 = compute_kl(m, teacher, device, KL_ROWS, sel1).detach()
        assert float(kl0) < proto["baseline_reproduction_M9"]["kl_zero_tolerance"], float(kl0)
        m.train()
    else:
        basecheck = read(out / "update0_checks.json")

    basecheck = read(out / "update0_checks.json")
    assert all(v["gate"] for v in basecheck["binding"].values())
    opt = torch.optim.AdamW(m.parameters(), lr=5e-5, betas=(.9, .999), eps=1e-8, weight_decay=.05,
                            amsgrad=False, foreach=False, fused=False)
    if a.resume:
        opt.load_state_dict(state["optimizer"])
        rt.restore_rng(state["rng"])
    pin = rt.pinned_binding(b)

    def commit(u):
        rt.atomic_torch_save({"completed": u, "model": m.state_dict(), "optimizer": opt.state_dict(),
                              "rng": rt.capture_rng(), "scope": "binding" if u % 10 == 0 else "english",
                              "provenance": provenance}, restart)

    def checkpoint_and_eval(u):
        cp = out / f"checkpoint_{u}.pt"
        if not cp.exists():
            rt.atomic_torch_save({"model_state_dict": m.state_dict(), "update": u, "provenance": provenance}, cp)
        cpmeta = out / f"checkpoint_{u}.sha256"
        if cpmeta.exists():
            assert cpmeta.read_text().split()[0] == sha(cp)
        else:
            cpmeta.write_text(sha(cp) + "\n")
        cq = out / f"update{u}_checks.json"
        if not cq.exists():
            rt.atomic_json(checks(m, b, device), cq)
        q = read(cq)
        agg = panel(m, b, out, f"update{u}_acquisition", train, tok, device)
        # D3 diagnostic at 100/200.
        d3sel = read(Path(proto["d3"]["selection_manifest"]))
        d3r = measure_d3(m, d3sel, tok, device)
        rt.atomic_json(d3_summary(d3r), out / f"d3_update{u}_summary.json")
        with (out / f"d3_update{u}_RAW.jsonl").open("w", encoding="utf-8", newline="\n") as h:
            for r in d3r:
                h.write(json.dumps(r, ensure_ascii=False) + "\n")
        ok = all(v["gate"] for v in q["binding"].values())
        lang = q["language"]["loss"] <= basecheck["language"]["loss"] + .25
        if not ok or not lang:
            rt.atomic_json({"status": "STOP_REGRESSION", "update": u, "binding_pass": ok,
                            "language_cost_pass": lang, "transfer_opened": False}, out / "STATUS.json")
            return False
        if u == 100:
            if (agg["correct"] < 10) and (agg["exact"] < 10):
                rt.atomic_json({"status": "SF2_STOP_ACQUISITION_GUARD", "update": u, "binding_pass": True,
                                "language_cost_pass": True, "acquisition_guard_tripped": True,
                                "correct": agg["correct"], "exact": agg["exact"], "transfer_opened": False},
                               out / "STATUS.json")
                return False
        return True

    if completed in [100, 200] and not checkpoint_and_eval(completed):
        return
    if completed == 0:
        commit(0)

    english_index = sum(1 for u in schedule if u["kind"] == "english" and u["update"] <= completed)
    for u in schedule[completed:]:
        opt.zero_grad(set_to_none=True)
        set_scope(m, u["kind"] == "binding")
        inactive = {n: (x.detach().clone(), {k: v.clone() if torch.is_tensor(v) else v for k, v in opt.state.get(x, {}).items()})
                    for n, x in m.named_parameters() if not x.requires_grad}
        ce = None
        kl_val = None
        if u["kind"] == "english":
            english_index += 1
            m.eval()
            sel = pick_kl_entries(KL_ENTRIES, english_index)
            kl_val = compute_kl(m, teacher, device, KL_ROWS, sel)
            m.train()
            batch = [idx[i] for i in u["ids"]]
            x, y = pad_batch(batch, u["pad"])
            x = x.to(device)
            y = y.to(device)
            ce = F.cross_entropy(m.base_model(x).reshape(-1, 1024), y.reshape(-1), ignore_index=-100)
            loss = ce + LAMBDA_KL * kl_val
        else:
            m.train()
            loss, _ = rt.binding_loss(m, rt.binding_docs_for_batch(pool, u["quartets"], u["documents"]), device, pin)
        loss.backward()
        assert all(x.grad is None for x in m.parameters() if not x.requires_grad)
        norm = torch.nn.utils.clip_grad_norm_([x for x in m.parameters() if x.grad is not None], 2.0)
        opt.step()
        for n, x in m.named_parameters():
            if n not in inactive:
                continue
            old, st = inactive[n]
            assert torch.equal(x, old)
            now = opt.state.get(x, {})
            assert now.keys() == st.keys()
            for k, v in st.items():
                assert torch.equal(now[k], v) if torch.is_tensor(v) else now[k] == v
        number = u["update"]
        commit(number)
        rec = {"update": number, "kind": u["kind"], "loss": float(loss.detach()), "grad_norm": float(norm)}
        if u["kind"] == "english":
            rec["ce"] = float(ce.detach())
            rec["kl"] = float(kl_val.detach())
        with (out / "TRAIN_METRICS.jsonl").open("a", encoding="utf-8") as h:
            h.write(json.dumps(rec) + "\n")
            h.flush()
            os.fsync(h.fileno())
        if number % 25 == 0:
            print("committed", number, flush=True)
        if number in [100, 200] and not checkpoint_and_eval(number):
            return

    cp = out / "checkpoint_200.pt"
    s = read(out / "update200_acquisition_RESULT.json")
    if not acquire(s):
        rt.atomic_json({"status": "SF2_ACQUISITION_FAIL", "completed": 200, "checkpoint_sha256": sha(cp),
                        "transfer_opened": False}, out / "STATUS.json")
        return
    for label in ["HELDOUT", "ALTERNATE", "COPY", "COMPETING"]:
        panel(m, b, out, label.lower(), read(b / (label + ".json")), tok, device)
    assert sha(Path(p["parent"])) == p["parent_sha256"]
    verify(b)
    rt.atomic_json({"status": "SF2_TRAINING_FAMILY_ACQUISITION_PASS_DIAGNOSTICS_COMPLETE", "completed": 200,
                    "checkpoint_sha256": sha(cp), "transfer_opened": True, "final_accessed": False},
                   out / "STATUS.json")


if __name__ == "__main__":
    main()
