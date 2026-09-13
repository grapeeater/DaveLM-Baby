from __future__ import annotations
import hashlib, json, math, os, platform, re, statistics, sys, time
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(r"C:\DaveLM-CADAVER")
FROZEN = ROOT / "post_p7_language_report_card_v1_seed8380"
CHECKPOINT = ROOT / "archive" / "DAVELM_P7_MILESTONE_seed8380" / "davelm_p7_latest.pt"
TOKENIZER = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")
OUT = ROOT / "post_p7_language_report_card_v1_seed8380_execution_p7"
EXPECTED_CHECKPOINT = "d41ed1186cc945aa05dbd2ba3086fac08ff3b97035c9532e4fa70e78b149a20e"
EXPECTED_TOKENIZER = "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"
ORDER = ("near_distribution", "counterfactual", "surface_form", "distractor")

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def write_json(path, obj):
    Path(path).write_bytes((json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8"))

def norm(s):
    return re.sub(r"\s+", " ", s.casefold()).strip()

def basis(u):
    un = u / torch.linalg.vector_norm(u)
    p = int(torch.argmax(torch.abs(un)))
    e = torch.zeros_like(un); e[p] = 1.0 if un[p] >= 0 else -1.0
    w = un - e
    H = torch.eye(un.numel(), device=u.device, dtype=u.dtype) - 2 * torch.outer(w, w) / torch.dot(w, w)
    return H[:, [i for i in range(un.numel()) if i != p]]

def sentence_complete(text):
    return bool(re.search(r"[.!?](?:\s|$)", text))

def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Refusing repeat or overwrite of existing execution directory: {OUT}")
    OUT.mkdir(parents=True, exist_ok=False)
    # Frozen material verification is completed before importing torch/model code.
    assert sha(CHECKPOINT) == EXPECTED_CHECKPOINT
    assert sha(TOKENIZER) == EXPECTED_TOKENIZER
    frozen_files = sorted(p for p in FROZEN.iterdir() if p.is_file())
    assert all(p.stat().st_mode & 0o200 == 0 or p.is_file() for p in frozen_files)  # Windows read-only checked below
    import stat
    assert all(not bool(p.stat().st_mode & stat.S_IWRITE) for p in frozen_files)
    sums = (FROZEN / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    checksum_records = {}
    for line in sums:
        digest, name = line.split("  ", 1)
        assert sha(FROZEN / name) == digest, name
        checksum_records[name] = digest
    assert "SHA256SUMS.txt" not in checksum_records
    receipt = json.loads((FROZEN / "RECEIPT.json").read_text(encoding="utf-8"))
    assert receipt["model_evaluation_performed"] is False and receipt["checkpoint_behavior_accessed"] is False
    assert json.loads((FROZEN / "PREFLIGHT.json").read_text(encoding="utf-8"))["status"] == "PASS_PREEXECUTION_ONLY"
    source_receipt = json.loads((FROZEN / "SOURCE_HASH_RECEIPT.json").read_text(encoding="utf-8"))
    for rec in source_receipt.values():
        if isinstance(rec, dict) and "path" in rec:
            assert sha(rec["path"]) == rec["sha256"]
    items = [json.loads(x) for x in (FROZEN / "ITEMS.jsonl").read_text(encoding="utf-8").splitlines()]
    controlled = [x for x in items if x["role"] == "controlled"]
    natural = [x for x in items if x["role"] == "generation"]
    assert len(controlled) == 288 and len(natural) == 24 and len(items) == 312
    assert len({x["prompt"] for x in items}) == 312
    families = json.loads((FROZEN / "FAMILIES.json").read_text(encoding="utf-8"))["families"]
    assert len(families) == 36 and all(sum(x["family_id"] == f["family_id"] for x in controlled) == 8 for f in families)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER))
    for x in controlled:
        pids = tok.encode(x["prompt"]).ids
        assert len(x["candidate_token_ids"]) == 2
        assert all(tok.encode(c).ids == ids for c, ids in zip(x["candidates"], x["candidate_token_ids"]))
        assert all(tok.decode(ids) == c for c, ids in zip(x["candidates"], x["candidate_token_ids"]))
        assert tok.encode(x["prompt"] + x["candidates"][0]).ids == pids + x["candidate_token_ids"][0]
        assert tok.encode(x["prompt"] + x["candidates"][1]).ids == pids + x["candidate_token_ids"][1]
        assert len(x["candidate_token_ids"][0]) == len(x["candidate_token_ids"][1]) == 3
    # Fixed sample rule, selected without inspecting model outputs: first item by item_id in each section.
    sample_ids = {section: sorted(x["item_id"] for x in controlled if x["section"] == section)[0] for section in ORDER}
    preflight = {"status": "PASS_BEFORE_INFERENCE", "checkpoint_sha256": sha(CHECKPOINT), "tokenizer_sha256": sha(TOKENIZER),
                 "frozen_checksum_records": checksum_records, "frozen_files_read_only": True, "items": 312,
                 "controlled_items": 288, "naturalistic_items": 24, "families": 36, "unique_prompts": 312,
                 "sample_rule": "lexicographically first item_id within each controlled section", "sample_ids": sample_ids,
                 "candidate_lengths": [3, 3], "scoring_definition_sha256": sha(FROZEN / "PROTOCOL.md"),
                 "serialization_balancing_checks": True, "sacred_exam_accessed": False, "model_loaded": False,
                 "inference_started": False, "frozen_battery_modified": False}
    write_json(OUT / "PREFLIGHT_RECEIPT.json", preflight)
    # Load exactly the P7 wrapper, but evaluate only model.base_model for English scoring.
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, r"C:\DaveLM-v0.9")
    import torch
    import torch.nn as nn
    from treatment13_model import Treatment13Model
    class OrthoLocalizer(nn.Module):
        def __init__(self, u, q, bs, ba):
            super().__init__(); self.u=nn.Parameter(u); self.q=nn.Parameter(q); self.bs=nn.Parameter(bs); self.ba=nn.Parameter(ba)
        def forward(self, h):
            v = basis(self.u) @ self.q; S = h @ self.u + self.bs; R = h @ v + self.ba
            return torch.stack((S + R, S - R), -1)
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    assert DEVICE == "cuda:0"
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"): torch.backends.cudnn.allow_tf32 = False
    raw = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    state = raw["model_state_dict"]
    model = Treatment13Model()
    model.localizer = OrthoLocalizer(*(state[f"localizer.{n}"].clone() for n in ("u", "q", "bs", "ba")))
    model.load_state_dict(state, strict=True)
    model.to(DEVICE).eval()
    assert sum(p.numel() for p in model.base_model.parameters()) == 10594944
    load_receipt = {"checkpoint_sha256": sha(CHECKPOINT), "strict_state_load": True, "base_model_parameters": 10594944,
                    "device": DEVICE, "model_dtype": "float32", "english_path": "model.base_model(input_ids)",
                    "specialized_path_used": False, "autocast": False, "tf32": False}
    write_json(OUT / "LOAD_RECEIPT.json", load_receipt)
    def score_pair(record):
        prefix = [2] + tok.encode(record["prompt"]).ids
        candidates = record["candidate_token_ids"]
        scores = []
        token_lps = []
        for cand in candidates:
            ids = prefix + cand
            inp = torch.tensor([ids], dtype=torch.long, device=DEVICE)
            with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
                logits = model.base_model(inp)[0].detach().float().cpu().double()
            start = len(prefix) - 1
            lp = torch.log_softmax(logits, dim=-1)
            vals = [float(lp[start+j, tid]) for j, tid in enumerate(cand)]
            token_lps.append(vals); scores.append(sum(vals))
        d = scores[0] - scores[1]; ci = record["correct_index"]
        margin = d if ci == 0 else -d
        mx = max(scores); logmass = mx + math.log(sum(math.exp(v-mx) for v in scores))
        word = [sum(v[:2]) for v in token_lps]; wd = word[0] - word[1]; wmargin = wd if ci == 0 else -wd
        return {"candidates":[{"candidate_token_ids":candidates[i],"token_log_probabilities":token_lps[i],"conditional_log_likelihood":scores[i],"word_only_log_likelihood":word[i]} for i in range(2)],
                "candidate0_minus_candidate1":d,"correct_minus_incorrect_margin":margin,"item_correct":margin>0,"tie":d==0,
                "word_only_candidate0_minus_candidate1":wd,"word_only_correct_minus_incorrect_margin":wmargin,"word_only_item_correct":wmargin>0,"word_only_tie":wd==0,
                "candidate_pair_log_probability_mass":logmass,"candidate_pair_probability_mass":math.exp(logmass),"correct_candidate_index":ci,
                "completed_input_length_including_bos":len(prefix)+3,"runtime":load_receipt}
    rows=[]; started=time.time(); forward_count=0
    for rec in controlled:
        rows.append({"checkpoint":"p7","checkpoint_sha256":EXPECTED_CHECKPOINT,"kind":"contextual","record":rec,"score":score_pair(rec)}); forward_count += 1
    for rec in natural:
        prefix=[2]+tok.encode(rec["prompt"]).ids; generated=[]; ids=prefix[:]
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            for _ in range(32):
                inp=torch.tensor([ids[-256:]],dtype=torch.long,device=DEVICE)
                nxt=int(model.base_model(inp)[0,-1,:].argmax().item()); generated.append(nxt); ids.append(nxt)
                text=tok.decode(generated,skip_special_tokens=True)
                if nxt==3 or sentence_complete(text): break
        out=tok.decode(generated,skip_special_tokens=True)
        rows.append({"checkpoint":"p7","checkpoint_sha256":EXPECTED_CHECKPOINT,"kind":"naturalistic","record":rec,"generation":{"token_ids":generated,"continuation":out,"rubric":"HUMAN REVIEW REQUIRED"}})
    assert forward_count == 288 and len(rows) == 312
    with (OUT/"RAW_SCORES.jsonl").open("w",encoding="utf-8",newline="\n") as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n")
    # Aggregate controlled metrics by section/family, retaining raw values.
    controlled_rows=[r for r in rows if r["kind"]=="contextual"]
    summary={"checkpoint":"p7","checkpoint_sha256":EXPECTED_CHECKPOINT,"runtime":load_receipt,"n_controlled":288,"n_naturalistic":24,"sections":{},"sample_ids":sample_ids}
    for section in ORDER:
        sr=[r for r in controlled_rows if r["record"]["section"]==section]; byfam=defaultdict(list)
        for r in sr: byfam[r["record"]["family_id"]].append(r)
        fs=[]; reversal=[]
        for fid, rr in sorted(byfam.items()):
            pairs=[]
            for q in (0,1):
                for o in (0,1):
                    pair=sorted([r for r in rr if r["record"]["query_index"]==q and r["record"]["fact_order"]==o],key=lambda r:r["record"]["assignment"])
                    pairs.append({"query_index":q,"fact_order":o,"both_correct":all(x["score"]["item_correct"] for x in pair),"assignment_margins":[x["score"]["correct_minus_incorrect_margin"] for x in pair],"signed_reversal_change":pair[0]["score"]["candidate0_minus_candidate1"]-pair[1]["score"]["candidate0_minus_candidate1"]})
            margins=[r["score"]["correct_minus_incorrect_margin"] for r in rr]
            fs.append({"family_id":fid,"item_correct":sum(x["score"]["item_correct"] for x in rr),"complete_family":all(x["score"]["item_correct"] for x in rr),"reversal_both_correct":sum(x["both_correct"] for x in pairs),"reversal_fraction":sum(x["both_correct"] for x in pairs)/4,"margins":{"n":8,"min":min(margins),"mean":statistics.fmean(margins),"median":statistics.median(margins),"max":max(margins)},"reversal_profile":pairs})
        margins=[r["score"]["correct_minus_incorrect_margin"] for r in sr]
        summary["sections"][section]={"families":fs,"items":len(sr),"item_accuracy":sum(x>0 for x in margins)/len(margins),"ties":sum(x==0 for x in margins),"mean_margin":statistics.fmean(margins),"median_margin":statistics.median(margins),"minimum_margin":min(margins),"maximum_margin":max(margins),"complete_families":sum(x["complete_family"] for x in fs),"complete_family_proportion":sum(x["complete_family"] for x in fs)/len(fs),"mean_within_family_reversal_success":statistics.fmean(x["reversal_fraction"] for x in fs),"sample":next(r for r in sr if r["record"]["item_id"]==sample_ids[section])}
    summary["naturalistic"]={"items":24,"rubric":"HUMAN REVIEW REQUIRED for all prompts; raw continuations preserved verbatim"}
    write_json(OUT/"SUMMARY.json",summary)
    lines=["# P7 post-P7 language report card execution", "", "This is the single authorized evaluation of the immutable P7 archive on the frozen battery. No training occurred; the frozen battery was not modified. Naturalistic rubric fields await human review.", "", "## Controlled results", ""]
    for sec in ORDER:
        s=summary["sections"][sec]; lines += [f"### {sec}", f"Items: {s['items']}; item accuracy: {s['item_accuracy']:.6f}; ties: {s['ties']}; complete families: {s['complete_families']}/{len(s['families'])}; mean within-family reversal success: {s['mean_within_family_reversal_success']:.6f}; margin mean/min/median/max: {s['mean_margin']:.6f}/{s['minimum_margin']:.6f}/{s['median_margin']:.6f}/{s['maximum_margin']:.6f}", "", "Deterministic representative sample:", "", "PROMPT:", s["sample"]["record"]["prompt"], "", f"CORRECT CANDIDATE: {s['sample']['record']['candidates'][s['sample']['record']['correct_index']]}", f"COMPETING CANDIDATE: {s['sample']['record']['candidates'][1-s['sample']['record']['correct_index']]}", f"CANDIDATE LOG-LIKELIHOODS: {[c['conditional_log_likelihood'] for c in s['sample']['score']['candidates']]}", f"MARGIN: {s['sample']['score']['correct_minus_incorrect_margin']}", f"SELECTED ANSWER: {s['sample']['record']['candidates'][0 if s['sample']['score']['candidate0_minus_candidate1']>0 else 1]}", f"RESULT: {'correct' if s['sample']['score']['item_correct'] else ('tie' if s['sample']['score']['tie'] else 'incorrect')}", ""]
    lines += ["## Naturalistic outputs", ""]
    for r in rows:
        if r["kind"]=="naturalistic": lines += ["PROMPT:", r["record"]["prompt"], "", "BABY:", r["generation"]["continuation"], "", "RUBRIC:", "HUMAN REVIEW REQUIRED", ""]
    lines += ["## Interpretation", "", "These results characterize only the frozen constructions. They do not establish broad English competence, conversation, general reasoning, or a causal effect of P7 training. Failures do not establish architectural impossibility. Exact-string non-overlap does not exclude semantic or near-duplicate contamination.", ""]
    (OUT/"REPORT.md").write_bytes("\n".join(lines).encode("utf-8"))
    post_hash=sha(CHECKPOINT); assert post_hash==EXPECTED_CHECKPOINT
    status={"status":"COMPLETE","checkpoint_sha256_before":EXPECTED_CHECKPOINT,"checkpoint_sha256_after":post_hash,"checkpoint_unchanged":True,"frozen_battery_modified":False,"sacred_exam_accessed":False,"training":False,"raw_rows":len(rows),"elapsed_seconds":time.time()-started,"runner_sha256":sha(Path(__file__))}
    write_json(OUT/"EXECUTION_RECEIPT.json",status)
    files=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!="OUTPUT_SHA256SUMS.txt")
    sums_out="".join(f"{sha(p)}  {p.name}\n" for p in files)
    (OUT/"OUTPUT_SHA256SUMS.txt").write_bytes(sums_out.encode("utf-8"))
    manifest={"status":"COMPLETE","checkpoint_sha256":post_hash,"runner_sha256":status["runner_sha256"],"artifacts":{p.name:{"bytes":p.stat().st_size,"sha256":sha(p)} for p in files},"detached_sha256s_sha256":sha(OUT/"OUTPUT_SHA256SUMS.txt")}
    write_json(OUT/"OUTPUT_MANIFEST.json",manifest)
    # Manifest was written after the first checksum list; replace the list once more to include it.
    files=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!="OUTPUT_SHA256SUMS.txt")
    (OUT/"OUTPUT_SHA256SUMS.txt").write_bytes("".join(f"{sha(p)}  {p.name}\n" for p in files).encode("utf-8"))
    for p in OUT.iterdir(): p.chmod(p.stat().st_mode & ~stat.S_IWRITE)
    print(json.dumps({"status":"COMPLETE","out":str(OUT),"controlled":288,"naturalistic":24,"checkpoint_sha256":post_hash,"output_sha256sums_sha256":sha(OUT/"OUTPUT_SHA256SUMS.txt")},indent=2))

if __name__ == "__main__":
    main()
