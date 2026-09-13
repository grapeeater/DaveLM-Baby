"""Read-only post-SF8 transfer evaluation of the three lambda_margin=0.25 checkpoints.

ZERO training / optimizer steps / weight modification. m.eval() + torch.no_grad only.
Scores the SF1-frozen non-sacred transfer panels (HELDOUT16, ALTERNATE48, COPY8,
COMPETING64) using the exact pinned SF2/SF1 panel machinery (SF2_ENGINE.panel -> score/
summarize), which is the frozen scoring semantics for these panels.

Eligibility (established from frozen artifacts):
- SF1 PROTOCOL.json evaluation_order: 'Only if endpoint acquisition AND language/binding
  checks pass: HELDOUT16, ALTERNATE48, COPY8, COMPETING64. All frozen before training.'
- The three endpoints (87017/87018/87019 lambda=0.25 @200) passed ACQUISITION_SUCCESS plus
  language/D3/binding gates in SF8.
- final_access=false, sacred_access=false in SF1 protocol; FINAL/sacred never accessed here.
- Control arms are NOT scored: SF8 controls failed acquisition; no frozen authorization
  exists to open panels for non-passing arms.
"""
import hashlib, json, os, sys
from pathlib import Path
import torch
import tokenizers
from tokenizers import Tokenizer

SF8 = Path(r"C:\DaveLM-CADAVER\sf8_margin_dose_comparison_v1")
SF1 = Path(r"C:\DaveLM-CADAVER\single_fact_acquisition_sf1_seed87011")
PARENT = Path(r"C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\pilot_run\checkpoints\seed_8380\latest.pt")
OUT = Path(__file__).resolve().parent

sys.path.insert(0, str(SF8))
import SF2_ENGINE as E  # noqa: E402
rt = E.rt


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def preflight():
    # SF8 protocol/tokenizer identity
    proto = read(SF8 / "PROTOCOL.json")
    assert proto["parent_sha256"] == "2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb"
    assert proto["tokenizer_sha256"] == "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"
    assert sha(PARENT) == proto["parent_sha256"]
    assert sha(proto["tokenizer"]) == proto["tokenizer_sha256"]
    # SF1 bundle receipt integrity
    assert sha(SF1 / "RECEIPT.json") == (SF1 / "RECEIPT.sha256").read_text().split()[0]
    r1 = read(SF1 / "RECEIPT.json")
    assert sha(SF1 / "SHA256SUMS.txt") == r1["manifest_sha256"]
    manifest = {}
    for line in (SF1 / "SHA256SUMS.txt").read_text().splitlines():
        if line.strip():
            h, n = line.split("  ", 1)
            manifest[n] = h
    # Checkpoints: identity from frozen SF8 STATUS.json + on-disk hash
    ck = {}
    for s in [87017, 87018, 87019]:
        d = SF8 / "runs" / f"seed_{s}_low"
        st = read(d / "STATUS.json")
        assert st["status"] == "ACQUISITION_SUCCESS" and st["completed"] == 200
        assert st["transfer"] == "LOCKED_UNSCORED"
        cp = d / "checkpoint_200.pt"
        assert sha(cp) == st["checkpoint_sha256"], s
        assert sha(d / "PROVENANCE.json") or True
        prov = read(d / "PROVENANCE.json")
        assert prov["arm"] == "low" and prov["lambda_margin"] == 0.25 and prov["seed"] == s
        ck[s] = {"path": str(cp), "sha256": st["checkpoint_sha256"], "status": st["status"]}
    # Panels: integrity from the sealed SF1 manifest + item counts
    panels = {}
    for label, count in [("HELDOUT", 16), ("ALTERNATE", 48), ("COPY", 8), ("COMPETING", 64)]:
        p = SF1 / f"{label}.json"
        assert manifest.get(f"{label}.json") == sha(p), label
        items = read(p)
        assert len(items) == count, (label, len(items))
        panels[label] = {"path": str(p), "sha256": manifest[f"{label}.json"], "items": items, "count": count}
    return proto, ck, panels


def main():
    proto, ck, panels = preflight()
    tok = Tokenizer.from_file(proto["tokenizer"])
    device = torch.device("cuda")
    raw_dir = OUT / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    receipt = {"study": "SF8_LOWDOSE_TRANSFER_EVAL_V1", "checkpoints": ck,
               "panels": {k: {"path": v["path"], "sha256": v["sha256"], "count": v["count"]} for k, v in panels.items()},
               "evaluator_sha256": sha(Path(__file__)), "engine_sha256": sha(SF8 / "SF2_ENGINE.py"),
               "transfer": "HELDOUT16+ALTERNATE48+COPY8+COMPETING64", "final_sacred_accessed": False}
    results = {}
    for s in [87017, 87018, 87019]:
        ckpt = ck[s]
        m = rt.load_model(PARENT, device, SF8)
        raw = torch.load(ckpt["path"], map_location=device, weights_only=False)
        assert raw["update"] == 200
        m.load_state_dict(raw["model_state_dict"], strict=True)
        m.eval()
        seed_dir = raw_dir / f"seed_{s}_low"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_res = {}
        with torch.no_grad():
            for label in ["HELDOUT", "ALTERNATE", "COPY", "COMPETING"]:
                agg = E.panel(m, SF8, seed_dir, label.lower(), panels[label]["items"], tok, device)
                seed_res[label] = agg
        results[s] = seed_res
        m = None
        torch.cuda.empty_cache()
    json_out = {"checkpoints": ck, "results": results}
    (OUT / "RESULTS.json").write_text(json.dumps(json_out, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    files = [OUT / "RESULTS.json", Path(__file__)] + sorted(p for p in raw_dir.rglob("*") if p.is_file())
    manifest_text = "".join(f"{sha(p)}  {p.relative_to(OUT).as_posix()}\n" for p in files)
    (OUT / "TRANSFER_SHA256SUMS.txt").write_text(manifest_text, encoding="utf-8", newline="\n")
    receipt["output_manifest_sha256"] = sha(OUT / "TRANSFER_SHA256SUMS.txt")
    (OUT / "PROVENANCE.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("TRANSFER_EVAL_COMPLETE")


if __name__ == "__main__":
    main()
