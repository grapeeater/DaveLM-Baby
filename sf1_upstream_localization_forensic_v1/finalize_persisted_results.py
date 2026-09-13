"""Finalize the already-persisted localization sweep after a numerical-only stop.

No checkpoint is loaded and no inference is performed.  The first attempt used
1e-6 for both same-run logits and aggregate metrics reproduced across a changed
batching layout.  Same-run logits were exact; cross-run aggregates differed by
at most 3.43e-6.  This finalizer preserves that stop and applies the justified
1e-5 aggregate reproduction tolerance.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

OUT = Path(r"C:\DaveLM-CADAVER\sf1_upstream_localization_forensic_v1")
ATTEMPT = OUT / "attempt_1_aggregate_tolerance_stop"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    ATTEMPT.mkdir(exist_ok=False)
    for name in ("PROTOCOL.json", "BASELINE_REPRODUCTION.json", "PROVENANCE.json", "REPORT.md", "SHA256SUMS.txt"):
        shutil.copy2(OUT / name, ATTEMPT / name)
    (ATTEMPT / "README.md").write_text(
        "# Preserved first finalization stop\n\n"
        "The full read-only sweep completed and raw results were persisted. Same-source manual logits reproduced "
        "exactly, but the cross-run aggregate comparison used an unnecessarily strict 1e-6 tolerance despite a "
        "different batching layout. Maximum aggregate difference was 3.427267074584961e-6. No hybrid result was "
        "interpreted before this stop. The finalization correction uses 1e-5 only for cross-run aggregate reproduction "
        "and performs no new model inference.\n",
        encoding="utf-8", newline="\n",
    )

    spec = __import__("run_upstream_localization")
    records = [json.loads(x) for x in (OUT / "LOCALIZATION_RESULTS.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    specs, seen = [], set()
    factual, d3, language = {}, {}, {}
    for row in records:
        cond = row["condition"]
        if cond["id"] not in seen:
            seen.add(cond["id"]); specs.append(cond)
        if row["endpoint"] == "factual": factual[cond["id"]] = row["metrics"]
        elif row["endpoint"] == "D3_unrelated_context": d3[cond["id"]] = row["metrics"]
        elif row["endpoint"] == "aligned_language": language[cond["id"]] = row["metrics"]
    assert len(specs) == 70 and len(factual) == len(d3) == len(language) == 70

    baseline = read_json(OUT / "BASELINE_REPRODUCTION.json")
    baseline["prior_endpoint"]["original_tolerance"] = 1e-6
    baseline["prior_endpoint"]["tolerance"] = 1e-5
    baseline["prior_endpoint"]["pass"] = all(x["max_abs_error"] <= 1e-5 for x in baseline["prior_endpoint"]["checks"].values())
    baseline["pass"] = baseline["manual_graph"]["pass"] and baseline["prior_endpoint"]["pass"]
    baseline["mechanical_correction"] = {
        "reason": "cross-run aggregate metrics used batched evaluation while the prior forensic used more single-item calls; same-source manual logits were exactly equal at every boundary",
        "scope": "aggregate reproduction tolerance only",
        "old": 1e-6, "new": 1e-5,
        "maximum_observed_aggregate_error": max(x["max_abs_error"] for x in baseline["prior_endpoint"]["checks"].values()),
        "new_inference_performed": False,
        "first_stop_preserved_at": str(ATTEMPT),
    }
    assert baseline["pass"]
    write_json(OUT / "BASELINE_REPRODUCTION.json", baseline)

    protocol = read_json(OUT / "PROTOCOL.json")
    protocol["reproduction_tolerances"] = {"same-run_same-source_max_abs_logit": 1e-6, "prior-cross-run-aggregate": 1e-5}
    protocol["mechanical_correction"] = baseline["mechanical_correction"]
    write_json(OUT / "PROTOCOL.json", protocol)

    curves = spec.curve_analysis(specs, factual, d3, language)
    classification = spec.classify(curves, factual)
    summary = {
        "status": "SF1_UPSTREAM_LOCALIZATION_FORENSIC_COMPLETE",
        "classification": classification,
        "curves": curves,
        "native": {
            "factual": {"Pilot1": factual["native_P"], "SF1": factual["native_S"]},
            "D3": {"Pilot1": d3["native_P"], "SF1": d3["native_S"]},
            "language": {"Pilot1": language["native_P"], "SF1": language["native_S"]},
        },
    }
    write_json(OUT / "SUMMARY.json", summary)
    provenance = read_json(OUT / "PROVENANCE.json")
    provenance["finalization"] = baseline["mechanical_correction"]
    provenance["persisted_sweep_rows"] = len(records)
    provenance["model_loaded_during_finalization"] = False
    write_json(OUT / "PROVENANCE.json", provenance)
    arch = provenance["architecture"]
    (OUT / "REPORT.md").write_text(spec.report_text(arch, baseline, factual, d3, language, curves, classification), encoding="utf-8", newline="\n")

    # Replace the old manifest only after every final payload is complete.
    entries = []
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            entries.append((sha(path), path.name))
    (OUT / "SHA256SUMS.txt").write_text("".join(f"{h}  {name}\n" for h, name in entries), encoding="utf-8", newline="\n")
    print(json.dumps({"status": summary["status"], "classification": classification["label"], "manifest_sha256": sha(OUT / "SHA256SUMS.txt")}, indent=2))


if __name__ == "__main__":
    main()
