"""Additive reporting-only correction: preserve sealed SF15 and expose every scored checkpoint."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

H = Path(__file__).resolve().parent
SOURCE = H.parent / "sf15_partial_first_token_ce_v1"
sys.path.insert(0, str(SOURCE))
import CONTROLLER as C  # noqa: E402


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def snapshot(directory, update):
    train = C.read(directory / f"update{update}_train16_RESULT.json")
    surface = C.read(directory / f"update{update}_devsurface_RESULT.json")
    order = C.read(directory / f"update{update}_devorder_RESULT.json")
    checks = C.read(directory / f"update{update}_checks.json")
    d3 = C.read(directory / f"d3_update{update}_summary.json")
    subgroups = order["subgroups"]
    result = {
        "update": update,
        "train16": {k: train[k] for k in ["correct", "exact", "reversals", "families"]},
        "surface": {"correct": surface["correct"], "exact": surface["exact"], "subgroups": {k: v["exact"] for k, v in surface["subgroups"].items()}},
        "order": {
            "correct": order["correct"], "exact": order["exact"],
            "fact_exact": sum(subgroups[f"order{i}:fact"]["exact"] for i in [0, 1]),
            "copy_exact": sum(subgroups[f"order{i}:copy"]["exact"] for i in [0, 1]),
            "subgroups": {k: v["exact"] for k, v in subgroups.items()},
        },
        "language": checks["language"],
        "d3": d3,
        "binding": {k: v["summary"] for k, v in checks["binding"].items()},
    }
    gate_path = directory / f"update{update}_GATES.json"
    result["gates"] = C.read(gate_path) if gate_path.exists() else "U0_BASELINE_NO_DECISION_GATE"
    return result


def binding_text(values):
    return "; ".join(
        f"{name}:{v['answer_exact']}/{v['both_distinct']}/collapse{v['slot_collapse']}/quartets{v['complete_quartets']}/rev{v['strict_reversal_both_correct']}"
        for name, v in values.items()
    )


def main():
    C.verify()
    sealed = C.read(SOURCE / "RESULTS.json")
    protocol = C.read(SOURCE / "PROTOCOL.json")
    rows = []
    for run in protocol["runs"]:
        seed = run["seed"]
        directory = SOURCE / "runs" / f"seed_{seed}_curriculum"
        status = C.read(directory / "STATUS.json")
        checkpoints = [0, 100] + ([200] if status["completed"] == 200 else [])
        snapshots = [snapshot(directory, u) for u in checkpoints]
        checkpoint_hash = sha(status["checkpoint"])
        assert checkpoint_hash == status["checkpoint_sha256"]
        metrics = [json.loads(line) for line in (directory / "TRAIN_METRICS.jsonl").read_text().splitlines()]
        english = [r for r in metrics if r["kind"] == "english"]
        binding = [r for r in metrics if r["kind"] == "binding"]
        assert len(metrics) == status["completed"]
        assert all(r["first_answer_ce_weight"] == 0.25 and r["later_response_ce_weight"] == 1.0 and r["ce_weight_sum"] == 153.0 for r in english)
        rows.append({
            "seed": seed,
            "parent_sf8_seed": run["parent_sf8_seed"],
            "parent_checkpoint": run["parent_checkpoint"],
            "parent_checkpoint_sha256": run["parent_checkpoint_sha256"],
            "status": status["status"], "completed": status["completed"],
            "checkpoint": status["checkpoint"], "checkpoint_sha256": checkpoint_hash,
            "updates": {"total": len(metrics), "english": len(english), "binding": len(binding)},
            "snapshots": snapshots,
        })

    corrected = {
        "study": "SF15_PARTIAL_FIRST_ANSWER_TOKEN_CE_RESTORATION_V1",
        "classification": sealed["classification"],
        "classification_source_sha256": sha(SOURCE / "RESULTS.json"),
        "reporting_correction": "The sealed reporter showed only U0 and terminal rows. This additive report includes preserved U100 and U200 separately for branches that reached U200; no score or classification changed.",
        "runs": rows,
        "cross_seed": sealed["cross_seed"],
        "sealed_provenance": {
            "bundle": str(SOURCE),
            "receipt_sha256": sha(SOURCE / "FREEZE_RECEIPT.json"),
            "manifest_sha256": sha(SOURCE / "SHA256SUMS.txt"),
            "controller_sha256": sha(SOURCE / "CONTROLLER.py"),
        },
        "locks": {"historical_transfer": "LOCKED_UNSCORED", "final_accessed": False, "sacred_accessed": False},
    }
    (H / "RESULTS.json").write_text(json.dumps(corrected, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    lines = [
        "# SF15 FINAL CLASSIFICATION", "", f"**{sealed['classification']}**", "",
        "# WHAT CHANGED", "",
        "The sole scientific change from sealed SF14 was first-answer-name causal CE weight `0.0 -> 0.25` during English updates. Per-record response weights were `[0.25, 1, 1, 1, 1]`, normalized by their sum. Later response/punctuation/EOS CE, first-token margin, broad KL, data, optimizer, scopes, schedule, binding, and gates were unchanged.", "",
        "# PER-SEED TRAJECTORY", "",
        "| Seed | Update | TRAIN16 c/e/r/f | Surface c/e | Order c/e (fact/copy exact) | Language CE/PPL | D3 | Binding |", "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        for snap in row["snapshots"]:
            t, s, o = snap["train16"], snap["surface"], snap["order"]
            lines.append(
                f"| {row['seed']} | {snap['update']} | {t['correct']}/{t['exact']}/{t['reversals']}/{t['families']} | "
                f"{s['correct']}/{s['exact']} | {o['correct']}/{o['exact']} ({o['fact_exact']}/{o['copy_exact']}) | "
                f"{snap['language']['loss']:.4f}/{snap['language']['perplexity']:.2f} | {snap['d3']['mean_combined_name_probability']:.5f} | {binding_text(snap['binding'])} |"
            )
        if row["completed"] < 200:
            lines.append(f"| {row['seed']} | 200 | — | — | — | — | — | NOT REACHED: frozen D3 stop at U100 |")

    lines += ["", "# U100 OH-FUCK SIGNAL", ""]
    for row in rows:
        u0 = next(s for s in row["snapshots"] if s["update"] == 0)
        u100 = next(s for s in row["snapshots"] if s["update"] == 100)
        g = u100["gates"]
        t = u100["train16"]
        lines.append(
            f"- Seed {row['seed']}: D3 `{u100['d3']['mean_combined_name_probability']:.5f}` ({'PASS' if g['d3'] else 'FAIL'}); "
            f"TRAIN16 `{t['correct']}/{t['exact']}/{t['reversals']}/{t['families']}`; language `{'PASS' if g['language'] else 'FAIL'}` "
            f"(`{u100['language']['loss']-u0['language']['loss']:+.4f}` nat); binding `{'PASS' if all(g['binding'].values()) else 'FAIL'}`; "
            f"surface exact `{u0['surface']['exact']}->{u100['surface']['exact']}`; order exact `{u0['order']['exact']}->{u100['order']['exact']}`."
        )

    lines += ["", "# FROZEN GATE RESULT", ""]
    for row in rows:
        terminal = row["snapshots"][-1]
        g = terminal["gates"]
        lines.append(
            f"- Seed {row['seed']} at U{row['completed']}: TRAIN16 `{'PASS' if g['retention_acquisition'] else 'FAIL'}`; "
            f"language `{'PASS' if g['language'] else 'FAIL'}`; binding `{'PASS' if all(g['binding'].values()) else 'FAIL'}`; "
            f"D3 `{'PASS' if g['d3'] else 'FAIL'}`; surface `{'PASS' if g['dev_surface'] else 'FAIL'}`; "
            f"order `{'PASS' if g['dev_order'] else 'FAIL'}`; complete endpoint `{'PASS' if g['endpoint_pass'] else 'FAIL'}`."
        )
    lines += [
        "", "- Complete endpoint passes: 0/3.", "- D3 failures: 3/3.", "- TRAIN16/language/binding retention failures: 0/3.", "- Surface and order exact scores improved over U0 in 3/3 branches.", "",
        "# SCIENTIFIC INTERPRETATION", "",
        "The 0.25 restoration recovered much of the exact-generation pressure lost in SF14, but it did not preserve the frozen D3 bound reproducibly. Seeds 87038 and 87039 crossed D3 at U100; seed 87040 was narrowly green at U100 and crossed by U200. All tested original acquisition, language, and binding retention remained intact. This supports only the narrow conclusion that weight 0.25 did not resolve the observed D3-versus-generation tradeoff under SF15. It does not establish an optimal CE weight, sole causation, a scaling law, or a capacity limit.", "",
        "# CHECKPOINTS", "",
    ]
    for row in rows:
        lines.append(f"- Seed {row['seed']} ({row['updates']['total']} updates): `{row['checkpoint']}` — `{row['checkpoint_sha256']}`")
    lines += [
        "", "# PROVENANCE AND LOCKS", "",
        f"- Sealed bundle receipt: `{corrected['sealed_provenance']['receipt_sha256']}`",
        f"- Sealed payload manifest: `{corrected['sealed_provenance']['manifest_sha256']}`",
        f"- Frozen controller: `{corrected['sealed_provenance']['controller_sha256']}`",
        "- Historical transfer panels: LOCKED/UNSCORED.",
        "- FINAL and sacred material: not accessed.",
        "- Reporting correction only: the sealed reporter omitted the separate U100 row for the U200 branch; this report reads the preserved raw artifacts and changes no metrics, gates, checkpoints, or classification.", "",
    ]
    (H / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "type": "ADDITIVE_REPORTING_ONLY_CORRECTION",
        "source_bundle": str(SOURCE),
        "source_receipt_sha256": sha(SOURCE / "FREEZE_RECEIPT.json"),
        "source_results_sha256": sha(SOURCE / "RESULTS.json"),
        "source_final_report_sha256": sha(SOURCE / "FINAL_REPORT.md"),
        "scientific_changes": 0,
        "checkpoint_mutations": 0,
        "inference_runs": 0,
        "training_updates": 0,
    }
    (H / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    payloads = [p for p in sorted(H.iterdir()) if p.is_file() and p.name != "SHA256SUMS.txt"]
    (H / "SHA256SUMS.txt").write_text("\n".join(f"{sha(p)}  {p.name}" for p in payloads) + "\n", encoding="utf-8", newline="\n")
    print(sealed["classification"])


if __name__ == "__main__":
    main()
