"""Render the completed HR-3 nonsacred comparison into an auditable report."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(r"C:\DaveLM-CADAVER")
RUN = ROOT / "human_readiness_hr3_causal_seed87006_execution_v7"
DATA = json.loads((RUN / "NONSACRED_DEV_COMPARISON.json").read_text(encoding="utf-8"))


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100*n/d:.1f}%)"


def main() -> None:
    lines: list[str] = []
    lines += ["# HR-3 block-3 causal scope-relaxation report", "", "## Classification", "", "**HR3_FAILED_MACHINE_READINESS_GATES_BINDING_PRESERVED**", "", "HR-3 completed 500/500 updates from the verified Pilot 1 parent. It did not meet the frozen readiness development gates. Both nonsacred binding pools independently passed their preservation gates. No FINAL or sacred material was accessed.", ""]
    lines += ["## Provenance", "", f"- Bundle: `{DATA['bundle']}`", f"- Bundle receipt SHA-256: `{DATA['bundle_receipt_sha256']}`", f"- Run directory: `{RUN}`", f"- Parent: Pilot 1, SHA-256 `{DATA['checkpoints'][0]['checkpoint_sha256']}`", f"- HR-3 checkpoint 500: `{DATA['checkpoints'][-1]['checkpoint']}`, SHA-256 `{DATA['checkpoints'][-1]['checkpoint_sha256']}`", "- Training: seed 87006, 500 updates (450 English / 50 binding), block 3 newly trainable on English; blocks 0-2 and T13 localization/retrieval protected.", "- FINAL readiness battery: sealed and untouched.", ""]
    lines += ["## Aligned DEV trajectory", "", "| checkpoint | loss | perplexity |", "|---|---:|---:|"]
    # The trainer trajectory is the durable update diagnostic; HR-3 endpoint
    # comparison below uses the same aligned objective.
    trajectory = json.loads((RUN / "DEV_TRAJECTORY.json").read_text(encoding="utf-8"))
    for key in ("0", "100", "250", "500"):
        value = trajectory.get(key)
        if value:
            metric = value["aligned_tinystories"]
            lines.append(f"| update {key} | {metric['loss']:.4f} | {metric['perplexity']:.2f} |")
    lines += ["", "Pilot 1 aligned loss was 3.3907 (PPL 29.69). Corrected causal HR-1 update 500 was 2.8305 (PPL 16.95). HR-3 update 500 was 2.8814 (PPL 17.83), so the scope relaxation did not outperform aligned HR-1 on this fixed loss diagnostic.", ""]
    lines += ["## Readiness DEV comparison", "", "| checkpoint | fact correct | reversals | complete families | fact greedy | instruction | continuity | generation non-EOS | automatic non-degenerate |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for checkpoint in DATA["checkpoints"]:
        s = checkpoint["controlled_summary"]
        g = checkpoint["generation_summary"]
        f, i, c = s["fact"], s["instruction"], s["continuity"]
        lines.append(f"| {checkpoint['name']} | {pct(f['correct'], f['items'])} | {pct(f['successful_reversal_pairs'], f['reversal_pairs'])} | {f['complete_families']}/{f['family_count']} | {pct(f['greedy_exact_correct_then_eos'], f['items'])} | {pct(i['correct'], i['items'])} | {pct(c['correct'], c['items'])} | {pct(g['non_immediate_eos'], g['prompts'])} | {pct(g['automatic_non_degenerate'], g['prompts'])} |")
    lines += ["", "Facts are scored by positive correct-minus-distractor conditional log-likelihood margin; ties fail. Reversal success requires both members of a matched assignment pair to be correct. Greedy exact requires the candidate token sequence followed immediately by EOS.", ""]
    lines += ["## Binding preservation", "", "| checkpoint | pool | answer exact | BOTH_DISTINCT | collapse | complete quartets | strict reversal | gate |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for checkpoint in DATA["checkpoints"]:
        for pool, result in checkpoint["binding"].items():
            s = result["summary"]
            gate = "PASS" if result["gate_pass"] else "FAIL"
            lines.append(f"| {checkpoint['name']} | {pool} | {s['answer_exact']}/80 | {s['both_distinct']}/80 | {s['slot_collapse']} | {s['complete_quartets']}/20 | {s['strict_reversal_both_correct']}/{s['strict_reversal_pairs']} | {gate} |")
    lines += ["", "Both HR-3 checkpoints evaluated at updates 100, 250, and 500 passed both pools (80/80 answer, 80/80 BOTH_DISTINCT, zero collapse).", ""]
    lines += ["## Frozen machine-gate outcome", ""]
    for checkpoint in DATA["checkpoints"]:
        lines.append(f"- **{checkpoint['name']}**: `{checkpoint['gates']['development_candidate_status']}`; all machine gates: `{checkpoint['gates']['machine_gates_all_pass']}`; binding gates: `{checkpoint['gates']['binding_gates_all_pass']}`.")
    lines += ["", "No checkpoint met the fact, reversal, complete-family, greedy, instruction, continuity, and generation criteria together. Human-review fields were left pending and did not affect this machine-gate classification.", ""]
    lines += ["## HR-3 update-500 raw generation diagnostic", "", "Raw responses below are copied verbatim from the frozen greedy procedure; they are descriptive and not repaired or human-scored.", ""]
    endpoint = next(x for x in DATA["checkpoints"] if x["name"] == "HR3_block3_update_500")
    for row in endpoint["generations"]:
        lines += [f"**{row['id']}** — prompt: `{row['prompt']}`", "", f"Response: `{row['raw_decoded_response']}`", ""]
    lines += ["## Parameter drift from Pilot 1", ""]
    for group, value in DATA["pilot1_to_hr3_update500_parameter_drift"].items():
        lines.append(f"- {group}: {value['relative_l2_delta_percent']:.4f}% relative L2 delta across {value['parameter_tensors']} tensors.")
    lines += ["", "The measured drift is a descriptive scope audit. It does not establish that block 3 caused or failed to cause any behavioral change.", ""]
    lines += ["## Interpretation", "", "HR-3 lowered the correctly aligned TinyStories DEV loss relative to Pilot 1, but controlled contextual fact behavior did not improve: 12/24 items, 0/12 matched reversals, and 0 complete families at update 500. This does not support reliable use of short contextual facts, counterfactual selection, or HUMAN_TEST_READY readiness. Generation remained short and malformed, with 14/20 non-immediate-EOS responses and 13/20 automatically non-degenerate responses; several responses were fragments, punctuation-only, repetitive, or context-inappropriate. Instruction and continuity remained below their frozen gates.", "", "The treatment preserves the established binding capability on both nonsacred pools. Failure is evidence about this bounded scope-only treatment and these diagnostics, not architectural impossibility or a general capacity ceiling. A future step would require a new prospective decision; this report makes no parent or curriculum selection.", ""]
    lines += ["## Artifact pointers", "", f"- Complete machine-readable comparison, including every controlled item, token-level candidate scores, margins, greedy IDs, raw generation text, and binding rows: `{RUN / 'NONSACRED_DEV_COMPARISON.json'}`", f"- Durable trainer metrics: `{RUN / 'UPDATE_METRICS.jsonl'}`", f"- Durable update trajectory: `{RUN / 'DEV_TRAJECTORY.json'}`", f"- Frozen v2 readiness battery: `C:\\DaveLM-CADAVER\\human_test_readiness_v2_seed87010` (not modified or reopened beyond its sealed manifest/receipt checks).", ""]
    (RUN / "HR3_REPORT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(str(RUN / "HR3_REPORT.md"))


if __name__ == "__main__":
    main()
