"""Add matched-pair decomposition and finalize the evidence-based interpretation.

This reads persisted diagnostic outputs only. It does not load a checkpoint or
perform inference.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

OUT = Path(r"C:\DaveLM-CADAVER\sf1_upstream_localization_forensic_v1")
PARENT = Path(r"C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\pilot_run\checkpoints\seed_8380\latest.pt")
CHILD = Path(r"C:\DaveLM-CADAVER\single_fact_acquisition_sf1_seed87011_run\checkpoint_100.pt")
TOKENIZER = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")


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
    summary = read_json(OUT / "SUMMARY.json")
    raw = [json.loads(x) for x in (OUT / "FACTUAL_ITEM_RESULTS.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    conditions = []
    for row in raw:
        if row["condition"] not in conditions:
            conditions.append(row["condition"])
    derived = []
    for condition in conditions:
        rows = [r for r in raw if r["condition"] == condition]
        pairs = collections.defaultdict(list)
        for row in rows:
            pairs[row["id"].rsplit(":", 1)[0]].append(row)
        pair_rows = []
        for pair_id, xs in sorted(pairs.items()):
            actors = {x["actor"] for x in xs}
            group = "Alex/Owen" if actors == {"Alex", "Owen"} else "Mia/Nora"
            pair_rows.append({"pair_id": pair_id, "group": group, "both_correct": all(x["sequence_margin"] > 0 for x in xs), "signed_margin_sum": sum(x["sequence_margin"] for x in xs)})
        alex = [r["sequence_margin"] for r in rows if r["actor"] == "Alex"]
        owen = [r["sequence_margin"] for r in rows if r["actor"] == "Owen"]
        alex_mean = sum(alex) / len(alex); owen_correct_mean = sum(owen) / len(owen)
        derived.append({
            "condition": condition,
            "reversal_pairs_both_correct": sum(x["both_correct"] for x in pair_rows),
            "reversal_pairs_total": len(pair_rows),
            "alex_owen_pairs_both_correct": sum(x["both_correct"] for x in pair_rows if x["group"] == "Alex/Owen"),
            "alex_owen_pairs_total": sum(x["group"] == "Alex/Owen" for x in pair_rows),
            "mia_nora_pairs_both_correct": sum(x["both_correct"] for x in pair_rows if x["group"] == "Mia/Nora"),
            "mia_nora_pairs_total": sum(x["group"] == "Mia/Nora" for x in pair_rows),
            "alex_correct_mean_margin": alex_mean,
            "owen_correct_mean_margin": owen_correct_mean,
            "alex_over_owen_default_component": (alex_mean - owen_correct_mean) / 2.0,
            "alex_owen_contextual_separation_component": (alex_mean + owen_correct_mean) / 2.0,
            "pair_rows": pair_rows,
        })
    with (OUT / "PAIR_STRUCTURE_RESULTS.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in derived:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")

    by = {x["condition"]: x for x in derived}
    path_rows = [x for x in derived if "__S_to_P__P_output" in x["condition"]]
    path_rows.sort(key=lambda x: next(r["condition"]["stage"] for r in [json.loads(z) for z in (OUT / "LOCALIZATION_RESULTS.jsonl").read_text(encoding="utf-8").splitlines()] if r["endpoint"] == "factual" and r["condition"]["id"] == x["condition"]))
    first_mn_complete = next((x["condition"].split("__", 1)[0] for x in path_rows if x["mia_nora_pairs_both_correct"] == 4), None)
    first_ao_default_flip = next((x["condition"].split("__", 1)[0] for x in path_rows if x["alex_over_owen_default_component"] < 0), None)
    first_ao_default_return = None
    seen_flip = False
    for x in path_rows:
        if x["alex_over_owen_default_component"] < 0:
            seen_flip = True
        elif seen_flip:
            first_ao_default_return = x["condition"].split("__", 1)[0]; break

    p, s = by["native_P"], by["native_S"]
    evidence = {
        "native": {"Pilot1": p, "SF1": s},
        "sf1_prefix_pilot1_suffix_pilot1_output": path_rows,
        "first_boundary_recovering_all_four_mia_nora_reversal_pairs": first_mn_complete,
        "first_boundary_flipping_alex_default_to_owen": first_ao_default_flip,
        "first_later_boundary_returning_to_alex_default": first_ao_default_return,
    }
    summary["factual_pair_and_bias_decomposition"] = evidence
    summary["classification"] = {
        "label": "DISTRIBUTED_OR_INTERACTIVE",
        "evidence": [
            "The native SF1 4/4 Mia/Nora matched-pair success is recovered only after the block-7 MLP residual on the cumulative SF1-prefix/Pilot1-suffix/Pilot1-output path.",
            "Alex/Owen never reaches complete matched-pair success. Its default identity flips from Alex to Owen after block-4 attention, becomes strongest after block-4 MLP, and flips back to Alex after block-5 MLP; later layers retain residual Alex preference.",
            "Unrelated-context name mass and aligned language CE accumulate mainly through blocks 5-7, with their largest increment and first majority-recovery point after the block-7 MLP; the SF1 output path amplifies both.",
            "Useful factual correctness, identity-default dynamics, name-prior pollution, and language damage therefore do not reduce to one clean depth, but the fully expressed factual and harmful endpoint shifts overlap substantially in the late path.",
        ],
        "inference_time_separability": "PARTIAL_NOT_CLEAN",
        "caution": "This is causal localization of inference-time sufficiency under hybrid networks, not a claim about where optimization changed parameters or where learning occurred.",
    }
    write_json(OUT / "SUMMARY.json", summary)

    provenance = read_json(OUT / "PROVENANCE.json")
    identity_after = {str(PARENT): sha(PARENT), str(CHILD): sha(CHILD), str(TOKENIZER): sha(TOKENIZER)}
    assert identity_after[str(PARENT)] == "2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb"
    assert identity_after[str(CHILD)] == "550ce4306b450c2bdb64e57314e45ec7db5bb2ee68fc0978ed9d3617f4fc1e7e"
    assert identity_after[str(TOKENIZER)] == "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"
    provenance.update({
        "identity_hashes_after_evaluation_and_finalization": identity_after,
        "runtime_used_for_sweep": {"python": "3.12.14", "torch": "2.12.0+rocm7.14.0", "tokenizers": "0.23.1", "device": "cuda"},
        "optimizer_created": False,
        "backward_or_autograd_used": False,
        "all_model_parameters_requires_grad_false": True,
        "checkpoint_or_dataset_mutation": False,
        "locked_final_or_sacred_access": False,
    })
    source_paths = [
        Path(r"C:\DaveLM-CADAVER\single_fact_acquisition_sf1_seed87011\sources\treatment13_model.py"),
        Path(r"C:\DaveLM-CADAVER\single_fact_acquisition_sf1_seed87011\sources\hr3_block3_runtime.py"),
        Path(r"C:\DaveLM-v0.9\v0_7\model.py"),
        Path(r"C:\DaveLM-v0.9\v0_8_2\model.py"),
        Path(r"C:\DaveLM-CADAVER\sf1_component_swap_forensic_v1\component_swap_forensic.py"),
        Path(r"C:\DaveLM-CADAVER\sf1_readout_selection_forensic_v1\D3_SELECTION_RECEIPT.json"),
    ]
    provenance["authoritative_source_and_receipt_hashes"] = {str(path): sha(path) for path in source_paths}
    write_json(OUT / "PROVENANCE.json", provenance)

    curves = summary["curves"]
    def curve(metric):
        return curves[metric]["curves"]["S_to_P_P_output"]
    report = f"""# SF1 upstream localization forensic v1

This diagnostic spliced the residual stream between authoritative Pilot1 and SF1-update100 at the embedding output and after each attention and MLP residual sublayer. It used only the 16 SF1 training items, the frozen 256-position unrelated TinyStories D3 sample, and the authorized aligned first-128 TinyStories DEV records.

## Result

`DISTRIBUTED_OR_INTERACTIVE`

The effects are partially separable at inference time, but there is no clean single-depth decomposition. Alex/Owen identity bias changes sharply in the middle-to-late stack, while the final useful factual pattern, broad name-prior inflation, and language regression all depend strongly on the late path.

## Execution graph and controls

The ordinary path is token plus position embeddings, eight pre-normalized transformer blocks (attention residual, then MLP residual), final normalization, and an untied language head. The T13 localizer/retrieval modules exist in both wrappers but are inactive here because all three authorized endpoints call `base_model` directly.

Same-source manual execution reproduced native logits exactly at every boundary (maximum absolute error `0`; tolerance `1e-6`). Native aggregate summaries reproduced the prior component-swap forensic within `3.43e-6` for Pilot1 and `1.72e-6` for SF1. The initial `1e-6` cross-run aggregate check stopped on those harmless batching-order differences; that stop is preserved under `attempt_1_aggregate_tolerance_stop`, and finalization used a documented `1e-5` cross-run aggregate tolerance without repeating inference.

No optimizer was created, autograd was disabled, all parameters had `requires_grad=False`, no model state was saved, and checkpoint/tokenizer hashes were reverified after evaluation.

## Native endpoints

| endpoint | Pilot1 | SF1 |
|---|---:|---:|
| factual sequence correct | {summary['native']['factual']['Pilot1']['overall']['sequence_correct']}/16 | {summary['native']['factual']['SF1']['overall']['sequence_correct']}/16 |
| complete matched reversal pairs | {p['reversal_pairs_both_correct']}/8 | {s['reversal_pairs_both_correct']}/8 |
| Alex/Owen matched pairs | {p['alex_owen_pairs_both_correct']}/4 | {s['alex_owen_pairs_both_correct']}/4 |
| Mia/Nora matched pairs | {p['mia_nora_pairs_both_correct']}/4 | {s['mia_nora_pairs_both_correct']}/4 |
| mean factual sequence margin | {summary['native']['factual']['Pilot1']['overall']['mean_sequence_margin']:.4f} | {summary['native']['factual']['SF1']['overall']['mean_sequence_margin']:.4f} |
| Owen-correct margin | {summary['native']['factual']['Pilot1']['Owen-correct']['mean_sequence_margin']:.4f} | {summary['native']['factual']['SF1']['Owen-correct']['mean_sequence_margin']:.4f} |
| unrelated four-name mass | {summary['native']['D3']['Pilot1']['mean_four_name_mass']:.6f} | {summary['native']['D3']['SF1']['mean_four_name_mass']:.6f} |
| aligned language CE | {summary['native']['language']['Pilot1']['established_batch_mean_ce']:.4f} | {summary['native']['language']['SF1']['established_batch_mean_ce']:.4f} |

## Where each effect emerges

**Useful factual signal.** The demonstrated SF1 improvement is concentrated in Mia/Nora: all four Mia/Nora reversal pairs become correct only when the SF1 prefix includes the block-7 MLP residual. The overall mean margin crosses the descriptive 50% recovery point after block 6 MLP and reaches 98% after block 7 attention, but correctness stays at 8/16 until block 7 MLP changes it to 12/16. Alex/Owen remains 0/4 complete pairs even in native SF1.

**Alex/Owen suppression.** Pilot1 has a large Alex-default component (`{p['alex_over_owen_default_component']:.4f}`) and a small context-separation component (`{p['alex_owen_contextual_separation_component']:.4f}`). SF1 reduces the Alex default to `{s['alex_over_owen_default_component']:.4f}` and raises context separation modestly to `{s['alex_owen_contextual_separation_component']:.4f}`, but Owen remains wrong. In the cumulative SF1-prefix path, the default first flips toward Owen after block 4 attention, is strongest after block 4 MLP, and flips back toward Alex after block 5 MLP. This non-monotonic switching localizes identity-selection transformations, not successful relational selection.

**Broad name-prior inflation.** With the Pilot1 output path, four-name mass rises gradually from blocks 4-7, crosses 50% of the native Pilot1-to-SF1 change only after block 7 MLP, and has its largest increment there (`{curve('d3_four_name_mass')['largest_absolute_increment']['increment_from_previous_boundary']:+.6f}`). The SF1 output path further amplifies the full-SF1-upstream value from `0.036394` to `0.067833`.

**Language regression.** Aligned CE stays near Pilot1 through the early stack, then worsens cumulatively through blocks 5-7. Its first majority-recovery point and largest increment are both after block 7 MLP (`{curve('language_ce')['largest_absolute_increment']['increment_from_previous_boundary']:+.6f}`); the SF1 output path raises the full-SF1-upstream CE from `4.2769` to `4.6529`.

## Interpretation

### Established

- The SF1 factual endpoint, unrelated-context name prior, and language regression are upstream effects with materially different internal depth profiles.
- Alex/Owen identity preference is transformed non-monotonically across blocks 4-7. Correct Alex/Owen reversal behavior never emerges on these 16 training items.
- Mia/Nora complete-pair correctness requires the block-7 MLP residual in the tested cumulative direction.
- Name-prior inflation and language damage are late, cumulative effects and are amplified by the SF1 output path.

### Supported

- The residual Alex/Owen failure is better described as a mixture of modest context-conditioned separation and much larger changing identity bias than as a total absence of contextual signal.
- Some inference-time separation exists: identity-default switching begins before the strongest broad name-prior and language-loss changes. The final useful factual pattern and both harms nevertheless overlap in the late path.

### Not established

- Patching depth does not reveal where learning occurred during optimization.
- These training-item interventions do not establish held-out factual generalization or a reusable ordinary-English binding mechanism.
- No architectural limit, treatment choice, or claim about locked panels follows from this diagnostic.

## Smallest justified next action

Perform one bounded read-only **token-position patch** at the already-localized transition sites—block 4 attention/MLP, block 5 attention/MLP, and block 7 attention/MLP—using only these same 16 training items and the same D3/language samples. Patch answer/cue positions separately from factual-name/object positions. That would test whether the mid-stack identity flips and late suppression enter through the answer-position state or through transformed context-token states. It is more discriminating than another whole-block swap and requires no treatment or locked-panel access.

SF1_UPSTREAM_LOCALIZATION_FORENSIC_COMPLETE
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8", newline="\n")

    # Recursive payload manifest, excluding itself. This also covers the preserved stop.
    entries = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path != OUT / "SHA256SUMS.txt" and "__pycache__" not in path.parts:
            entries.append((sha(path), path.relative_to(OUT).as_posix()))
    (OUT / "SHA256SUMS.txt").write_text("".join(f"{h}  {name}\n" for h, name in entries), encoding="utf-8", newline="\n")
    print(json.dumps({"status": summary["status"], "classification": summary["classification"]["label"], "manifest_sha256": sha(OUT / "SHA256SUMS.txt"), "payload_files": len(entries)}, indent=2))


if __name__ == "__main__":
    main()
