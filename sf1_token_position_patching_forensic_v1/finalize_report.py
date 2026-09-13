"""Finalize interpretation and seal persisted token-position results only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(r"C:\DaveLM-CADAVER")
OUT = ROOT / "sf1_token_position_patching_forensic_v1"


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
    baseline = read_json(OUT / "BASELINE_REPRODUCTION.json")
    assert baseline["pass"]
    full_rows = [json.loads(x) for x in (OUT / "POSITION_PATCH_RESULTS.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    full_factual = {x["condition"]["id"]: x["metrics"] for x in full_rows if x["endpoint"] == "factual"}
    factual = {x["condition"]["id"]: x for x in summary["position_effects"]["factual"]}
    d3 = {x["condition"]["id"]: x for x in summary["position_effects"]["D3"]}
    language = {x["condition"]["id"]: x for x in summary["position_effects"]["language"]}

    def F(site, group, direction="S_into_P", output="P"):
        return factual[f"{site}__{direction}__{group}__{output}_output"]

    def D(site, group, direction="S_into_P", output="P"):
        return d3[f"{site}__{direction}__{group}__{output}_output"]

    def L(site, group, direction="S_into_P", output="P"):
        return language[f"{site}__{direction}__{group}__{output}_output"]

    b4a = "after_block_4_attention_residual"; b4m = "after_block_4_mlp_residual"
    b5a = "after_block_5_attention_residual"; b5m = "after_block_5_mlp_residual"
    b7a = "after_block_7_attention_residual"; b7m = "after_block_7_mlp_residual"
    key = {
        "block4": {
            "attention_answer_path": F(b4a, "answer_generation_path"), "attention_all": F(b4a, "all_positions"),
            "mlp_final_predictive": F(b4m, "final_predictive"), "mlp_answer_path": F(b4m, "answer_generation_path"), "mlp_all": F(b4m, "all_positions"),
        },
        "block5": {
            "attention_answer_path": F(b5a, "answer_generation_path"), "attention_all": F(b5a, "all_positions"),
            "mlp_final_predictive": F(b5m, "final_predictive"), "mlp_answer_path": F(b5m, "answer_generation_path"), "mlp_all": F(b5m, "all_positions"),
            "mlp_answer_path_sf1_output": F(b5m, "answer_generation_path", output="S"),
        },
        "block7_factual": {
            "attention_answer_path": F(b7a, "answer_generation_path"), "attention_all": F(b7a, "all_positions"),
            "mlp_final_predictive": F(b7m, "final_predictive"), "mlp_answer_path": F(b7m, "answer_generation_path"), "mlp_all": F(b7m, "all_positions"),
            "necessity_final_predictive": F(b7m, "final_predictive", direction="P_into_S", output="S"),
            "necessity_answer_path": F(b7m, "answer_generation_path", direction="P_into_S", output="S"),
        },
        "block7_D3": {
            "predictive_P_output": D(b7m, "final_predictive"), "prior_P_output": D(b7m, "all_prior_context"), "all_P_output": D(b7m, "all_positions"),
            "predictive_S_output": D(b7m, "final_predictive", output="S"), "all_S_output": D(b7m, "all_positions", output="S"),
            "necessity_predictive": D(b7m, "final_predictive", direction="P_into_S", output="S"),
        },
        "block7_language": {
            "final_P_output": L(b7m, "final_predictive"), "prior_P_output": L(b7m, "all_prior_context"), "all_P_output": L(b7m, "all_positions"),
            "prior_S_output": L(b7m, "all_prior_context", output="S"), "all_S_output": L(b7m, "all_positions", output="S"),
            "necessity_final": L(b7m, "final_predictive", direction="P_into_S", output="S"), "necessity_prior": L(b7m, "all_prior_context", direction="P_into_S", output="S"),
        },
    }
    key["block7_factual"]["mlp_final_predictive"]["first_correct"] = full_factual[f"{b7m}__S_into_P__final_predictive__P_output"]["overall"]["first_correct"]
    key["block7_factual"]["mlp_answer_path"]["first_correct"] = full_factual[f"{b7m}__S_into_P__answer_generation_path__P_output"]["overall"]["first_correct"]
    key["block7_factual"]["mlp_all"]["first_correct"] = full_factual[f"{b7m}__S_into_P__all_positions__P_output"]["overall"]["first_correct"]
    classification = {
        "label": "ANSWER_GENERATION_STATE_DOMINANT_WITH_OUTPUT_PATH_INTERACTION; LANGUAGE_DAMAGE_DISTRIBUTED_ACROSS_PREDICTIVE_POSITIONS",
        "short_labels": ["ANSWER_STATE_DOMINANT", "OUTPUT_PATH_INTERACTION", "POSITION_SEPARABLE_FOR_LANGUAGE_BREADTH"],
        "answer_to_questions": {
            "block4_owen_flip": "The sequence-likelihood Owen flip is carried primarily by answer-generation states, especially candidate-prefix states after the first prediction. The final prompt predictor alone and every isolated static subject/object/predicate group fail to flip the sequence preference.",
            "block5_alex_return": "The return occurs within the answer-generation trajectory between block-5 attention and block-5 MLP. No isolated semantic prompt span reproduces it; the SF1 output path further amplifies the return toward Alex.",
            "block7_mia_nora": "The final prompt predictor is sufficient for 12/16 first-token correctness, but full answer-generation states are necessary and sufficient for 12/16 sequence correctness, 12/16 exact answer+EOS, and all 4/4 Mia/Nora reversal pairs.",
            "block7_name_prior": "The current predictive state is sufficient and approximately necessary for D3 four-name inflation. Earlier context states have zero effect after the final MLP. The SF1 output path amplifies the state effect.",
            "block7_language": "Regression is distributed over the many token states that make aligned next-token predictions. Patching only the last record position does not cause the regression; patching the preceding scored states carries most or more than all of it under the nonlinear hybrid.",
            "separability": "Factual sequence recovery and D3 name-prior pollution share the answer/current-predictive route, so they are not token-route separable at block 7. Language damage is broader across token positions, giving partial positional separation but not an independent useful-only route.",
        },
        "interpretive_limit": "Activation patching establishes inference-time sufficiency/necessity in these hybrid networks, not where optimization changed weights and not held-out generalization.",
    }
    summary["status"] = "SF1_TOKEN_POSITION_PATCHING_FORENSIC_COMPLETE"
    summary["classification"] = classification
    summary["key_evidence"] = key
    write_json(OUT / "SUMMARY.json", summary)

    token_map = read_json(OUT / "TOKEN_INDEX_MAP.json")["records"]
    prompt_lengths = sorted({x["prompt_token_count"] for x in token_map})
    subject_lengths = sorted({len(x["token_indices_without_bos"]["factual_subject"]) for x in token_map})
    object_lengths = sorted({len(x["token_indices_without_bos"]["factual_object_first"]) for x in token_map})
    predicate_lengths = sorted({len(x["token_indices_without_bos"]["predicate_fact"]) for x in token_map})
    p_native = summary["native"]; p_f = p_native["factual"]["Pilot1"]; s_f = p_native["factual"]["SF1"]

    report = f"""# SF1 token-position patching forensic v1

## Headline result

`ANSWER_GENERATION_STATE_DOMINANT_WITH_OUTPUT_PATH_INTERACTION; LANGUAGE_DAMAGE_DISTRIBUTED_ACROSS_PREDICTIVE_POSITIONS`

SF1's factual sequence behavior and unrelated-context name-prior pollution are both carried through answer/current-predictive token states. They are therefore **not cleanly separable token routes** at block 7. Language regression is broader: it is distributed across the tokenwise predictive states used throughout ordinary causal scoring rather than concentrated at the final record position.

## Frozen construction and execution graph

Before either checkpoint was loaded, the final rendered prompts were parsed and encoded with the authoritative tokenizer. All 16 prompts round-tripped exactly and matched their frozen token IDs. Prompt lengths are {prompt_lengths}; every subject/name span is three tokens, object spans are {object_lengths} tokens per occurrence, predicates are {predicate_lengths} tokens per occurrence, and both exact candidate completions are four tokens. `TOKEN_INDEX_MAP.json` records every character span, token index, model position after BOS, token ID, and dynamic answer-generation range.

The ordinary path is token plus position embeddings, eight pre-normalized transformer blocks with attention and MLP residual sub-boundaries, final normalization, and an untied language head. T13 localization/retrieval modules are registered in the wrapper but inactive because this diagnostic calls only the ordinary `base_model` path.

At each frozen site, the donor and recipient were run to the same residual boundary, selected `[batch,time]` positions were replaced, and the recipient's downstream blocks were retained. Pilot1 and SF1 output paths were evaluated separately.

## Baseline and splice integrity

- Same-source patching across every site/group/source reproduced native logits exactly: maximum absolute error `{baseline['same_source']['max_abs_logit_error']}` under the frozen `1e-6` tolerance.
- Native endpoint reproduction differed from the prior forensic by at most `{max(x['max_abs_error'] for x in baseline['native_prior']['errors'].values()):.3g}` under the prospectively frozen `1e-5` cross-run aggregate tolerance.
- All-position token patches reproduced the corresponding prior whole-boundary interventions within `{baseline['all_positions_upstream_forensic']['max_aggregate_error']:.3g}`.
- No tolerance was changed after inference.

Native factual behavior reproduced: Pilot1 `{p_f['overall']['sequence_correct']}/16` sequence-correct and SF1 `{s_f['overall']['sequence_correct']}/16`; SF1 retained `4/4` Mia/Nora complete reversal pairs and `0/4` Alex/Owen pairs.

## 1. Block 4: temporary Owen-favoring flip

The earlier “Owen flip” is specifically a **full sequence-likelihood** effect. At block-4 MLP, SF1 states patched only at the final prompt predictor leave a strong Alex sequence default (`{key['block4']['mlp_final_predictive']['alex_default']:.3f}`) and an Owen-correct mean margin of `{key['block4']['mlp_final_predictive']['owen_margin']:.3f}`. Patching the whole answer-generation path changes the default to Owen (`{key['block4']['mlp_answer_path']['alex_default']:.3f}`) and raises the Owen-correct margin to `{key['block4']['mlp_answer_path']['owen_margin']:.3f}`, nearly matching all-position patching (`{key['block4']['mlp_all']['alex_default']:.3f}`, `{key['block4']['mlp_all']['owen_margin']:.3f}`).

The first-token decision itself still favors Alex. The flip emerges in the later candidate-token likelihoods. Subject, either object occurrence, predicate/cue, or remaining context patched alone does not reproduce it. Thus the block-4 sequence flip is answer-generation-state dominant; it is not localized to one semantic prompt span.

## 2. Block 5: return toward Alex

With the Pilot1 output path, the answer-generation state carries an Owen default after block-5 attention (`{key['block5']['attention_answer_path']['alex_default']:.3f}`) but is nearly neutral after block-5 MLP (`{key['block5']['mlp_answer_path']['alex_default']:.3f}`). The corresponding all-position default moves from `{key['block5']['attention_all']['alex_default']:.3f}` to `{key['block5']['mlp_all']['alex_default']:.3f}`. Using the SF1 output path moves the block-5-MLP answer trajectory further toward Alex (`{key['block5']['mlp_answer_path_sf1_output']['alex_default']:.3f}`).

No isolated subject, object, or predicate span explains this transition. It reflects a change in the answer-generation trajectory plus an output-path interaction. This is identity-selection behavior, not successful Alex/Owen reversal: complete Alex/Owen pairs remain zero in native SF1.

## 3. Block 7 MLP: factual recovery

| SF1 state injected into Pilot1 | first-token correct | sequence correct | exact answer+EOS | all complete pairs | Mia/Nora pairs |
|---|---:|---:|---:|---:|---:|
| final prompt predictor only | {key['block7_factual']['mlp_final_predictive']['first_correct']}/16 | {key['block7_factual']['mlp_final_predictive']['sequence_correct']}/16 | {key['block7_factual']['mlp_final_predictive']['exact_answer_eos']}/16 | {key['block7_factual']['mlp_final_predictive']['complete_pairs']}/8 | {key['block7_factual']['mlp_final_predictive']['MN_complete_pairs']}/4 |
| full answer-generation path | {key['block7_factual']['mlp_answer_path']['first_correct']}/16 | {key['block7_factual']['mlp_answer_path']['sequence_correct']}/16 | {key['block7_factual']['mlp_answer_path']['exact_answer_eos']}/16 | {key['block7_factual']['mlp_answer_path']['complete_pairs']}/8 | {key['block7_factual']['mlp_answer_path']['MN_complete_pairs']}/4 |
| all positions | {key['block7_factual']['mlp_all']['first_correct']}/16 | {key['block7_factual']['mlp_all']['sequence_correct']}/16 | {key['block7_factual']['mlp_all']['exact_answer_eos']}/16 | {key['block7_factual']['mlp_all']['complete_pairs']}/8 | {key['block7_factual']['mlp_all']['MN_complete_pairs']}/4 |

The final prompt predictor is sufficient for the SF1 first-token result, but not the complete four-token name plus EOS. Patching the full answer-generation trajectory restores `12/16` sequence decisions, `12/16` exact completions, and all four Mia/Nora reversal pairs, exactly matching all-position patching. In the reverse direction, replacing SF1's answer trajectory with Pilot1 states reduces behavior to `{key['block7_factual']['necessity_answer_path']['sequence_correct']}/16`, `{key['block7_factual']['necessity_answer_path']['exact_answer_eos']}/16` exact, and `{key['block7_factual']['necessity_answer_path']['MN_complete_pairs']}/4` Mia/Nora pairs. Static context-only patches at this final tokenwise MLP cannot affect the answer position and leave native SF1 behavior unchanged.

## 4. Block 7 MLP: unrelated name prior

On the exact frozen 256-position D3 sample, patching only the current predictive state produces four-name mass `{key['block7_D3']['predictive_P_output']['four_name_mass']:.6f}` with the Pilot1 output path, exactly the all-position value. Patching all earlier context states produces the Pilot1 baseline `{key['block7_D3']['prior_P_output']['four_name_mass']:.6f}`. The SF1 output path amplifies the same predictive state to `{key['block7_D3']['predictive_S_output']['four_name_mass']:.6f}`, the native SF1 value. Reverse replacement of the predictive state reduces native-SF1 mass to `{key['block7_D3']['necessity_predictive']['four_name_mass']:.6f}`.

The late D3 prior is therefore localized to the current predictive state and materially amplified by the SF1 output path.

## 5. Block 7 MLP: language regression

Pilot1 aligned CE is `{p_native['language']['Pilot1']['established_batch_mean_ce']:.4f}` and SF1 is `{p_native['language']['SF1']['established_batch_mean_ce']:.4f}`. Injecting SF1 only at each record's final predictor yields CE `{key['block7_language']['final_P_output']['ce']:.4f}`; injecting SF1 into the preceding valid/scored positions yields `{key['block7_language']['prior_P_output']['ce']:.4f}`; all valid positions yield `{key['block7_language']['all_P_output']['ce']:.4f}` with the Pilot1 output path. In reverse, replacing those preceding SF1 positions with Pilot1 states reduces CE to `{key['block7_language']['necessity_prior']['ce']:.4f}`, while replacing only the final record predictor leaves CE at `{key['block7_language']['necessity_final']['ce']:.4f}`.

The language regression is spread across the many token states that perform ordinary next-token prediction. “All prior context” here means all valid positions before each record's final EOS-predicting position; it is not evidence for a single semantic context span. Recovery fractions above 100% or non-additive totals are hybrid-network interactions, not literal shares.

## Evidence ledger

### Established

- The block-4 sequence-likelihood Owen flip is carried mainly by answer-generation/candidate-prefix states, not the final first-answer predictor alone.
- The block-5 return toward Alex occurs within that answer-generation trajectory and is further shifted by the SF1 output path.
- At block-7 MLP, the final predictor carries first-token factual choice and D3 name-prior inflation; later answer-prefix states are additionally required for the full candidate sequence, exact EOS behavior, and Mia/Nora reversal recovery.
- Block-7 language damage is distributed across many causal predictive positions.

### Supported

- Useful SF1 factual sequence behavior and harmful name-prior inflation share a predictive/answer-state route and cannot be cleanly separated by the tested token groups.
- Language damage is positionally broader, giving partial separation by breadth but no demonstrated useful-only route.
- Static subject, object, and predicate states can influence middle-layer hybrids, but no isolated static group is sufficient for the decisive SF1 factual endpoint.

### Not established

- The patches do not identify where training changed the mechanism.
- Results on these 16 training records do not establish held-out factual generalization.
- Non-additive hybrid effects cannot be interpreted as a linear allocation of causal responsibility.
- No treatment choice, architecture conclusion, or locked-panel claim follows.

The bounded diagnostic ends here. No training treatment is proposed or executed.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8", newline="\n")

    provenance = read_json(OUT / "PROVENANCE.json")
    source_paths = [
        OUT / "build_protocol.py", OUT / "run_token_position_patching.py", OUT / "finalize_report.py",
        ROOT / "single_fact_acquisition_sf1_seed87011" / "sources" / "treatment13_model.py",
        ROOT / "single_fact_acquisition_sf1_seed87011" / "sources" / "hr3_block3_runtime.py",
        Path(r"C:\DaveLM-v0.9\v0_7\model.py"), Path(r"C:\DaveLM-v0.9\v0_8_2\model.py"),
        ROOT / "sf1_upstream_localization_forensic_v1" / "SUMMARY.json",
        ROOT / "sf1_upstream_localization_forensic_v1" / "LOCALIZATION_RESULTS.jsonl",
    ]
    provenance["authoritative_and_created_source_hashes"] = {str(path): sha(path) for path in source_paths}
    provenance["final_classification"] = classification["label"]
    provenance["finalization_loaded_checkpoint"] = False
    write_json(OUT / "PROVENANCE.json", provenance)

    entries = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path != OUT / "SHA256SUMS.txt" and "__pycache__" not in path.parts:
            entries.append((sha(path), path.relative_to(OUT).as_posix()))
    (OUT / "SHA256SUMS.txt").write_text("".join(f"{h}  {name}\n" for h, name in entries), encoding="utf-8", newline="\n")
    print(json.dumps({"status": summary["status"], "classification": classification["label"], "payload_files": len(entries), "manifest_sha256": sha(OUT / "SHA256SUMS.txt"), "report_sha256": sha(OUT / "REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
