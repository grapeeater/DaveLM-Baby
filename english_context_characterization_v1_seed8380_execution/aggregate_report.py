"""Aggregate the sealed prospective battery; importing performs no IO or inference.

Call aggregate(execution_dir, frozen_dir) only after all authorized scores exist.
Only standard-library data processing is used. Checkpoints are never opened.
"""

from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path


DEFAULT_FROZEN = Path(r"C:\DaveLM-CADAVER\english_context_characterization_v1_seed8380")
CHECKPOINTS = ("graduate", "pilot0", "pilot1")
LABELS = {"graduate": "Graduate", "pilot0": "Pilot 0", "pilot1": "Pilot 1"}
HASHES = {
    "graduate": "fed298748e62def6f1751ef1bb7df0cdec51d53fac6e4c86e8c00a95dccdd430",
    "pilot0": "769bd01efd28e7888064c0d5fe1dd0d85a4344e2039aef04bcbf7d9e906f47f5",
    "pilot1": "2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb",
}
STRATA = (
    ("possession", "core"), ("location", "core"),
    ("naming", "core"), ("possession", "frame_holdout"),
    ("location", "frame_holdout"), ("possession", "qa"), ("location", "qa"),
)
STRATUM_LABELS = {
    "possession/core": "Possession primary cloze",
    "location/core": "Location primary cloze",
    "naming/core": "Naming diagnostic",
    "possession/frame_holdout": "Possession alternate-frame diagnostic",
    "location/frame_holdout": "Location alternate-frame diagnostic",
    "possession/qa": "Possession QA diagnostic",
    "location/qa": "Location QA diagnostic",
}
UNIVERSES = {"naming": 18, "possession": 36, "location": 24}
SAMPLES = {"naming": 8, "possession": 16, "location": 16}


def _load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, content):
    Path(path).write_bytes(content.encode("utf-8"))


def _describe(values):
    values = sorted(float(v) for v in values)
    if not values:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
    return {"n": len(values), "min": values[0], "median": statistics.median(values),
            "mean": statistics.fmean(values), "max": values[-1]}


def exact_finite_population_interval(N, n, x):
    """Exact specified hypergeometric inversion, with rational tail arithmetic."""
    if not (0 <= x <= n <= N):
        raise ValueError("Invalid finite-population counts")
    denom = math.comb(N, n)
    retained = []
    for K in range(N + 1):
        support = range(max(0, n - (N - K)), min(n, K) + 1)
        upper = sum(math.comb(K, k) * math.comb(N - K, n - k)
                    for k in support if k >= x)
        lower = sum(math.comb(K, k) * math.comb(N - K, n - k)
                    for k in support if k <= x)
        if Fraction(upper, denom) >= Fraction(1, 40) and Fraction(lower, denom) >= Fraction(1, 40):
            retained.append(K)
    if not retained:
        raise ValueError("Hypergeometric inversion unexpectedly empty")
    return {"confidence_level": 0.95, "N": N, "n": n, "x": x,
            "retained_K": retained, "lower": min(retained) / N,
            "upper": max(retained) / N,
            "scope": "Only this explicitly enumerated finite lexical-family universe under this fixed frame; not general English or training-seed uncertainty."}


def _close(actual, expected, label):
    if actual is None or not math.isfinite(float(actual)) or not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"Inconsistent score: {label}: {actual!r} versus {expected!r}")


def _validate_score(record, score, contextual):
    if len(score["candidates"]) != 2:
        raise ValueError("Expected two candidate scores")
    full, word = [], []
    for i, candidate in enumerate(score["candidates"]):
        if candidate["candidate_token_ids"] != record["candidate_token_ids"][i]:
            raise ValueError("Candidate token IDs differ from sealed record")
        logps = candidate["token_log_probabilities"]
        if len(logps) != 3 or any(not math.isfinite(float(v)) or v > 1e-12 for v in logps):
            raise ValueError("Invalid per-token log-probabilities")
        _close(candidate["conditional_log_likelihood"], sum(logps), "candidate LL")
        _close(candidate["word_only_log_likelihood"], sum(logps[:2]), "word LL")
        full.append(candidate["conditional_log_likelihood"])
        word.append(candidate["word_only_log_likelihood"])
    d, wd = full[0] - full[1], word[0] - word[1]
    _close(score["candidate0_minus_candidate1"], d, "candidate difference")
    _close(score["word_only_candidate0_minus_candidate1"], wd, "word difference")
    if score["tie"] != (d == 0) or score["word_only_tie"] != (wd == 0):
        raise ValueError("Tie flags inconsistent")
    ci = record["correct_index"] if contextual else None
    if score["correct_candidate_index"] != ci:
        raise ValueError("Scoring answer index differs from frozen answer")
    if contextual:
        for prefix, dif in (("", d), ("word_only_", wd)):
            margin = dif if ci == 0 else -dif
            _close(score[f"{prefix}correct_minus_incorrect_margin"], margin, "margin")
            if score[f"{prefix}item_correct"] != (margin > 0):
                raise ValueError("Correctness flag inconsistent")
    else:
        for key in ("correct_minus_incorrect_margin", "word_only_correct_minus_incorrect_margin", "item_correct", "word_only_item_correct"):
            if score[key] is not None:
                raise ValueError("Prior must not have an answer correctness score")
    mx = max(full)
    logmass = mx + math.log(sum(math.exp(v - mx) for v in full))
    _close(score["candidate_pair_log_probability_mass"], logmass, "log pair mass")
    _close(score["candidate_pair_probability_mass"], math.exp(logmass), "pair mass")
    if score["completed_input_length_including_bos"] != len(record["prompt_token_ids"]) + 4:
        raise ValueError("Completed input length inconsistent")


def _mode_stats(rows, prefix):
    margins = [row["score"][f"{prefix}correct_minus_incorrect_margin"] for row in rows]
    correct = [value > 0 for value in margins]
    groups = defaultdict(list)
    for row in rows:
        groups[row["record"]["reversal_pair_id"]].append(row)
    if len(rows) != 8 or len(groups) != 4:
        raise ValueError("Incomplete family/frame")
    pairs = []
    for pair_id, pair_rows in sorted(groups.items()):
        pair_rows.sort(key=lambda row: row["record"]["assignment"])
        if [r["record"]["assignment"] for r in pair_rows] != [0, 1]:
            raise ValueError("Invalid reversal assignments")
        first, second = pair_rows
        if first["record"]["correct_index"] == second["record"]["correct_index"]:
            raise ValueError("Counterfactual pair does not reverse the correct answer")
        d0 = first["score"][f"{prefix}candidate0_minus_candidate1"]
        d1 = second["score"][f"{prefix}candidate0_minus_candidate1"]
        orientation = 1 if first["record"]["correct_index"] == 0 else -1
        m0 = first["score"][f"{prefix}correct_minus_incorrect_margin"]
        m1 = second["score"][f"{prefix}correct_minus_incorrect_margin"]
        pairs.append({"reversal_pair_id": pair_id,
                      "both_correct": m0 > 0 and m1 > 0,
                      "d_assignment0": d0, "d_assignment1": d1,
                      "raw_d_assignment0_minus_assignment1": d0 - d1,
                      "answer_oriented_reversal_change": orientation * (d0 - d1),
                      "assignment0_margin": m0, "assignment1_margin": m1})
    order_complete = {}
    for order in (0, 1):
        subset = [r for r in rows if r["record"]["fact_order"] == order]
        order_complete[str(order)] = all(r["score"][f"{prefix}correct_minus_incorrect_margin"] > 0 for r in subset)
    return {"n_items": 8, "item_correct": sum(correct), "ties": sum(v == 0 for v in margins),
            "complete": all(correct), "reversal_pairs": 4,
            "reversal_both_correct": sum(p["both_correct"] for p in pairs),
            "reversal_fraction": sum(p["both_correct"] for p in pairs) / 4,
            "mean_margin": statistics.fmean(margins), "minimum_margin": min(margins),
            "margin_distribution": _describe(margins),
            "fact_order_complete": order_complete, "matched_reversals": pairs}


def _family_results(context_rows):
    grouped = defaultdict(list)
    for row in context_rows:
        r = row["record"]
        grouped[(row["checkpoint"], r["task"], r["frame"], r["family_id"])].append(row)
    result = []
    for (cp, task, frame, fid), rows in sorted(grouped.items()):
        r = rows[0]["record"]
        keys = {(x["record"]["assignment"], x["record"]["query_index"], x["record"]["fact_order"]) for x in rows}
        if keys != set(itertools.product((0, 1), repeat=3)):
            raise ValueError("Family binary factors not complete")
        result.append({"checkpoint": cp, "task": task, "frame": frame, "family_id": fid,
                       "entities": r["entities"], "values": r["values"],
                       "full": _mode_stats(rows, ""), "word_only": _mode_stats(rows, "word_only_")})
    return result


def _stratum_summary(cp, task, frame, families, context_rows, prior_rows):
    fs = [f for f in families if (f["checkpoint"], f["task"], f["frame"]) == (cp, task, frame)]
    rows = [r for r in context_rows if (r["checkpoint"], r["record"]["task"], r["record"]["frame"]) == (cp, task, frame)]
    ps = [r for r in prior_rows if (r["checkpoint"], r["record"]["task"], r["record"]["frame"]) == (cp, task, frame)]
    if len(fs) != SAMPLES[task] or len(rows) != 8 * len(fs) or len(ps) != 2 * len(fs):
        raise ValueError(f"Wrong stratum size: {cp}/{task}/{frame}")
    result = {"task": task, "frame": frame, "n_families": len(fs), "n_items": len(rows),
              "n_logical_priors": len(ps)}
    for mode, prefix in (("full", ""), ("word_only", "word_only_")):
        successes = sum(f[mode]["complete"] for f in fs)
        hist = Counter(f[mode]["reversal_both_correct"] for f in fs)
        shifts = [p["answer_oriented_reversal_change"] for f in fs for p in f[mode]["matched_reversals"]]
        order_profiles = Counter("both" if f[mode]["fact_order_complete"]["0"] and f[mode]["fact_order_complete"]["1"]
                                 else "order0_only" if f[mode]["fact_order_complete"]["0"]
                                 else "order1_only" if f[mode]["fact_order_complete"]["1"] else "neither" for f in fs)
        result[mode] = {
            "complete_family_count": successes, "complete_family_proportion": successes / len(fs),
            "reversal_profile_counts_0_to_4": [hist.get(i, 0) for i in range(5)],
            "reversal_both_correct": sum(f[mode]["reversal_both_correct"] for f in fs),
            "reversal_pairs": 4 * len(fs),
            "mean_within_family_reversal_success": statistics.fmean(f[mode]["reversal_fraction"] for f in fs),
            "median_family_mean_margin": statistics.median(f[mode]["mean_margin"] for f in fs),
            "median_family_minimum_margin": statistics.median(f[mode]["minimum_margin"] for f in fs),
            "item_correct": sum(f[mode]["item_correct"] for f in fs),
            "item_accuracy": sum(f[mode]["item_correct"] for f in fs) / len(rows),
            "ties": sum(f[mode]["ties"] for f in fs),
            "secondary_finite_universe_interval": exact_finite_population_interval(UNIVERSES[task], len(fs), successes),
            "answer_oriented_reversal_change": _describe(shifts),
            "positive_answer_oriented_changes": sum(v > 0 for v in shifts),
            "fact_order_complete_profiles": {k: order_profiles.get(k, 0) for k in ("both", "order0_only", "order1_only", "neither")},
        }
        grouped_order = {}
        for factor in ("fact_order", "queried_fact_rank", "correct_candidate_mention_rank", "query_index"):
            grouped_order[factor] = {}
            for val in (0, 1):
                subset = [r for r in rows if r["record"][factor] == val]
                margins = [r["score"][f"{prefix}correct_minus_incorrect_margin"] for r in subset]
                grouped_order[factor][str(val)] = {"n": len(subset), "correct": sum(m > 0 for m in margins),
                                                  "ties": sum(m == 0 for m in margins), "margins": _describe(margins)}
        result[mode]["order_and_query_diagnostics"] = grouped_order
    result["word_vs_full"] = {
        "item_correctness_disagreements": sum(r["score"]["item_correct"] != r["score"]["word_only_item_correct"] for r in rows),
        "full_only_correct": sum(r["score"]["item_correct"] and not r["score"]["word_only_item_correct"] for r in rows),
        "word_only_correct": sum(r["score"]["word_only_item_correct"] and not r["score"]["item_correct"] for r in rows),
        "complete_family_disagreements": sum(f["full"]["complete"] != f["word_only"]["complete"] for f in fs),
    }
    result["candidate_pair_probability_mass"] = _describe(r["score"]["candidate_pair_probability_mass"] for r in rows)
    result["candidate_pair_log_probability_mass"] = _describe(r["score"]["candidate_pair_log_probability_mass"] for r in rows)
    result["candidate_log_likelihoods"] = _describe(c["conditional_log_likelihood"] for r in rows for c in r["score"]["candidates"])
    result["raw_score_source"] = "RAW_SCORES.jsonl: filter checkpoint, kind=contextual, record.task, record.frame; exact per-candidate and per-token scores retained there."
    priors = {}
    for mode, prefix in (("full", ""), ("word_only", "word_only_")):
        diffs = [p["score"][f"{prefix}candidate0_minus_candidate1"] for p in ps]
        prior_by_id = {p["record"]["prior_id"]: p for p in ps}
        agreements = 0
        comparable = 0
        for row in rows:
            d = row["score"][f"{prefix}candidate0_minus_candidate1"]
            pd = prior_by_id[row["record"]["prior_id"]]["score"][f"{prefix}candidate0_minus_candidate1"]
            if d != 0 and pd != 0:
                comparable += 1
                agreements += (d > 0) == (pd > 0)
        priors[mode] = {"candidate0_preferred": sum(v > 0 for v in diffs),
                        "candidate1_preferred": sum(v < 0 for v in diffs), "ties": sum(v == 0 for v in diffs),
                        "candidate0_minus_candidate1": _describe(diffs),
                        "absolute_candidate_difference": _describe(abs(v) for v in diffs),
                        "context_prior_preference_agreement": {"agree": agreements, "n_comparable": comparable},
                        "calibration_applied": False}
    result["query_only_priors"] = priors
    return result


def _paired(families):
    result = []
    for task, frame in STRATA:
        for first, second in itertools.combinations(CHECKPOINTS, 2):
            a = {f["family_id"]: f for f in families if (f["checkpoint"], f["task"], f["frame"]) == (first, task, frame)}
            b = {f["family_id"]: f for f in families if (f["checkpoint"], f["task"], f["frame"]) == (second, task, frame)}
            if set(a) != set(b):
                raise ValueError("Paired checkpoint families differ")
            entry = {"task": task, "frame": frame, "first": first, "second": second,
                     "difference_orientation": "second minus first", "n_families": len(a)}
            for mode in ("full", "word_only"):
                per_family = []
                for fid in sorted(a):
                    af, bf = a[fid][mode], b[fid][mode]
                    per_family.append({"family_id": fid, "first_complete": af["complete"], "second_complete": bf["complete"],
                                       "reversal_fraction_difference": bf["reversal_fraction"] - af["reversal_fraction"],
                                       "family_mean_margin_difference": bf["mean_margin"] - af["mean_margin"]})
                entry[mode] = {
                    "both_complete": sum(v["first_complete"] and v["second_complete"] for v in per_family),
                    "first_only_complete": sum(v["first_complete"] and not v["second_complete"] for v in per_family),
                    "second_only_complete": sum(v["second_complete"] and not v["first_complete"] for v in per_family),
                    "neither_complete": sum(not v["first_complete"] and not v["second_complete"] for v in per_family),
                    "complete_family_proportion_difference": sum(int(v["second_complete"]) - int(v["first_complete"]) for v in per_family) / len(per_family),
                    "median_paired_family_mean_margin_difference": statistics.median(v["family_mean_margin_difference"] for v in per_family),
                    "family_reversal_difference_distribution": _describe(v["reversal_fraction_difference"] for v in per_family),
                    "per_family": per_family,
                }
            result.append(entry)
    return result


def _shortcuts(frozen_items):
    output = {}
    for task, frame in STRATA:
        items = [r for r in frozen_items if (r["task"], r["frame"]) == (task, frame)]
        result = {}
        for rule in ("fixed_candidate0", "first_mentioned", "last_mentioned"):
            groups = defaultdict(list)
            correct_n = 0
            for item in items:
                pred = 0 if rule == "fixed_candidate0" else item["candidate_mention_order"][0 if rule == "first_mentioned" else 1]
                correct = pred == item["correct_index"]
                correct_n += correct
                groups[item["family_id"]].append((item, correct))
            complete, reversals = 0, 0
            for rows in groups.values():
                complete += all(v for _, v in rows)
                pairs = defaultdict(list)
                for item, correct in rows:
                    pairs[item["reversal_pair_id"]].append(correct)
                reversals += sum(all(v) for v in pairs.values())
            result[rule] = {"item_correct": correct_n, "n_items": len(items), "complete_family_count": complete,
                            "n_families": len(groups), "reversal_both_correct": reversals, "reversal_pairs": len(items) // 2}
        output[f"{task}/{frame}"] = result
    return output


def _csv(families):
    fields = ["checkpoint", "task", "frame", "family_id", "entities", "values"]
    for mode in ("full", "word_only"):
        fields += [f"{mode}_{key}" for key in ("complete", "item_correct", "ties", "reversal_both_correct", "reversal_fraction", "mean_margin", "minimum_margin", "fact_order0_complete", "fact_order1_complete")]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for family in families:
        row = {key: family[key] for key in ("checkpoint", "task", "frame", "family_id")}
        row["entities"] = "|".join(family["entities"])
        row["values"] = "|".join(family["values"])
        for mode in ("full", "word_only"):
            for key in ("complete", "item_correct", "ties", "reversal_both_correct", "reversal_fraction", "mean_margin", "minimum_margin"):
                row[f"{mode}_{key}"] = family[mode][key]
            for order in (0, 1):
                row[f"{mode}_fact_order{order}_complete"] = family[mode]["fact_order_complete"][str(order)]
        writer.writerow(row)
    return buffer.getvalue()


def _fmt(x):
    return "NA" if x is None else f"{x:.6g}"


def _pct(x):
    return f"{100 * x:.2f}%"


def _report(summary, binding):
    lines = ["# Frozen English context-sensitivity characterization", "",
             "This report characterizes the three specified checkpoints on the identical sealed battery. Raw complete-family counts/proportions, reversal profiles, and likelihood-margin summaries are the headline evidence. No pass/fail gates were added.", "",
             "The English scores use only the ordinary `base_model` causal-LM path. The separate synthetic reference uses the existing structurally assisted binding path. All candidate comparisons use the full leading-space word plus period; priors are reported without calibration. Exact ties count as incorrect.", "",
             "These observations concern the frozen naming, having/association, and location constructions. They do not measure general English capability, establish conversational competence, isolate block-protection effects, establish synthetic-to-English transfer, or prove an architectural/capacity limit.", "",
             "## Headline primary evidence", "",
             "Possession and location are separate endpoints. Possession queries an object to recover a person; location queries a person to recover a place. Differences can reflect query direction and wording as well as relation content.", "",
             "Reversal profile entries below count families with 0, 1, 2, 3, or 4 successful reversal pairs, in that order. Complete-family success requires all eight items correct, including both fact orders.", ""]
    def table_for(task, frame, mode="full"):
        key = f"{task}/{frame}"
        table = [f"### {STRATUM_LABELS[key]}" + (" — word-only" if mode == "word_only" else ""), "",
                 "| Checkpoint | Complete families | Proportion | Reversal profile [0,1,2,3,4] | Both-correct reversal pairs | Mean family reversal rate | Median family mean margin | Median family minimum margin |",
                 "|---|---:|---:|---|---:|---:|---:|---:|"]
        for cp in CHECKPOINTS:
            s = summary["checkpoints"][cp][key]
            m = s[mode]
            table.append(f"| {LABELS[cp]} | {m['complete_family_count']}/{s['n_families']} | {_pct(m['complete_family_proportion'])} | {m['reversal_profile_counts_0_to_4']} | {m['reversal_both_correct']}/{m['reversal_pairs']} | {_pct(m['mean_within_family_reversal_success'])} | {_fmt(m['median_family_mean_margin'])} | {_fmt(m['median_family_minimum_margin'])} |")
        return table + [""]
    for task, frame in STRATA[:2]:
        lines += table_for(task, frame)
    lines += ["## Separate diagnostic strata", "",
              "Naming is diagnostic and does not gate later strata. QA is scored separately from cloze. Alternate frames measure robustness to those specified query frames; their vocabulary is not claimed unseen during TinyStories training. No lexical holdout or distractor condition was added.", ""]
    for task, frame in STRATA[2:]:
        lines += table_for(task, frame)
    lines += ["## Completion-boundary, item, and mass diagnostics", "",
              "The word-only diagnostic excludes the period and retains both word tokens. Headline scores remain the full completed-answer scores. Any disagreement is explicitly retained as completion-boundary sensitivity; it is not used to choose a preferred scoring method after seeing results.", "",
              "| Stratum | Checkpoint | Full item correct | Full ties | Word-only item correct | Word ties | Full-only correct / word-only correct | Complete families full / word | Median pair mass | Median pooled candidate LL |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for task, frame in STRATA:
        key = f"{task}/{frame}"
        for cp in CHECKPOINTS:
            s = summary["checkpoints"][cp][key]
            f, w, d = s["full"], s["word_only"], s["word_vs_full"]
            lines.append(f"| {key} | {LABELS[cp]} | {f['item_correct']}/{s['n_items']} | {f['ties']} | {w['item_correct']}/{s['n_items']} | {w['ties']} | {d['full_only_correct']} / {d['word_only_correct']} | {f['complete_family_count']} / {w['complete_family_count']} | {_fmt(s['candidate_pair_probability_mass']['median'])} | {_fmt(s['candidate_log_likelihoods']['median'])} |")
    lines += ["", "Candidate-pair mass is the sum of probabilities of the two complete sequences. A preference within this pair does not establish that either answer would be freely generated. Exact candidate and per-token log-likelihoods are preserved in [RAW_SCORES.jsonl](RAW_SCORES.jsonl); distributions and word-only reversal/margin profiles are in [SUMMARY.json](SUMMARY.json).", "",
              "### Word-only family and margin profiles", ""]
    for task, frame in STRATA:
        lines += table_for(task, frame, "word_only")
    lines += ["## Query-only priors and matched contextual modulation", "",
              "Candidate 0/1 refers to each family's fixed candidate ordering, not a universal word. Query-only priors change prefix length and position as well as removing facts. They are diagnostic; no prior was subtracted from a contextual margin.", "",
              "| Stratum | Checkpoint | Prior preferences c0 / c1 / tie | Median prior d | Median absolute prior d | Context/prior preference agreement | Median answer-oriented reversal change | Positive changes |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for task, frame in STRATA:
        key = f"{task}/{frame}"
        for cp in CHECKPOINTS:
            s = summary["checkpoints"][cp][key]
            p, f = s["query_only_priors"]["full"], s["full"]
            a = p["context_prior_preference_agreement"]
            lines.append(f"| {key} | {LABELS[cp]} | {p['candidate0_preferred']} / {p['candidate1_preferred']} / {p['ties']} | {_fmt(p['candidate0_minus_candidate1']['median'])} | {_fmt(p['absolute_candidate_difference']['median'])} | {a['agree']}/{a['n_comparable']} | {_fmt(f['answer_oriented_reversal_change']['median'])} | {f['positive_answer_oriented_changes']}/{f['reversal_pairs']} |")
    lines += ["", "For a matched pair, the raw change is d(a=0)−d(a=1). The answer-oriented change multiplies it by +1 when candidate 0 is correct at a=0 and by −1 otherwise; equivalently it is margin(a=0)+margin(a=1). A positive change supports contextual modulation, and does not by itself establish successful reversal or selection. Both raw and oriented changes, including word-only versions, are preserved per pair in SUMMARY.json.", "",
              "## Order, mention, and query diagnostics", "",
              "Fact order 0 means the E0 fact is presented first. Mention rank 0 means the correct candidate appears earlier among the two factual candidate mentions. Queried fact rank 0 means the queried association is presented first. Query index identifies the queried identity in the frozen pair.", "",
              "| Stratum | Checkpoint | Fact-order correctness 0 / 1 | Correct-candidate mention correctness early / late | Queried-fact correctness first / second | Query correctness 0 / 1 | Families complete both orders / only 0 / only 1 / neither |",
              "|---|---|---|---|---|---|---|"]
    for task, frame in STRATA:
        key = f"{task}/{frame}"
        for cp in CHECKPOINTS:
            m = summary["checkpoints"][cp][key]["full"]
            order = m["order_and_query_diagnostics"]
            cells = []
            for field in ("fact_order", "correct_candidate_mention_rank", "queried_fact_rank", "query_index"):
                cells.append(" / ".join(f"{order[field][str(v)]['correct']}/{order[field][str(v)]['n']}" for v in (0, 1)))
            p = m["fact_order_complete_profiles"]
            profile = " / ".join(str(p[k]) for k in ("both", "order0_only", "order1_only", "neither"))
            lines.append(f"| {key} | {LABELS[cp]} | " + " | ".join(cells) + f" | {profile} |")
    lines += ["", "These order subsets are diagnostic, with correlated members. Their full and word-only margins are retained in SUMMARY.json; no separate significance tests or new gates are introduced.", "",
              "## Paired checkpoint comparisons", "",
              "All differences are second checkpoint minus first on identical families. 'Both / first only / second only / neither' classifies complete-family success. The comparison is behavioral: training trajectories, degree of TinyStories adaptation, and binding material differ.", "",
              "| Stratum | First → second | CF proportion difference | Both / first only / second only / neither | Median paired family-mean margin difference |",
              "|---|---|---:|---|---:|"]
    for pair in summary["paired_checkpoint_comparisons"]:
        p = pair["full"]
        cells = " / ".join(str(p[k]) for k in ("both_complete", "first_only_complete", "second_only_complete", "neither_complete"))
        lines.append(f"| {pair['task']}/{pair['frame']} | {LABELS[pair['first']]} → {LABELS[pair['second']]} | {_pct(p['complete_family_proportion_difference'])} | {cells} | {_fmt(p['median_paired_family_mean_margin_difference'])} |")
    lines += ["", "Per-family reversal-rate differences and paired margin differences, including word-only versions, are in SUMMARY.json. There is no pooled cloze/QA score, primary possession/location aggregate, statistical general-winner label, causal protection claim, or automatic training-parent selection.", "",
              "## Synthetic-binding development references", "",
              "The two previously used nonsacred DEV panels are evaluated separately and identically across checkpoints. They are development references with prior research exposure, not an untouched confirmation exam. English and binding use distinct inference paths and are not combined into an endpoint.", "",
              "| Checkpoint | Reference | Answer exact | BOTH_DISTINCT | Collapse | Complete quartets | Strict reversal both-correct | Queried row given BD | Answer given BD |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cp in CHECKPOINTS:
        for ref, r in sorted(binding[cp].items()):
            o = r["overall"]
            q, a = r["queried_row_given_BD"], r["answer_given_BD"]
            n = o["n"]
            lines.append(f"| {LABELS[cp]} | {ref} | {o['answer_exact']}/{n} | {o['BD']}/{n} | {o['collapse']} | {r['complete_quartets']}/{n // 4} | {r['reversal_both_correct']}/{r['reversal_pairs']} | {q['correct']}/{q['n']} | {a['correct']}/{a['n']} |")
    lines += ["", "Exact binding rows and provenance are retained in BINDING_REFERENCE_RESULTS.json and the execution manifest. The redundant measure that the selected row belongs to either true source is not counted as independent evidence.", "",
              "## Secondary finite-universe confidence intervals", "",
              "These 95% exact hypergeometric intervals concern only the explicitly enumerated eligible lexical-family universe for the specified fixed frame: naming N=18, possession N=36, location N=24. They are secondary to the raw behavioral counts, reversal profiles, and margins. They are not uncertainty over Baby's general English capability, all vocabulary, all constructions, or training-seed variation. Each complete family is the unit; its eight items are not independent trials.", "",
              "| Stratum | Checkpoint | Sample CF successes | Universe N | Secondary finite-universe interval |",
              "|---|---|---:|---:|---:|"]
    for task, frame in STRATA:
        key = f"{task}/{frame}"
        for cp in CHECKPOINTS:
            ci = summary["checkpoints"][cp][key]["full"]["secondary_finite_universe_interval"]
            lines.append(f"| {key} | {LABELS[cp]} | {ci['x']}/{ci['n']} | {ci['N']} | [{_pct(ci['lower'])}, {_pct(ci['upper'])}] |")
    lines += ["", "Intervals invert both hypergeometric tails at 0.025 using exact integer/rational probabilities. No p-value grid, independent-item binomial interval, or 1/256 family-chance baseline is used.", "",
              "## Shortcut references", "",
              "| Stratum | Rule | Item correct | Complete families | Reversal both-correct |",
              "|---|---|---:|---:|---:|"]
    for key, rules in summary["shortcut_references"].items():
        for rule, r in rules.items():
            lines.append(f"| {key} | {rule} | {r['item_correct']}/{r['n_items']} | {r['complete_family_count']}/{r['n_families']} | {r['reversal_both_correct']}/{r['reversal_pairs']} |")
    lines += ["", "First/last mention can produce successful reversal pairs in some constructions while failing every complete family. The battery does not exclude every shallow rule: success may reflect a narrow learned rule for these constructions.", "",
              "## Interpretation and next-decision boundaries", ""]
    for task in ("possession", "location"):
        key = f"{task}/core"
        values = [summary["checkpoints"][cp][key] for cp in CHECKPOINTS]
        counts = "; ".join(f"{LABELS[cp]} {s['full']['complete_family_count']}/{s['n_families']} complete families and {s['full']['reversal_both_correct']}/{s['full']['reversal_pairs']} reversal pairs" for cp, s in zip(CHECKPOINTS, values))
        lines += [f"For {task} primary cloze: {counts}. These are the measured checkpoint behaviors under the frozen construction; no new pass threshold is inferred.", ""]
    lines += ["Use the following prospective interpretations when weighing these numerical profiles:", "",
              "| Pattern | Supports | Does not support | Most useful next question |",
              "|---|---|---|---|",
              "| Graduate weak; both pilots stronger | Increased tested context sensitivity accompanies the language-trained checkpoint histories | English updates alone caused it; broad English competence | Robustness across reserved frames and additional combinations |",
              "| Pilot 0 stronger than Pilot 1 | Better Pilot 0 behavior on these matched tasks | Freezing caused weaker English | Learning progress versus protection under matched training conditions |",
              "| Pilot 1 stronger than Pilot 0 | Better contextual discrimination can coexist with worse TinyStories perplexity | Causal protection improvement or synthetic-binding transfer | Replication with matched rehearsal and controlled trajectories |",
              "| Cloze stronger than QA | Context-sensitive continuation with additional question-format difficulty | Invalid cloze evidence or conversational competence | Which QA-format requirement limits accessible answers? |",
              "| Perplexity improves but reversal remains weak | Distribution prediction improves without reliable tested counterfactual selection | Pure memorization or no semantic learning of any kind | Do priors, wording, or association selection explain errors? |",
              "| All three weak | Failure on the frozen battery | Architectural impossibility, insufficient capacity, or inability to acquire binding | Frame accessibility and modulation without successful selection |",
              "| Graduate unexpectedly strong | Tested ordinary-path behavior predates the language pilots | Pilot-acquired capability or broad language competence | Verify path isolation/scoring, alternate frames, later independent replication |",
              "| Naming weak, relational cloze stronger | Relational success despite a weaker naming diagnostic | Contradiction or reason to discard cloze | Naming wording and corpus familiarity |",
              "| Naming strong, relational cloze weak | A simpler identity dependency succeeds while these relations do not | Universal binding or a known architectural barrier | Relation wording, query direction, and role selection |",
              "", "'Stronger' must be read through the displayed profiles, not an undisclosed cutoff. No mechanism was probed here. Missing attention patterns or the earlier T10/T11 failures cannot establish mathematical impossibility. No result changes synthetic graduation criteria, warrants replacing the model, or automatically authorizes a training curriculum.", "",
              "The report deliberately retains disagreements across endpoints and frames. The next research decision should use the observed contextual modulation, successful selection, order sensitivity, and QA/frame differences to select a discriminating follow-up. No additional training, mechanistic experiment, or revised test design is executed by this aggregation.", "",
              "## Reproducibility and artifacts", "",
              "- [RAW_SCORES.jsonl](RAW_SCORES.jsonl): every frozen item/prior, checkpoint hash, and exact per-token/per-candidate score.",
              "- [FAMILY_RESULTS.csv](FAMILY_RESULTS.csv): complete-family, reversal, item, and margin outcomes in both scoring modes.",
              "- [SUMMARY.json](SUMMARY.json): all distributions, order/priors, signed reversal changes, paired comparisons, finite-universe intervals, and validation counts.",
              "- [BINDING_REFERENCE_RESULTS.json](BINDING_REFERENCE_RESULTS.json): separately evaluated nonsacred synthetic panels.",
              "", "Aggregation validated every input record against the sealed materialized exam, answer index, candidate tokens, token-score sums, correctness/tie flags, matched family structure, and required counts before emitting outputs. It loads no model and performs no inference.", ""]
    return "\n".join(lines)


def aggregate(execution_dir, frozen_dir=DEFAULT_FROZEN):
    """Validate complete existing scores, then write the three aggregate artifacts."""
    execution_dir, frozen_dir = Path(execution_dir), Path(frozen_dir)
    raw_path = execution_dir / "RAW_SCORES.jsonl"
    binding_path = execution_dir / "BINDING_REFERENCE_RESULTS.json"
    items, priors = _load_jsonl(frozen_dir / "ITEMS.jsonl"), _load_jsonl(frozen_dir / "PRIORS.jsonl")
    item_by_id = {item["item_id"]: item for item in items}
    prior_by_id = {prior["prior_id"]: prior for prior in priors}
    if len(items) != 832 or len(priors) != 208 or len(item_by_id) != 832 or len(prior_by_id) != 208:
        raise ValueError("Frozen battery counts differ from protocol")
    raw = _load_jsonl(raw_path)
    if len(raw) != 3 * 1040:
        raise ValueError(f"Raw scores incomplete: expected 3120 records, got {len(raw)}")
    seen, runtime_strings = set(), set()
    context_rows, prior_rows = [], []
    for row in raw:
        cp, kind, record, score = row["checkpoint"], row["kind"], row["record"], row["score"]
        if cp not in CHECKPOINTS or row["checkpoint_sha256"] != HASHES[cp]:
            raise ValueError("Unexpected checkpoint registry entry")
        if kind not in {"contextual", "prior"}:
            raise ValueError("Unexpected raw record kind")
        rid = record["item_id" if kind == "contextual" else "prior_id"]
        expected = (item_by_id if kind == "contextual" else prior_by_id).get(rid)
        if expected != record:
            raise ValueError(f"Raw record differs from sealed item: {rid}")
        identity = cp, kind, rid
        if identity in seen:
            raise ValueError(f"Duplicate score record: {identity}")
        seen.add(identity)
        _validate_score(record, score, kind == "contextual")
        runtime_strings.add(json.dumps(score["runtime"], sort_keys=True))
        (context_rows if kind == "contextual" else prior_rows).append(row)
    expected_identities = {(cp, "contextual", rid) for cp in CHECKPOINTS for rid in item_by_id}
    expected_identities |= {(cp, "prior", rid) for cp in CHECKPOINTS for rid in prior_by_id}
    if seen != expected_identities or len(runtime_strings) != 1:
        raise ValueError("Incomplete coverage or runtime differs across scores")
    runtime = json.loads(next(iter(runtime_strings)))
    for key, expected in {"model_dtype": "float32", "log_softmax_and_sum_dtype": "float64",
                          "autocast": False, "cuda_matmul_allow_tf32": False,
                          "cudnn_allow_tf32": False, "sequence_batch_size": 1,
                          "padding": False, "truncation": False,
                          "english_forward_path": "model.base_model(input_ids)"}.items():
        if runtime.get(key) != expected:
            raise ValueError(f"Runtime does not match frozen protocol: {key}")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if set(binding) != set(CHECKPOINTS):
        raise ValueError("Binding reference checkpoint coverage differs")
    references = None
    for cp in CHECKPOINTS:
        if len(binding[cp]) != 2:
            raise ValueError("Both binding panels required for every checkpoint")
        if references is None:
            references = set(binding[cp])
        elif set(binding[cp]) != references:
            raise ValueError("Binding reference names differ across checkpoints")
        for ref in binding[cp].values():
            if ref["overall"]["n"] != 80 or ref["reversal_pairs"] != 40:
                raise ValueError("Binding panel counts differ from protocol")
    families = _family_results(context_rows)
    summary = {
        "schema_version": 1,
        "headline_evidence": "Raw complete-family counts/proportions, reversal profiles, and likelihood-margin summaries.",
        "confidence_interval_status": "Secondary; only explicitly enumerated eligible lexical-family universes, never general English capability.",
        "pass_fail_gates": None, "prior_calibration": False,
        "inputs": {"raw_scores_sha256": _sha(raw_path), "binding_results_sha256": _sha(binding_path),
                   "sealed_items_sha256": _sha(frozen_dir / "ITEMS.jsonl"),
                   "sealed_priors_sha256": _sha(frozen_dir / "PRIORS.jsonl"),
                   "sealed_protocol_sha256": _sha(frozen_dir / "PROTOCOL.md")},
        "validation": {"records": len(raw), "contextual_records": len(context_rows), "prior_records": len(prior_rows),
                       "identical_sealed_records": True, "identical_runtime": True,
                       "score_arithmetic_validated": True, "runtime": runtime},
        "checkpoints": {}, "families": families,
    }
    for cp in CHECKPOINTS:
        summary["checkpoints"][cp] = {
            f"{task}/{frame}": _stratum_summary(cp, task, frame, families, context_rows, prior_rows)
            for task, frame in STRATA
        }
    summary["paired_checkpoint_comparisons"] = _paired(families)
    summary["shortcut_references"] = _shortcuts(items)
    contents = {
        "FAMILY_RESULTS.csv": _csv(families),
        "SUMMARY.json": json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        "REPORT.md": _report(summary, binding),
    }
    for name, content in contents.items():
        _write(execution_dir / name, content)
    return {"outputs": {name: {"sha256": _sha(execution_dir / name), "bytes": (execution_dir / name).stat().st_size}
                        for name in contents}, "validation": summary["validation"]}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("execution_dir", type=Path)
    parser.add_argument("--frozen-dir", type=Path, default=DEFAULT_FROZEN)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.execution_dir, args.frozen_dir), indent=2))
