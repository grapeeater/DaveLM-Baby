from __future__ import annotations

"""Data-only autopsy of the frozen v2R4 terminal metrics.

This tool joins committed panel items with committed U16000 rows. It does not
load weights, does not train, and does not rewrite historical artifacts.
Diagnostic value-span / separator splits are extra measurements; they are not
a change to Gate L/C/R.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .emission_source import emission_source_report
from .mechanism_census import mechanism_census, mechanism_headlines
from .v2r4_provenance import (
    EOS_TOKEN,
    FROZEN_PANELS_SHA256,
    TERMINAL_METRICS_SHA256,
    TERMINAL_UPDATE,
    UNIQUE_SCORED_PANELS,
    require_frozen_file,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS = ROOT / "runs" / "structured_v2r4_seed106001_from6000_terminal" / "metrics.jsonl"
DEFAULT_PANELS = ROOT / "data" / "generated" / "foundation_v2" / "panels.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_metrics(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise RuntimeError(f"no metrics records in {path}")
    return records


def terminal_record(records: list[dict]) -> dict:
    matched = [row for row in records if int(row.get("update", -1)) == TERMINAL_UPDATE]
    if not matched:
        raise RuntimeError(f"no U{TERMINAL_UPDATE} record in metrics")
    return matched[-1]


def join_panel(panels: dict, term: dict, name: str) -> list[tuple[dict, dict]]:
    items = panels[name]
    rows = term["rows"][name]
    if len(items) != len(rows):
        raise RuntimeError(f"{name}: panel n={len(items)} row n={len(rows)}")
    return list(zip(items, rows))


def classify_item_row(item: dict, row: dict) -> dict:
    span = list(item["target_span"])
    target = list(item["target"])
    emitted = [int(tok) for tok in row.get("emitted", [])]
    ranks = [int(x) for x in row.get("all_ranks", [])]
    sep = target[-2] if len(target) >= 2 else None
    value_ok = emitted[: len(span)] == span
    tf_value = bool(ranks) and len(ranks) >= len(span) and all(rank == 1 for rank in ranks[: len(span)])
    first_ok = int(row["target_rank"]) == 1
    if emitted == target:
        fail = "exact"
    elif value_ok:
        if len(emitted) <= len(span):
            fail = "value_ok_truncated"
        elif sep is not None and emitted[len(span)] != sep and emitted[len(span)] != EOS_TOKEN:
            fail = "value_ok_bad_sep"
        elif sep is not None and emitted[len(span)] == sep and (
            len(emitted) <= len(span) + 1 or emitted[len(span) + 1] != EOS_TOKEN
        ):
            fail = "value_ok_bad_eos"
        else:
            fail = "value_ok_other_suffix"
    elif first_ok:
        fail = "first_ok_midspan_fail"
    else:
        fail = "first_fail"
    emitted_sep = emitted[len(span)] if value_ok and len(emitted) > len(span) else None
    return {
        "fail": fail,
        "value_ok": value_ok,
        "tf_value": tf_value,
        "first_ok": first_ok,
        "free_exact": bool(row.get("free_exact")),
        "tf_exact": bool(row.get("tf_exact")),
        "sep": sep,
        "emitted_sep": emitted_sep,
        "first_error": int(row.get("first_error", -1)),
        "span_len": len(span),
        "pair_count": item.get("pair_count"),
        "value_length": item.get("value_length"),
        "variant": item.get("variant"),
        "query_index": item.get("query_index"),
        "surface": item.get("surface"),
        "kind": item.get("kind"),
        "low_prior": bool(item.get("low_prior", False)),
        "broken": item.get("broken"),
        "target_rank": int(row["target_rank"]),
        "first_margin": float(row.get("first_margin", 0.0)),
        "immediate_eos": bool(emitted) and emitted[0] == EOS_TOKEN,
        "all_same_token": bool(emitted) and len(set(emitted)) == 1,
        "emitted_after_value": emitted[len(span) : len(span) + 3] if len(emitted) > len(span) else emitted[len(span) :],
    }


def _rate(rows: list[dict], key: str) -> float | None:
    if not rows:
        return None
    return sum(bool(row[key]) for row in rows) / len(rows)


def _slice_table(rows: list[dict], field: str) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field))].append(row)
    out: dict[str, dict] = {}
    for key, group in sorted(grouped.items(), key=lambda kv: (kv[0] == "None", kv[0])):
        n = len(group)
        pair_count = group[0].get("pair_count") if field == "pair_count" else None
        chance = (1.0 / int(pair_count)) if isinstance(pair_count, int) and pair_count > 0 else None
        out[key] = {
            "n": n,
            "first_ok": _rate(group, "first_ok"),
            "tf_value": _rate(group, "tf_value"),
            "value_ok": _rate(group, "value_ok"),
            "free_exact": _rate(group, "free_exact"),
            "chance_1_over_k": chance,
        }
    return out


def panel_census(pairs: list[tuple[dict, dict]]) -> dict:
    classified = [classify_item_row(item, row) for item, row in pairs]
    fail_counts = Counter(row["fail"] for row in classified)
    seps = Counter(row["sep"] for row in classified if row["sep"] is not None)
    emitted_seps = Counter(row["emitted_sep"] for row in classified if row["value_ok"])
    tf_not_free_value = sum(row["tf_value"] and not row["value_ok"] for row in classified)
    free_not_tf_value = sum(row["value_ok"] and not row["tf_value"] for row in classified)
    return {
        "n": len(classified),
        "first_ok": sum(row["first_ok"] for row in classified),
        "tf_value": sum(row["tf_value"] for row in classified),
        "value_ok": sum(row["value_ok"] for row in classified),
        "tf_exact": sum(row["tf_exact"] for row in classified),
        "free_exact": sum(row["free_exact"] for row in classified),
        "immediate_eos": sum(row["immediate_eos"] for row in classified),
        "all_same_token": sum(row["all_same_token"] for row in classified),
        "tf_value_not_free_value": tf_not_free_value,
        "free_value_not_tf_value": free_not_tf_value,
        "fail": dict(fail_counts),
        "target_separators": {str(key): value for key, value in seps.most_common()},
        "emitted_separators_given_value_ok": {str(key): value for key, value in emitted_seps.most_common()},
        "by_pair_count": _slice_table(classified, "pair_count"),
        "by_value_length": _slice_table(classified, "value_length"),
        "by_variant": _slice_table(classified, "variant"),
        "by_query_index": _slice_table(classified, "query_index"),
    }


def collapse_stats(term: dict, names: tuple[str, ...] | list[str]) -> dict:
    rows: list[dict] = []
    for name in names:
        rows.extend(term["rows"][name])
    immediate_eos = sum(1 for row in rows if row.get("emitted") and row["emitted"][0] == EOS_TOKEN)
    all_same = sum(1 for row in rows if row.get("emitted") and len(set(row["emitted"])) == 1)
    return {"n": len(rows), "immediate_eos": immediate_eos, "all_same_token": all_same}


def trajectory(records: list[dict]) -> list[dict]:
    out = []
    for record in records:
        summaries = record.get("summaries", {})
        def grab(name: str) -> dict:
            block = summaries.get(name) or {}
            return {
                "n": block.get("n"),
                "first_top1": block.get("first_top1"),
                "free_exact": block.get("free_exact"),
            }
        out.append({
            "update": record.get("update"),
            "stage": record.get("stage"),
            "language_dev_ce": record.get("language_dev_ce"),
            "probe_limit_inferred": summaries.get("same_surface_novel", {}).get("n"),
            "primitive_keyed": grab("primitive_keyed"),
            "primitive_induction": grab("primitive_induction"),
            "short_keyed": grab("short_keyed"),
            "same_surface_novel": grab("same_surface_novel"),
            "same_surface_induction": grab("same_surface_induction"),
            "heldout_surface": grab("heldout_surface"),
            "unseen_length": grab("unseen_length"),
            "low_prior": grab("low_prior"),
            "distractor": grab("distractor"),
            "broken_context": grab("broken_context"),
            "broken_order": grab("broken_order"),
        })
    return out


def heldout_rank1_suffix_examples(pairs: list[tuple[dict, dict]], limit: int = 12) -> list[dict]:
    examples = []
    for item, row in pairs:
        classified = classify_item_row(item, row)
        if not classified["first_ok"]:
            continue
        examples.append({
            "pair_count": item.get("pair_count"),
            "value_length": item.get("value_length"),
            "variant": item.get("variant"),
            "target": list(item["target"]),
            "emitted": list(row.get("emitted", [])),
            "fail": classified["fail"],
            "first_error": classified["first_error"],
            "all_ranks": list(row.get("all_ranks", [])),
        })
        if len(examples) >= limit:
            break
    return examples


def run_autopsy(metrics_path: Path, panels_path: Path) -> dict:
    panels_id = require_frozen_file(panels_path, FROZEN_PANELS_SHA256, "panels")
    metrics_id = require_frozen_file(metrics_path, TERMINAL_METRICS_SHA256, "metrics")
    panels = load_json(panels_path)
    records = load_metrics(metrics_path)
    term = terminal_record(records)
    censuses = {name: panel_census(join_panel(panels, term, name)) for name in UNIQUE_SCORED_PANELS}
    hold = censuses["heldout_surface"]
    broken = censuses["broken_context"]
    novel = censuses["same_surface_novel"]
    emission = emission_source_report(panels, term)
    mechanism = mechanism_census(panels, records, term, emission)
    return {
        "status": "V2R4_INDEPENDENT_AUTOPSY",
        "protocol": "BABY_V010_FOUNDATION_V2R4",
        "v2r5_status": "absent_in_github_unresolved_pending",
        "protected_material_opened": False,
        "gates_weakened": False,
        "historical_artifacts_rewritten": False,
        "panels_identity": panels_id,
        "metrics_identity": metrics_id,
        "panels_sha256": panels_id["frozen_sha256"],
        "metrics_sha256": metrics_id["frozen_sha256"],
        "terminal_update": term.get("update"),
        "language_dev_ce": term.get("language_dev_ce"),
        "language_baseline_ce": term.get("language_baseline_ce"),
        "alias_inflation": {
            "aliased_scored_rows": collapse_stats(term, list(term["rows"].keys())),
            "unique_scored_rows": collapse_stats(term, UNIQUE_SCORED_PANELS),
            "note": "Gate L in the terminal report used 992 rows including novel/induction aliases of same_surface_* panels.",
        },
        "generator_confounds": {
            "primitive_keyed_all_pair_count_one": censuses["primitive_keyed"]["by_pair_count"].keys() == {"1"},
            "heldout_nested_axes": ["heldout_surface", "unseen_length", "low_prior", "distractor", "broken_context", "broken_order"],
            "broken_context_replaces_key_not_value": True,
            "positive_changed_position_panel_present": False,
        },
        "value_vs_separator": {
            "same_surface_novel_value_ok": novel["value_ok"],
            "same_surface_novel_free_exact": novel["free_exact"],
            "heldout_value_ok": hold["value_ok"],
            "heldout_free_exact": hold["free_exact"],
            "broken_context_value_ok": broken["value_ok"],
            "heldout_value_rate": hold["value_ok"] / hold["n"],
            "broken_context_value_rate": broken["value_ok"] / broken["n"],
            "tf_value_equals_free_value_all_unique_panels": all(
                censuses[name]["tf_value_not_free_value"] == 0 and censuses[name]["free_value_not_tf_value"] == 0
                for name in UNIQUE_SCORED_PANELS
            ),
        },
        "panels": censuses,
        "heldout_first_ok_examples": heldout_rank1_suffix_examples(join_panel(panels, term, "heldout_surface")),
        "emission_source": emission,
        "mechanism": mechanism,
        "mechanism_headlines": mechanism_headlines(mechanism),
        "trajectory": trajectory(records),
        "hypothesis_read": {
            "A_internal_identification": "supported_tf_continuation_lock_on_train_gold_span",
            "A_prime_heldout_separator": "supported_value_ok_without_exact",
            "B_query_binding": "failed_queried_copy_not_above_1_over_k_including_2_pair",
            "C_payload_copy": "supported_train_novel_inventory_copy_93_of_96",
            "D_curriculum": "supported_copy_saturates_then_selection_does_not_lift",
            "D_free_emission_of_selected_span": "supported_tf_equals_free",
            "E_separator_eos": "heldout_exact_is_separator_ood_and_induction_eos_is_trailing_sep",
            "F_surface": "heldout_inventory_copy_collapses_at_long_values",
            "probe_limit_confound": "interim_evals_n_16_only_terminal_full_panel",
            "architecture": "not_justified_as_next_claim",
        },
    }


def render_markdown(report: dict) -> str:
    panels = report["panels"]
    lines = [
        "# Baby v0.10 v2R4 independent autopsy (generated)",
        "",
        "Status: diagnostic only. Frozen Gate L/C/R are unchanged.",
        f"v2R5: `{report['v2r5_status']}`.",
        "",
        f"- panels sha256: `{report['panels_sha256']}`",
        f"- metrics sha256: `{report['metrics_sha256']}`",
        f"- language DEV CE: `{report['language_dev_ce']}`",
        "",
        "## Alias-aware Gate L collapse",
        "",
        json.dumps(report["alias_inflation"], indent=2),
        "",
        "## Panel value-span census",
        "",
        "| panel | n | first | tf_value | value_ok | free_exact |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in UNIQUE_SCORED_PANELS:
        block = panels[name]
        lines.append(
            f"| {name} | {block['n']} | {block['first_ok']} | {block['tf_value']} | {block['value_ok']} | {block['free_exact']} |"
        )
    lines.extend([
        "",
        "## Emission source (queried vs competitor vs off-inventory)",
        "",
        json.dumps(report.get("emission_source", {}).get("headline"), indent=2),
        "",
        "## Mechanism headlines",
        "",
        json.dumps(report.get("mechanism_headlines"), indent=2),
        "",
        "",
    ])
    for name in ("primitive_keyed", "short_keyed", "same_surface_novel", "heldout_surface", "broken_context"):
        lines.append(f"### {name}")
        lines.append("")
        lines.append(json.dumps(panels[name]["by_pair_count"], indent=2))
        lines.append("")
    lines.extend([
        "## Hypothesis read",
        "",
        json.dumps(report["hypothesis_read"], indent=2),
        "",
        "This generated table is an audit companion. The narrative independent autopsy lives in `research/V010_V2R4_INDEPENDENT_AUTOPSY.md`.",
        "",
    ])
    return "\n".join(lines) + "\n"


def write_report(report: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "AUTOPSY.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "AUTOPSY.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "out": str(out), "v2r5_status": report["v2r5_status"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Data-only v2R4 terminal autopsy")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--panels", type=Path, default=DEFAULT_PANELS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.metrics.exists():
        raise SystemExit(f"metrics missing: {args.metrics}")
    if not args.panels.exists():
        raise SystemExit(f"panels missing: {args.panels}")
    report = run_autopsy(args.metrics, args.panels)
    write_report(report, args.out)


if __name__ == "__main__":
    main()
