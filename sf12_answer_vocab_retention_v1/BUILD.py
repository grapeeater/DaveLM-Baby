"""Build the additive SF12 answer-vocabulary retention study from sealed SF11 payloads."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SF11 = ROOT / "sf11_narrow_breadth_widening_v1"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def copy_payload() -> None:
    files = [
        "SF2_ENGINE.py", "SF2_PROTOCOL.json", "KL_POOL.json", "KL_POOL_MANIFEST.json",
        "D3_SELECTION.json", "EXTERNAL_INPUTS.json", "TRAIN16_RETENTION.json",
        "TRAIN.json", "SCHEDULE.json", "DEV_SURFACE.json", "DEV_ORDER.json",
        "CURRICULUM_MANIFEST.json",
    ]
    for name in files:
        shutil.copyfile(SF11 / name, HERE / name)
    for directory in ["data", "sources", "authority"]:
        shutil.copytree(SF11 / directory, HERE / directory, dirs_exist_ok=False)


def build_controller() -> None:
    src = (SF11 / "CONTROLLER.py").read_text(encoding="utf-8")
    src = src.replace(
        '"""SF11: narrower curriculum; identical SF10 loss, optimizer, scope and persistence."""',
        '"""SF12: SF11 recipe plus teacher-anchored answer-vocabulary excess retention."""',
    )
    src = src.replace(
        "LAMBDA_MARGIN = 0.25\n",
        "LAMBDA_MARGIN = 0.25\nLAMBDA_NAME = 1.0\nNAME_DELTA = 0.001\nNAME_TOKEN_IDS = (314, 536, 925, 512)  # Alex, Owen, Mia, Nora\n",
        1,
    )
    marker = "\ndef verify(sealed=True):\n"
    helper = r'''

def compute_kl_and_name_retention(m, teacher, device, pool_rows, sel):
    """Preserve SF2 forward KL exactly and derive R_name from those same distributions."""
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
    ls = m.base_model(x)
    with torch.no_grad():
        lt = teacher(x)
    rows_idx = torch.tensor([ref[ri] for (ri, _) in sel], device=device, dtype=torch.long)
    cols = torch.tensor([c for (_, c) in sel], device=device, dtype=torch.long)
    ls_sel = ls[rows_idx, cols].double()
    lt_sel = lt[rows_idx, cols].double()
    ps = ls_sel.softmax(-1)
    pt = lt_sel.softmax(-1)
    kl_pos = (ps * (ls_sel.log_softmax(-1) - lt_sel.log_softmax(-1))).sum(-1)
    student_mass = ps[:, list(NAME_TOKEN_IDS)].sum(-1)
    teacher_mass = pt[:, list(NAME_TOKEN_IDS)].sum(-1)
    excess = student_mass - teacher_mass - NAME_DELTA
    active = excess > 0
    relu_excess = torch.relu(excess)
    r_name = relu_excess.mean().float()
    telemetry = {
        "mean_S_student": float(student_mass.detach().mean().cpu()),
        "mean_S_teacher": float(teacher_mass.detach().mean().cpu()),
        "mean_excess_before_relu": float(excess.detach().mean().cpu()),
        "active_fraction": float(active.detach().double().mean().cpu()),
        "mean_active_excess": float(excess.detach()[active].mean().cpu()) if bool(active.any()) else 0.0,
        "R_name": float(r_name.detach().cpu()),
        "R_name_loss_contribution": float((LAMBDA_NAME * r_name).detach().cpu()),
        "positions": len(sel),
    }
    return kl_pos.mean().float(), r_name, telemetry


def save_retention_probe(m, teacher, device, kl, out, update):
    """Fixed descriptive probe: first frozen 160-position KL selection at every endpoint."""
    path = out / f"update{update}_NAME_RETENTION_TELEMETRY.json"
    if path.exists():
        return read(path)
    was_training = m.training
    m.eval()
    with torch.inference_mode():
        kl_value, r_name, telemetry = compute_kl_and_name_retention(
            m, teacher, device, kl["rows"], E.pick_kl_entries(kl["entries"], 1)
        )
    telemetry.update({"update": update, "probe_kl": float(kl_value.cpu()),
                      "selection": "frozen KL English-index 1 (160 positions)",
                      "gate": "DESCRIPTIVE_ONLY"})
    rt.atomic_json(telemetry, path)
    if was_training:
        m.train()
    return telemetry
'''
    assert marker in src
    src = src.replace(marker, helper + marker, 1)
    src = src.replace("SF11_PROSPECTIVE_PREFLIGHT_PASS", "SF12_PROSPECTIVE_PREFLIGHT_PASS")
    src = src.replace("SF11_PRE_PARENT_LOAD_PASS", "SF12_PRE_PARENT_LOAD_PASS")
    src = src.replace(
        "'kl_pool': sha(H / 'KL_POOL.json'), 'margin_M': MARGIN_M, 'lambda_margin': LAMBDA_MARGIN}",
        "'kl_pool': sha(H / 'KL_POOL.json'), 'margin_M': MARGIN_M, 'lambda_margin': LAMBDA_MARGIN,\n"
        "                  'lambda_name': LAMBDA_NAME, 'name_delta': NAME_DELTA, 'name_token_ids': list(NAME_TOKEN_IDS)}",
        1,
    )
    src = src.replace(
        "rt.atomic_json({'status': 'PASS', 'parent_sha256': run['parent_checkpoint_sha256'],",
        "save_retention_probe(m, teacher, device, kl, out, 0)\n        rt.atomic_json({'status': 'PASS', 'parent_sha256': run['parent_checkpoint_sha256'],",
        1,
    )
    src = src.replace(
        "g = gates(ret, dsurf, dord, ch, d3, base_loss, u)\n",
        "name_probe = save_retention_probe(m, teacher, device, kl, out, u)\n        g = gates(ret, dsurf, dord, ch, d3, base_loss, u)\n",
        1,
    )
    src = src.replace(
        "'dev_order': {'exact': dord['exact']}, 'gates': g, 'language': ch['language'], 'd3': d3}), flush=True)",
        "'dev_order': {'exact': dord['exact']}, 'gates': g, 'language': ch['language'], 'd3': d3, 'R_name': name_probe}), flush=True)",
        1,
    )
    src = src.replace("ce = kl_val = margin_loss = None", "ce = kl_val = margin_loss = name_val = name_telemetry = None", 1)
    src = src.replace(
        "kl_val = E.compute_kl(m, teacher, device, kl['rows'], E.pick_kl_entries(kl['entries'], english_index))",
        "kl_val, name_val, name_telemetry = compute_kl_and_name_retention(\n"
        "                m, teacher, device, kl['rows'], E.pick_kl_entries(kl['entries'], english_index))",
        1,
    )
    src = src.replace(
        "loss = loss + LAMBDA_MARGIN * margin_loss\n",
        "loss = loss + LAMBDA_MARGIN * margin_loss\n            loss = loss + LAMBDA_NAME * name_val\n",
        1,
    )
    src = src.replace(
        "if margin_loss is not None: rec.update(margin_loss=float(margin_loss.detach()))",
        "if margin_loss is not None: rec.update(margin_loss=float(margin_loss.detach()), "
        "name_retention=name_telemetry)",
        1,
    )
    assert "E.compute_kl(m, teacher" not in src
    assert src.count("compute_kl_and_name_retention(") == 3
    (HERE / "CONTROLLER.py").write_text(src, encoding="utf-8", newline="\n")


def build_protocol() -> None:
    old = json.loads((SF11 / "PROTOCOL.json").read_text(encoding="utf-8"))
    old.update({
        "study": "SF12_ANSWER_VOCAB_RETENTION_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": [87029, 87030, 87031],
        "runs": [
            {"seed": 87029, "arm": "curriculum", "parent_checkpoint": str(ROOT / "sf8_margin_dose_comparison_v1/runs/seed_87017_low/checkpoint_200.pt"), "parent_checkpoint_sha256": "9e293af6d16cb642ba8e2bf1aeb5ad392ff4b27399ebbf0b7b8fd2ef7339b81d"},
            {"seed": 87030, "arm": "curriculum", "parent_checkpoint": str(ROOT / "sf8_margin_dose_comparison_v1/runs/seed_87018_low/checkpoint_200.pt"), "parent_checkpoint_sha256": "839f7f5a60b33a1736376c8a68a02985fb05ba3230e56b7aaf20d1eba5b35102"},
            {"seed": 87031, "arm": "curriculum", "parent_checkpoint": str(ROOT / "sf8_margin_dose_comparison_v1/runs/seed_87019_low/checkpoint_200.pt"), "parent_checkpoint_sha256": "eb6a725173ff26516d876914ea4357cce7e3758616c3d659ff059d2354404517"},
        ],
        "hypothesis": "Dedicated teacher-anchored retention of the four trained answer-token classes is sufficient, under the frozen SF11 curriculum and recipe, to contain the previously replicated D3 regression while retaining widening behavior.",
        "manipulated_variable": "Add lambda_name * mean ReLU(S_student-S_teacher-delta) at the same frozen 160 KL positions, where N={314,536,925,512}, delta=.001 and lambda_name=1.0. No second forward and no change to full-KL semantics or sampling.",
        "english_objective": "Frozen SF11 mean causal answer4tokens+EOS CE +1.0 D_KL(P_student||P_Pilot1) over the same160frozenpositions +.25 mean max(0,1-first-answer gap) +1.0 mean max(0,sum_N P_student-sum_N P_teacher-.001).",
        "answer_vocabulary_retention": {
            "name_token_ids": {"Alex": 314, "Owen": 536, "Mia": 925, "Nora": 512},
            "delta": 0.001, "lambda_name": 1.0, "positions": 160,
            "sampling": "identical frozen SF11 KL selection on every English update",
            "teacher": "frozen Pilot1 parent used by existing KL",
            "implementation": "reuse the exact student/teacher distributions already computed for full KL; no additional forward",
            "coverage_limit": "Does not change or test corpus coverage/disjointness; it addresses only aggregate answer-vocabulary under-attention/dilution on existing KL positions.",
        },
        "classification": {
            "priority1": "INCONCLUSIVE_EARLY_STOP if any run lacks a scientific classification due to mechanical/integrity failure",
            "priority2": "RETENTION_REGRESSION if any run fails TRAIN16, language, or either binding pool at a required checkpoint",
            "priority3": "ANSWER_VOCAB_RETENTION_SUPPORTED_FULL_ENDPOINT if >=2/3 runs reach u200 and pass every frozen endpoint gate",
            "priority4": "ANSWER_VOCAB_RETENTION_WEAKENED if D3 exceeds .01 in >=2/3 runs",
            "priority5": "ANSWER_VOCAB_RETENTION_SUPPORTED_PARTIAL_WIDENING if >=2/3 runs reach u200 with D3/TRAIN16/language/binding green, SF11 development gain, DEV_SURFACE exact>=12 and DEV_ORDER exact>=8",
            "priority6": "WIDENING_STALLED_UNDER_RETENTION if >=2/3 runs reach u200 with D3/TRAIN16/language/binding green but fewer than2 meet partial-widening criteria",
            "otherwise": "MIXED_OR_UNRESOLVED",
            "partial_widening_rationale": "12 surface exact and 8 order exact each strictly exceed the best frozen SF11 u100 values (11 and7), while the inherited +4/+4 own-baseline development-gain requirement prevents a favorable starting point from qualifying alone.",
        },
        "telemetry": "At u0/u100/u200 evaluate R_name on the fixed first160 KL positions. During training record actual scheduled mean S_student, S_teacher, raw excess, active fraction, active-only excess, R_name and weighted contribution. Descriptive only.",
        "next_action_rule": "Freeze SF12 classification, then recommend exactly one review/action appropriate to it. Do not execute it.",
    })
    write_json(HERE / "PROTOCOL.json", old)


def build_launcher() -> None:
    text = (SF11 / "LAUNCH.py").read_text(encoding="utf-8")
    (HERE / "LAUNCH.py").write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    existing = [p.name for p in HERE.iterdir() if p.name != "BUILD.py"]
    if existing:
        raise RuntimeError(f"Refuse nonempty build directory: {existing}")
    copy_payload()
    build_controller()
    build_protocol()
    build_launcher()
    write_json(HERE / "CLASSIFICATION.json", json.loads((HERE / "PROTOCOL.json").read_text())["classification"])
    write_json(HERE / "PROVENANCE.json", {
        "study": "SF12_ANSWER_VOCAB_RETENTION_V1",
        "predecessor": str(SF11),
        "predecessor_receipt_sha256": sha(SF11 / "FREEZE_RECEIPT.json"),
        "unchanged_payload_hashes": {name: sha(HERE / name) for name in ["TRAIN.json", "SCHEDULE.json", "DEV_SURFACE.json", "DEV_ORDER.json", "TRAIN16_RETENTION.json", "KL_POOL.json", "D3_SELECTION.json", "SF2_ENGINE.py"]},
        "sole_scientific_change": "R_name term defined in PROTOCOL.json and CONTROLLER.py",
        "historical_panels": "LOCKED_UNSCORED",
        "final_sacred": "NOT_ACCESSED",
    })
    print("SF12_BUILD_COMPLETE")


if __name__ == "__main__":
    main()
