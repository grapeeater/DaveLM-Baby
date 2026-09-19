from __future__ import annotations

"""Stack2 s4: genuine short English answers from the s3s composition survivor.

U16000 stays authoritative. TEST/FINAL/SACRED stay sealed. s3s is not promoted.
Who-bind / D3 / composition are retention, not the objective.
"""

import json
from pathlib import Path

from .data_language_bridge import (
    build_s3_panels,
    make_compose_sentence_item,
    make_who_sentence_item,
)
from .selection_language_bridge import (
    eval_panels,
    load_experimental_baby,
    run_sentence_decode,
    run_usable_chat,
    slim_panels,
)
from .selection_s1 import digest, write
from .selection_stack2 import (
    CAMPAIGN,
    DROP_BAR,
    OUT,
    S2A_COMBINE,
    campaign_update,
    cheap_d3_slice,
    english_native_holds,
    ledger_append,
    mix_scores,
    require_identities,
    run_d3_log,
    slim_usable,
    _drop_cuda,
    _exact,
    _top1,
)
from .selection_stack2_s3 import (
    LIGHT_SENTENCE,
    SCAFFOLD_GAP,
    SENT_BARE_GRAD,
    compose_scores,
    slim_sentence,
    usable_holds_s2a,
)

S3S_SURVIVOR = OUT / "s3s_compose_lock_324121" / "checkpoint_00025.pt"
S3S_SURVIVOR_SHA = "8101c421bedf0514bd4ad7c403557f4e72cc2cfc0626090d779eafb2343b382d"
S3S_D3_SLICE = 0.875
S3S_D3_FULL = 190
S3S_MIXED = 0.875
S3S_COMBINE = S2A_COMBINE
S3S_WHO2 = 0.50
S3S_WHO3 = 0.46875
S3S_USABLE = 0.9047619047619048
S3S_USABLE4 = 0.9375
WHO_KEEP = 0.35
WHO3_KEEP = 0.25
SENT_BARE_SIGNAL = 0.25
SENT_1E_SIGNAL = 0.25
D3_SLICE_KILL = 0.125
SENTENCE_UPDATES = 25
MAX_ARMS = 6


def mix_holds_s3s(native: dict) -> tuple[bool, str]:
    scores = mix_scores(native)
    mixed_ok = scores["mixed_2e"] + 1e-12 >= S3S_MIXED - DROP_BAR
    combine_ok = scores["fact_combine"] + 1e-12 >= S3S_COMBINE - DROP_BAR
    lesson = f"mixed={scores['mixed_2e']:.3f} fact_combine={scores['fact_combine']:.3f} story_combine={scores['story_combine']:.3f}"
    if mixed_ok and combine_ok:
        return True, f"s3s mix hold {lesson}"
    return False, f"s3s mix drop {lesson}"


def usable_holds_s3s(usable: dict | None) -> tuple[bool, str]:
    ok, lesson = usable_holds_s2a(usable)
    if usable is None:
        return False, "usable-chat missing"
    auto = usable.get("autoregressive") or {}
    auto4 = usable.get("autoregressive_4turn") or auto
    turn = float(auto.get("usable_turn") or 0.0)
    turn4 = float(auto4.get("usable_turn") or 0.0)
    stop = float(auto4.get("period_stop") or auto.get("period_stop") or 0.0)
    reuse = float(auto4.get("fact_reuse") or auto.get("fact_reuse") or 0.0)
    hold = (
        turn + 1e-12 >= S3S_USABLE - DROP_BAR
        and turn4 + 1e-12 >= S3S_USABLE4 - DROP_BAR
        and stop + 1e-12 >= 0.95
        and reuse + 1e-12 >= 0.95
    )
    note = f"usable4={turn4:.3f} usable={turn:.3f} stop={stop:.3f} reuse={reuse:.3f}"
    if hold:
        return True, f"s3s usable hold {note}"
    if not ok:
        return False, lesson
    return False, f"s3s usable drop {note}"


def who_holds_s3s(scores: dict) -> tuple[bool, str]:
    who2 = float(scores.get("who_2e") or 0.0)
    who3 = float(scores.get("who_3e") or 0.0)
    lesson = f"who2={who2:.3f} who3={who3:.3f}"
    if who2 + 1e-12 >= WHO_KEEP and who3 + 1e-12 >= WHO3_KEEP:
        return True, f"s3s who hold {lesson}"
    return False, f"s3s who drop {lesson}"


def sample_s4_item(rng, tokenizer, kind: str) -> dict:
    if kind == "easy":
        return make_compose_sentence_item(rng, tokenizer, n_entities=1, surface="train")
    if kind == "bind":
        return make_compose_sentence_item(rng, tokenizer, n_entities=2, surface="train")
    if kind == "instr":
        return make_compose_sentence_item(rng, tokenizer, n_entities=1, surface="train", instruct=True)
    if kind == "chat":
        return make_compose_sentence_item(rng, tokenizer, n_entities=1, surface="train", chat=True)
    if kind == "fade":
        if rng.random() < 0.5:
            return make_compose_sentence_item(rng, tokenizer, n_entities=1, surface="train", instruct=True)
        return make_compose_sentence_item(rng, tokenizer, n_entities=1, surface="train")
    if kind == "who":
        return make_who_sentence_item(rng, tokenizer, n_entities=2, surface="train")
    if kind == "easy_bind":
        n_entities = 1 if rng.random() < 0.7 else 2
        return make_compose_sentence_item(rng, tokenizer, n_entities=n_entities, surface="train")
    raise ValueError(f"unknown sent_kind {kind}")


def sentence_scores(compose: dict, sentence: dict | None = None) -> dict:
    scores = compose_scores(compose, sentence)
    scores["sent_1e_exact"] = _exact(compose, "compose_sent_1e_heldout")
    baby = ((sentence or {}).get("operators") or {}).get("baby") or {}
    scores["baby_sentence"] = float(baby.get("sentence_ok") or 0.0)
    return scores


def _scaffold_only(scores: dict) -> bool:
    scaffold = float(scores.get("prefix_sentence") or 0.0) - float(scores.get("bare_sentence") or 0.0)
    return float(scores.get("bare_sentence") or 0.0) < SENT_BARE_SIGNAL and scaffold >= SCAFFOLD_GAP


def _sent_real(scores: dict) -> bool:
    bare = float(scores.get("bare_sentence") or 0.0)
    exact_1e = float(scores.get("sent_1e_exact") or 0.0)
    if _scaffold_only(scores):
        return False
    return bare >= SENT_BARE_SIGNAL or (exact_1e >= SENT_1E_SIGNAL and bare >= 0.12)


def adjudicate_sentence(row: dict, *, parent: dict) -> tuple[str, str]:
    native = row.get("native") or {}
    compose = row.get("compose") or {}
    sentence = row.get("sentence")
    eng_ok, eng_lesson, collapsed = english_native_holds(native)
    mix_ok, mix_lesson = mix_holds_s3s(native)
    scores = sentence_scores(compose, sentence)
    slice_exact = float((row.get("d3_slice") or {}).get("free_exact") or 0.0)
    who_ok, who_lesson = who_holds_s3s(scores)
    bare = scores["bare_sentence"]
    exact_1e = scores["sent_1e_exact"]
    sent_delta = bare - float(parent.get("bare_sentence") or 0.0)
    exact_delta = exact_1e - float(parent.get("sent_1e_exact") or 0.0)
    lesson = (
        f"{mix_lesson}; {eng_lesson}; {who_lesson}; "
        f"bare={bare:.3f} prefix={scores['prefix_sentence']:.3f} "
        f"sent_1e={exact_1e:.3f} sent_2e={scores['sent_exact']:.3f} "
        f"one_word={scores['bare_one_word']:.3f} D3 slice {slice_exact:.3f}"
    )
    if collapsed or not eng_ok:
        return "KILL", lesson
    if not mix_ok:
        return "KILL", lesson
    if not who_ok:
        return "KILL", lesson
    if slice_exact + 1e-12 < S3S_D3_SLICE - D3_SLICE_KILL:
        return "KILL", lesson
    if _scaffold_only(scores) and bare < 0.12:
        return "HOLD", f"scaffold-only sentence {lesson}"
    if bare >= SENT_BARE_GRAD and not _scaffold_only(scores):
        return "ADVANCE", f"bare sentence {lesson} sentΔ={sent_delta:+.3f}"
    if _sent_real(scores):
        return "HOLD+", f"sentence signal {lesson} sentΔ={sent_delta:+.3f} 1eΔ={exact_delta:+.3f}"
    if sent_delta >= 0.12 or exact_delta >= 0.12:
        return "HOLD", lesson
    return "HOLD", lesson


def sentence_milestone(d3_free, native, usable, compose, sentence) -> tuple[bool, str]:
    mix_ok, _ = mix_holds_s3s(native)
    eng_ok, _, collapsed = english_native_holds(native)
    use_ok, _ = usable_holds_s3s(usable)
    scores = sentence_scores(compose, sentence)
    who_ok, _ = who_holds_s3s(scores)
    native_mix = mix_scores(native)
    d3_ok = d3_free is None or d3_free >= 180
    if collapsed or not who_ok or not use_ok or not d3_ok:
        return False, "no"
    if not _sent_real(scores):
        return False, "no"
    if mix_ok and eng_ok:
        if scores["sent_exact"] < 0.12:
            return True, "direct-fact-sentence"
        return True, "sentence"
    # One-word first_top1 can fall because some ordinary QA answers now start
    # with "The"/"entity". That is style mixing, not English death, if size-stop,
    # combine, mixed, and usable-chat remain alive above collapse.
    if (
        native_mix["color"] + 1e-12 >= 0.70
        and native_mix["size_stop"] + 1e-12 >= 0.90
        and native_mix["fact_combine"] + 1e-12 >= 0.60
        and native_mix["mixed_2e"] + 1e-12 >= 0.80
        and scores["bare_sentence"] + 1e-12 >= SENT_BARE_GRAD
        and not _scaffold_only(scores)
    ):
        return True, "direct-fact-sentence"
    return False, "no"


def probe_sentence(device, path: Path, tag: str, *, with_d3: bool = False, with_usable: bool = False) -> dict:
    from .data_language_bridge import build_e13_panels, load_tokenizer

    tokenizer = load_tokenizer()
    model, _config, _ckpt = load_experimental_baby(path, device)
    native = eval_panels(model, build_e13_panels(tokenizer, n=32), device, overwrite=None, arms=("native",))["native"]
    compose = eval_panels(model, build_s3_panels(tokenizer, n=32), device, overwrite=None, arms=("native",))["native"]
    sentence = run_sentence_decode(model, tokenizer, device, include=LIGHT_SENTENCE)
    usable = run_usable_chat(model, tokenizer, device) if with_usable else None
    d3 = None
    d3_free = None
    if with_d3:
        eng_ok, _, _ = english_native_holds(native)
        d3 = run_d3_log(model, device, tag, english_ok=eng_ok)
        d3_free = int(d3["long_gap"]["free_exact"])
    else:
        slice_row = cheap_d3_slice(model, device)
        d3 = {"slice": slice_row}
    scores = sentence_scores(compose, sentence)
    report = {
        "id": tag,
        "checkpoint": str(path),
        "checkpoint_sha256": digest(path),
        "authoritative": False,
        "native": mix_scores(native),
        "compose": {
            k: {"first_top1": (compose.get(k) or {}).get("first_top1"), "free_exact": (compose.get(k) or {}).get("free_exact")}
            for k in compose
        },
        "sentence_scores": scores,
        "sentence": slim_sentence(sentence),
        "usable": None if usable is None else slim_usable(usable),
        "d3_free": d3_free,
        "d3_slice": None if d3 is None else (d3.get("slice") or d3.get("long_gap")),
        "native_full": slim_panels(native),
        "compose_full": slim_panels(compose),
        "bare_examples": [
            {"id": row.get("id"), "decoded": row.get("decoded"), "full": row.get("full"), "sentence_ok": row.get("sentence_ok")}
            for row in ((sentence.get("operators") or {}).get("bare") or {}).get("rows") or []
        ][:8],
    }
    write(
        OUT / f"{tag}.json",
        {k: v for k, v in report.items() if k not in {"native_full", "compose_full"}},
    )
    print(json.dumps({"phase": f"{tag}_sentence", "mix": report["native"], "sentence": scores, "d3_free": d3_free}, default=str), flush=True)
    _drop_cuda(model)
    return report


def packed_best(row: dict) -> dict:
    return {
        "update": row["update"],
        "checkpoint": row["checkpoint"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "verdict": row.get("recover_verdict"),
        "lesson": row.get("recover_lesson"),
        "native": mix_scores(row["native"]),
        "sentence": row.get("sentence_scores") or sentence_scores(row.get("compose") or {}, row.get("sentence")),
        "d3_slice": row.get("d3_slice"),
    }


def score_trained(recipe_id: str, trained: dict, parent_scores: dict) -> dict:
    history = trained.get("history") or []
    scored = []
    best_row = None
    rank = {"KILL": -1, "HOLD": 0, "HOLD+": 1, "ADVANCE": 2}
    best_key = (-1, -1.0, -1.0)
    last_packed = None
    for row in history:
        verdict, lesson = adjudicate_sentence(row, parent=parent_scores)
        scores = sentence_scores(row.get("compose") or {}, row.get("sentence"))
        ledger_append(
            {
                "id": f"{recipe_id}_u{int(row['update']):05d}",
                "change": trained["recipe"]["note"],
                "verdict": verdict,
                "lesson": lesson,
                "native": mix_scores(row["native"]),
                "d3": row.get("d3_slice"),
                "checkpoint_sha256": row.get("checkpoint_sha256"),
                "recipe": recipe_id,
                "update": row.get("update"),
            }
        )
        print(
            json.dumps(
                {
                    "phase": "sentence_canary",
                    "recipe": recipe_id,
                    "update": row["update"],
                    "verdict": verdict,
                    "lesson": lesson,
                    "sentence": scores,
                },
                default=str,
            ),
            flush=True,
        )
        packed = {
            "update": row["update"],
            "checkpoint": row["checkpoint"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "verdict": verdict,
            "lesson": lesson,
            "native": mix_scores(row["native"]),
            "sentence": scores,
            "d3_slice": row.get("d3_slice"),
        }
        last_packed = packed
        scored.append(packed)
        if verdict == "KILL":
            continue
        key = (rank.get(verdict, 0), scores["bare_sentence"], scores["sent_1e_exact"])
        if key > best_key:
            best_key = key
            best_row = {**row, "recover_verdict": verdict, "recover_lesson": lesson, "sentence_scores": scores}
    return {
        "history": scored,
        "best": None if best_row is None else packed_best(best_row),
        "best_row": best_row,
        "last": last_packed,
        "out_dir": str(trained.get("out_dir")),
    }


def verify_sentence(device, recipe_id: str, best_row: dict) -> dict:
    ckpt = Path(best_row["checkpoint"])
    print(json.dumps({"phase": "sentence_verify", "recipe": recipe_id, "checkpoint": str(ckpt)}, default=str), flush=True)
    report = probe_sentence(device, ckpt, f"{recipe_id}_verify", with_d3=True, with_usable=True)
    native = report["native_full"]
    mix_ok, mix_lesson = mix_holds_s3s(native)
    eng_ok, eng_lesson, _ = english_native_holds(native)
    use_ok, use_lesson = usable_holds_s3s(report.get("usable"))
    d3_free = report.get("d3_free")
    kind_ok, kind = sentence_milestone(d3_free, native, report.get("usable"), report["compose_full"], report.get("sentence"))
    payload = {
        "recipe": recipe_id,
        "checkpoint": str(ckpt),
        "sha256": best_row.get("checkpoint_sha256"),
        "d3_full": d3_free,
        "mix_ok": mix_ok,
        "eng_ok": eng_ok,
        "use_ok": use_ok,
        "milestone": kind_ok,
        "milestone_kind": kind if kind_ok else None,
        "lesson": f"{mix_lesson}; {eng_lesson}; {use_lesson}; D3 {d3_free}/215; {report.get('sentence_scores')}",
        "native": mix_scores(native),
        "sentence": report.get("sentence_scores"),
        "bare_examples": report.get("bare_examples"),
        "usable": None
        if report.get("usable") is None
        else {
            "turn": report["usable"]["autoregressive"]["usable_turn"],
            "turn4": report["usable"]["autoregressive_4turn"]["usable_turn"],
            "stop": report["usable"]["autoregressive_4turn"]["period_stop"],
            "reuse": report["usable"]["autoregressive_4turn"]["fact_reuse"],
        },
    }
    campaign_update(
        {
            "status": f"s4 {kind} MILESTONE" if kind_ok else f"s4 verify {recipe_id}",
            "last_verify": {
                k: payload[k]
                for k in ("recipe", "checkpoint", "sha256", "d3_full", "milestone", "milestone_kind", "lesson")
                if k in payload
            },
        }
    )
    write(OUT / "S4.json", {"last_verify": payload})
    print(json.dumps({"phase": "sentence_verify_done", "milestone": kind_ok, "kind": kind, "lesson": payload["lesson"]}, default=str), flush=True)
    return payload


def _install_followup(recipe_id: str, *, sentence_p: float, sent_kind: str, seed: int, note: str, lr_scale: float = 0.5) -> None:
    from .selection_stack2 import RECIPES

    RECIPES[recipe_id] = {
        "seed": seed,
        "mix": "compose_lock",
        "language_p": max(0.20, 0.35 - sentence_p * 0.4),
        "structured_p": max(0.20, 0.30 - sentence_p * 0.3),
        "sentence_p": sentence_p,
        "sent_kind": sent_kind,
        "updates": SENTENCE_UPDATES,
        "eval_every": SENTENCE_UPDATES,
        "parent": "s3s",
        "cheap_d3": True,
        "remainder_span": True,
        "compose": True,
        "lock_from_drop": True,
        "lr_scale": lr_scale,
        "note": note,
    }


def _run_arm(device, recipe_id: str, parent_scores: dict, results: dict, *, resume: Path | None = None) -> dict:
    from .selection_stack2 import RECIPES, train_recipe

    print(
        json.dumps(
            {
                "phase": "sentence_arm",
                "recipe": recipe_id,
                "note": RECIPES[recipe_id]["note"],
                "resume": None if resume is None else str(resume),
            },
            default=str,
        ),
        flush=True,
    )
    trained = train_recipe(device, recipe_id, resume=resume)
    scored = score_trained(recipe_id, trained, parent_scores)
    _drop_cuda(trained.pop("model", None))
    results["arms"][recipe_id] = {k: v for k, v in scored.items() if k != "best_row"}
    write(OUT / "S4.json", {k: v for k, v in results.items() if k != "promising_row"})
    return scored


def _bare(best: dict | None) -> float:
    if not best:
        return 0.0
    return float((best.get("sentence") or {}).get("bare_sentence") or 0.0)


def _verdict(best: dict | None) -> str:
    return str((best or {}).get("verdict") or "KILL")


def run_sentence_loop(device) -> dict:
    require_identities(require_s2a=True, require_s2m=True, require_s3s=True)
    campaign_update(
        {
            "status": "s4 genuine short-sentence canaries from s3s",
            "s3s_parent": {"path": str(S3S_SURVIVOR), "sha256": S3S_SURVIVOR_SHA},
            "hypothesis": (
                "s3s retrieves/composes but answers one word. A light 1-entity sentence "
                "dose with varied surfaces can teach expression without repeating s3b-e overwrite."
            ),
            "d3_role": "retention, not the objective",
            "who_role": "retention, not the objective",
        }
    )
    parent = probe_sentence(device, S3S_SURVIVOR, "s3s_sentence_zeroshot")
    parent_scores = parent["sentence_scores"]
    results = {
        "parent": {k: parent[k] for k in ("sentence_scores", "native", "sentence", "d3_slice") if k in parent},
        "arms": {},
        "verifies": [],
        "milestone": False,
    }
    write(OUT / "S4.json", results)

    next_id = "s4a"
    resume = None
    seed = 325041
    for _step in range(MAX_ARMS):
        scored = _run_arm(device, next_id, parent_scores, results, resume=resume)
        best = scored.get("best")
        last = scored.get("last") or best
        verdict = _verdict(best) if best is not None else "KILL"
        bare = _bare(best) if best is not None else 0.0
        last_scores = (last or {}).get("sentence") or {}
        prefix = float(last_scores.get("prefix_sentence") or 0.0)
        exact_1e = float(last_scores.get("sent_1e_exact") or 0.0)
        english_dead = verdict == "KILL" and best is None

        if best is not None and verdict in {"HOLD+", "ADVANCE"} and bare >= SENT_BARE_SIGNAL:
            verified = verify_sentence(device, next_id, {"checkpoint": best["checkpoint"], "checkpoint_sha256": best.get("checkpoint_sha256")})
            results.setdefault("verifies", []).append(verified)
            results["last_verify"] = verified
            write(OUT / "S4.json", {k: v for k, v in results.items() if k != "promising_row"})
            if verified.get("milestone"):
                results["milestone"] = True
                results["winner"] = verified
                print(json.dumps({"phase": "sentence_done", "milestone": True, "kind": verified.get("milestone_kind"), "winner": next_id}, default=str), flush=True)
                return results
            if verified.get("eng_ok") and verified.get("mix_ok") and float((verified.get("sentence") or {}).get("bare_sentence") or 0.0) >= 0.12:
                seed += 10
                follow = f"{next_id}x"
                _install_followup(
                    follow,
                    sentence_p=0.12,
                    sent_kind="easy",
                    seed=seed,
                    note=f"Verify of {next_id} was real-but-shy. Extend light 1e sentence +25.",
                )
                next_id, resume = follow, Path(verified["checkpoint"])
                continue

        if english_dead or verdict == "KILL":
            if "s4f" not in results["arms"]:
                _install_followup(
                    "s4f",
                    sentence_p=0.08,
                    sent_kind="easy",
                    seed=seed,
                    note="s4a/prior killed retention. Same 8% 1e dose at quarter LR.",
                    lr_scale=0.25,
                )
                next_id, resume, seed = "s4f", None, seed + 10
                continue
            campaign_update({"status": "s4 killed retention; light sentence still overwrites", "s4": {k: v.get("best") for k, v in results.get("arms", {}).items()}})
            write(OUT / "S4.json", {k: v for k, v in results.items() if k != "promising_row"})
            print(json.dumps({"phase": "sentence_done", "milestone": False, "reason": "retention kill"}, default=str), flush=True)
            return results

        if bare < 0.12 and exact_1e < 0.12 and prefix < 0.12:
            if "s4c" not in results["arms"] and next_id != "s4c":
                next_id, resume = "s4c", None
                continue
            if "s4d" not in results["arms"]:
                next_id, resume = "s4d", None
                continue
            if "s4h" not in results["arms"] and prefix >= 0.12:
                seed += 10
                _install_followup(
                    "s4h",
                    sentence_p=0.10,
                    sent_kind="fade",
                    seed=seed,
                    note="Instruct moved prefix. Fade instruct→bare so expression is unprompted.",
                )
                next_id, resume = "s4h", Path((last or best)["checkpoint"]) if last or best else None
                continue
            if "s4g" not in results["arms"]:
                seed += 10
                _install_followup(
                    "s4g",
                    sentence_p=0.12,
                    sent_kind="chat",
                    seed=seed,
                    note="Ordinary CE did not become bare decode. Try Human/Baby surface, still score bare.",
                )
                next_id, resume = "s4g", None
                continue
            campaign_update({"status": "s4 canaries finished without genuine bare sentences", "s4": {k: v.get("best") for k, v in results.get("arms", {}).items()}})
            write(OUT / "S4.json", {k: v for k, v in results.items() if k != "promising_row"})
            print(json.dumps({"phase": "sentence_done", "milestone": False, "reason": "no bare sentence signal"}, default=str), flush=True)
            return results

        if 0.12 <= bare < SENT_BARE_SIGNAL or exact_1e >= SENT_1E_SIGNAL:
            seed += 10
            follow = f"{next_id}x"
            if follow in results["arms"]:
                follow = f"{next_id}y"
            _install_followup(
                follow,
                sentence_p=0.12,
                sent_kind="easy",
                seed=seed,
                note=f"Weak sentence signal on {next_id}. Keep light 1e, +25 more updates.",
            )
            next_id, resume = follow, Path((best or last)["checkpoint"])
            continue

        if "s4c" not in results["arms"]:
            next_id, resume = "s4c", None
            continue
        campaign_update({"status": "s4 canaries finished without milestone", "s4": {k: v.get("best") for k, v in results.get("arms", {}).items()}})
        write(OUT / "S4.json", {k: v for k, v in results.items() if k != "promising_row"})
        print(json.dumps({"phase": "sentence_done", "milestone": False}, default=str), flush=True)
        return results

    campaign_update({"status": "s4 hit arm budget without milestone", "s4": {k: v.get("best") for k, v in results.get("arms", {}).items()}})
    write(OUT / "S4.json", {k: v for k, v in results.items() if k != "promising_row"})
    print(json.dumps({"phase": "sentence_done", "milestone": False, "reason": "arm budget"}, default=str), flush=True)
    return results
