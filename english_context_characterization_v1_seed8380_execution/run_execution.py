"""Execute the sealed battery. This glue does not redefine scoring or items."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"C:\DaveLM-CADAVER")
FROZEN = ROOT / "english_context_characterization_v1_seed8380"
OUT = Path(__file__).resolve().parent
DEVICE = "cuda:0"
ORDER = ("graduate", "pilot0", "pilot1")
BINDING_SOURCE = ROOT / "language_pilot_0_tinystories_seed8380" / "run.py"
BINDING_SOURCE_SHA = "0ec915c4f1a7ed452e84c872305b67c22e930d07837dc8abcbb03257d9644649"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(name, obj):
    (OUT / name).write_bytes((json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                       indent=2, allow_nan=False) + "\n").encode("utf-8"))


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def state_digest(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)],
                            separators=(",", ":")).encode("utf-8") + b"\n")
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def source_inventory():
    paths = [Path(__file__), OUT / "verify_inputs.py", BINDING_SOURCE,
             ROOT / "treatment13_model.py", ROOT / "treatment13_config.py",
             Path(r"C:\DaveLM-v0.9\v0_8_2\model.py"),
             Path(r"C:\DaveLM-v0.9\v0_7\model.py"),
             FROZEN / "implementation" / "english_scoring.py"]
    return {str(p): {"sha256": sha(p), "bytes": p.stat().st_size} for p in paths}


def main():
    if (OUT / "RAW_SCORES.jsonl").exists():
        raise RuntimeError("RAW_SCORES already exists: refusing an unreviewed repeat/overwrite.")
    started = timestamp()
    start = time.monotonic()
    print("Reproducing all frozen material validation before model loading.", flush=True)
    verifier = load_module("execution_verify_inputs", OUT / "verify_inputs.py")
    verification = verifier.verify()
    if verification.get("status") != "PASS_EXECUTION_INPUT_VERIFICATION":
        raise RuntimeError(f"Frozen verification did not PASS: {verification.get('status')}")
    write_json("VERIFICATION.json", verification)
    manifest = json.loads((FROZEN / "MANIFEST.json").read_text(encoding="utf-8"))
    scoring = load_module("sealed_english_scoring", FROZEN / "implementation" / "english_scoring.py")
    import torch
    import torch.nn as nn
    import tokenizers
    assert platform.python_version() == manifest["runtime"]["construction_python"]
    assert tokenizers.__version__ == manifest["runtime"]["tokenizers"]
    assert str(torch.__version__) == manifest["runtime"]["torch_distribution_version_not_imported"]
    assert torch.cuda.is_available()
    runtime = scoring.configure_runtime(DEVICE)
    probe = torch.log_softmax(torch.tensor([0., 1., -1.], dtype=torch.float64,
                                          device=DEVICE), dim=0)
    assert probe.dtype == torch.float64 and bool(torch.isfinite(probe).all())
    runtime.update({"python": platform.python_version(), "tokenizers": tokenizers.__version__,
                    "hip": torch.version.hip, "device_name": torch.cuda.get_device_name(0),
                    "fp64_log_softmax_device_probe_passed": True})
    assert sha(BINDING_SOURCE) == BINDING_SOURCE_SHA
    # Compile only the four unchanged, existing inference definitions. Never
    # import the training runner or execute any of its training/main code.
    tree = ast.parse(BINDING_SOURCE.read_text(encoding="utf-8"))
    names = {"basis", "OrthoLocalizer", "rowpos", "binding_eval"}
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in nodes} == names and len(nodes) == 4
    namespace = {"torch": torch, "nn": nn, "defaultdict": defaultdict}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(BINDING_SOURCE), "exec"), namespace)
    sys.path.insert(0, str(ROOT))
    from treatment13_model import Treatment13Model
    sources_before = source_inventory()
    models, load_receipts, state_hashes = {}, {}, {}
    for key in ORDER:
        checkpoint = manifest["checkpoint_registry"][key]
        assert sha(checkpoint["path"]) == checkpoint["sha256"]
        raw = torch.load(checkpoint["path"], map_location="cpu", weights_only=True)
        state = raw["model_state_dict"]
        assert all(not t.is_floating_point() or t.dtype == torch.float32 for t in state.values())
        model = Treatment13Model()
        model.localizer = namespace["OrthoLocalizer"](
            *(state[f"localizer.{name}"].clone() for name in ("u", "q", "bs", "ba")))
        model.load_state_dict(state, strict=True)
        restored = model.state_dict()
        assert set(restored) == set(state)
        for name in restored:
            assert restored[name].shape == state[name].shape
            assert restored[name].dtype == state[name].dtype
            assert torch.equal(restored[name], state[name]), name
        assert sum(p.numel() for p in model.base_model.parameters()) == 10594944
        assert model.base_model.context_size == 256
        state_hashes[key] = state_digest(restored)
        model.to(DEVICE).eval()
        assert state_digest(model.state_dict()) == state_hashes[key]
        u = model.localizer.u.detach()
        norm = torch.linalg.vector_norm(u)
        assert bool(torch.isfinite(norm)) and float(norm) > 0
        unit = u / norm
        pivot = int(torch.argmax(torch.abs(unit)))
        e = torch.zeros_like(unit)
        e[pivot] = 1. if unit[pivot] >= 0 else -1.
        denominator = torch.dot(unit - e, unit - e)
        assert bool(torch.isfinite(denominator)) and abs(float(denominator)) >= 1e-12
        load_receipts[key] = {
            "checkpoint_sha256_before": checkpoint["sha256"],
            "strict_state_load": True, "all_state_tensors_equal_including_buffers": True,
            "state_tensor_count": len(state), "state_digest_before": state_hashes[key],
            "base_parameters": 10594944, "localizer_u_norm": float(norm),
            "householder_denominator": float(denominator),
            "graduate_guard_and_pilot_basis_arithmetic_equivalent": True}
        models[key] = model
        del raw, state, restored
        print(f"Loaded and exactly verified {key}; no forward pass yet.", flush=True)
    write_json("RUNTIME_AND_LOAD_RECEIPT.json", {
        "status": "PASS", "runtime": runtime, "loads": load_receipts,
        "implementation_sources_before": sources_before,
        "binding_helpers": sorted(names), "binding_helpers_ast_extracted_unchanged": True,
        "training_module_not_imported": True})
    items = [json.loads(line) for line in (FROZEN / "ITEMS.jsonl").read_text(encoding="utf-8").splitlines()]
    priors = [json.loads(line) for line in (FROZEN / "PRIORS.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(items) == 832 and len(priors) == 208
    forward_counts = {}
    with (OUT / "RAW_SCORES.jsonl").open("x", encoding="utf-8", newline="\n") as output:
        for key in ORDER:
            model = models[key]
            count = {"base_model": 0, "wrapper_or_specialized": 0}

            def count_base(module, args):
                count["base_model"] += 1

            def prohibit_specialized(module, args):
                count["wrapper_or_specialized"] += 1
                raise RuntimeError("English evaluation attempted a forbidden specialized path.")

            hooks = [model.base_model.register_forward_pre_hook(count_base)]
            hooks += [m.register_forward_pre_hook(prohibit_specialized)
                      for m in (model, model.localizer, model.wq, model.wk, model.wv, model.wo)]
            try:
                completed = 0
                for kind, records in (("contextual", items), ("prior", priors)):
                    for record in records:
                        score = scoring.score_candidate_pair(
                            model, record["prompt_token_ids"], record["candidate_token_ids"],
                            device=DEVICE,
                            correct_candidate_index=record["correct_index"] if kind == "contextual" else None)
                        row = {"checkpoint": key,
                               "checkpoint_sha256": manifest["checkpoint_registry"][key]["sha256"],
                               "kind": kind, "record": record, "score": score}
                        output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
                        completed += 1
                        if completed % 64 == 0 or completed == 1040:
                            output.flush()
                            print(f"English {key}: {completed}/1040 comparisons; {time.monotonic()-start:.1f}s elapsed", flush=True)
            finally:
                for hook in hooks:
                    hook.remove()
            assert count == {"base_model": 2080, "wrapper_or_specialized": 0}
            forward_counts[key] = dict(count)
            assert state_digest(model.state_dict()) == state_hashes[key]
    print("All English scoring complete; beginning the two nonsacred binding references.", flush=True)
    binding_results = {}
    for key in ORDER:
        binding_results[key] = {}
        for ref, metadata in manifest["common_binding_reference"]["references"].items():
            assert sha(metadata["path"]) == metadata["sha256"]
            data = json.loads(Path(metadata["path"]).read_text(encoding="utf-8"))
            docs = [d for quartet in data["quartets"] for d in quartet["docs"]]
            assert len(docs) == 80
            with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
                result = namespace["binding_eval"](models[key], docs, torch.device(DEVICE))
            bd = [r for r in result["rows"] if r["localization_category"] == "BOTH_DISTINCT"]
            result["queried_row_given_BD"] = {"correct": sum(r["selected_query_row_correct"] for r in bd), "n": len(bd)}
            result["answer_given_BD"] = {"correct": sum(r["answer_correct"] for r in bd), "n": len(bd)}
            result["reference_sha256"] = metadata["sha256"]
            result["checkpoint_sha256"] = manifest["checkpoint_registry"][key]["sha256"]
            binding_results[key][ref] = result
            print(f"Binding reference complete: {key}/{ref}, 80 documents.", flush=True)
    write_json("BINDING_REFERENCE_RESULTS.json", binding_results)
    post = {}
    for key in ORDER:
        actual = sha(manifest["checkpoint_registry"][key]["path"])
        digest = state_digest(models[key].state_dict())
        assert actual == manifest["checkpoint_registry"][key]["sha256"]
        assert digest == state_hashes[key]
        post[key] = {"checkpoint_sha256_after": actual, "state_digest_after": digest,
                     "checkpoint_bytes_unchanged": True, "in_memory_state_unchanged": True}
    assert source_inventory() == sources_before
    print("Rechecking the complete frozen material validation after evaluation.", flush=True)
    post_verification = verifier.verify()
    assert post_verification["status"] == "PASS_EXECUTION_INPUT_VERIFICATION"
    write_json("POST_VERIFICATION.json", post_verification)
    write_json("EXECUTION_STATUS.json", {
        "status": "SCORING_COMPLETE", "started_utc": started, "finished_utc": timestamp(),
        "elapsed_seconds": time.monotonic()-start, "checkpoints_order": list(ORDER),
        "runtime": runtime, "contextual_comparisons_per_checkpoint": 832,
        "logical_prior_comparisons_per_checkpoint": 208, "raw_score_rows": 3120,
        "english_forward_counts": forward_counts, "binding_documents_per_checkpoint": 160,
        "checkpoint_post_verification": post, "implementation_sources_unchanged": True,
        "mechanical_corrections": [], "sacred_exam_opened": False,
        "training": False, "optimizer_created": False, "backward": False,
        "generation": False, "frozen_artifacts_modified": False})
    print("SCORING_COMPLETE: all checkpoint and frozen-material postchecks passed.", flush=True)


if __name__ == "__main__":
    main()
