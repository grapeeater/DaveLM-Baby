"""Authorized read-only SF20 parent reproduction. Zero optimizer updates."""
import gc, json
from datetime import datetime, timezone
from pathlib import Path
import torch
from tokenizers import Tokenizer
import CONTROLLER as C
import SF2_ENGINE as E

H = Path(__file__).resolve().parent
REFERENCE = {
    87053: ("87017", 3.5873013138771057, 0.007036717671962123),
    87054: ("87018", 3.5746048688888550, 0.006672408242356376),
    87055: ("87019", 3.5949734449386597, 0.007839491795780695),
}

def main():
    p = C.verify(sealed=False)
    _, _, _, train16, dev_surface, dev_order, _, _, _ = C.load_inputs()
    root = H / "u0_preflight"
    assert not root.exists(), "U0 preflight is single-execution evidence"
    root.mkdir()
    tok = Tokenizer.from_file(p["tokenizer"])
    branches = {}
    for run in p["runs"]:
        seed = run["seed"]
        parent_label, expected_loss, expected_d3 = REFERENCE[seed]
        assert C.sha(run["parent_checkpoint"]) == run["parent_checkpoint_sha256"]
        E.rt.configure_runtime(seed)
        device = torch.device("cuda")
        model = E.rt.load_model(Path(p["parent_anchor"]), device, H)
        checkpoint = torch.load(run["parent_checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        out = root / f"seed_{seed}"
        out.mkdir()
        with torch.inference_mode():
            ret, surface, order, checks, d3 = C.run_eval(
                model, train16, dev_surface, dev_order, tok, device, out, 0)
        binding = {name: bool(value["gate"]) for name, value in checks["binding"].items()}
        observed_loss = checks["language"]["loss"]
        observed_d3 = d3["mean_combined_name_probability"]
        assert {k: ret[k] for k in ["correct", "exact", "reversals", "families"]} == {
            "correct": 16, "exact": 16, "reversals": 8, "families": 4}
        assert all(binding.values())
        assert abs(observed_loss - expected_loss) < 1e-9
        assert abs(observed_d3 - expected_d3) < 1e-12
        assert C.sha(run["parent_checkpoint"]) == run["parent_checkpoint_sha256"]
        branches[str(seed)] = {
            "sf8_parent": parent_label,
            "parent_checkpoint_sha256": run["parent_checkpoint_sha256"],
            "train16": {k: ret[k] for k in ["correct", "exact", "reversals", "families"]},
            "dev_surface": {"correct": surface["correct"], "exact": surface["exact"]},
            "dev_order": {"correct": order["correct"], "exact": order["exact"]},
            "language_ce": observed_loss,
            "binding": binding,
            "d3_measured": observed_d3,
            "d3_reference": expected_d3,
            "d3_diff": observed_d3 - expected_d3,
            "reproduces_parent": True,
        }
        del checkpoint, model
        gc.collect(); torch.cuda.empty_cache()
    record = {
        "study": p["study"],
        "record_type": "ACTUALLY_EXECUTED_U0_PREFLIGHT",
        "method": "Loaded each SF8-low parent under the frozen SF2 runtime and ran the inherited SF13 evaluation path. No optimizer was created and no update was performed.",
        "executed_utc": datetime.now(timezone.utc).isoformat(),
        "branches": branches,
        "checkpoint_loaded": True,
        "optimizer_created": False,
        "updates": 0,
        "treatment_outcomes_scored": False,
        "final_accessed": False,
        "sacred_accessed": False,
        "summary": "3/3 branches exactly reproduce their SF8-low parent U0 language and D3 references, preserve perfect TRAIN16, and pass both binding pools."
    }
    E.rt.atomic_json(record, H / "U0_PREFLIGHT_RECORD.json")
    print("SF20_U0_PREFLIGHT_PASS")

if __name__ == "__main__":
    main()
