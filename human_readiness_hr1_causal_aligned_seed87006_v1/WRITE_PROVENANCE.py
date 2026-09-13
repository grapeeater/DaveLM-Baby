import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ref(path, role):
    p = Path(path)
    return {"path": str(p), "sha256": sha(p), "role": role}


payload = {
    "status": "CAUSAL_ALIGNMENT_PREFLIGHT_PASS_NO_TRAINING",
    "bundle": str(R),
    "parent": ref(r"C:\\DaveLM-CADAVER\\language_pilot_1_early_block_protection_seed8380\\pilot_run\\checkpoints\\seed_8380\\latest.pt", "explicit Pilot 1 parent"),
    "parent_lineage": {
        "material_manifest": ref(r"C:\\DaveLM-CADAVER\\language_pilot_1_early_block_protection_seed8380\\PILOT1_MATERIAL_MANIFEST.json", "Pilot 1 material provenance"),
        "completion_record": ref(r"C:\\DaveLM-CADAVER\\language_pilot_1_early_block_protection_seed8380\\pilot_run\\RESULTS.json", "Pilot 1 completion and final hash"),
        "training_spec": ref(r"C:\\DaveLM-CADAVER\\language_pilot_1_early_block_protection_seed8380\\pilot_run\\TRAINING_SPEC.json", "Pilot 1 training specification"),
        "starting_checkpoint_sha256": "fed298748e62def6f1751ef1bb7df0cdec51d53fac6e4c86e8c00a95dccdd430",
        "final_checkpoint_sha256": "2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb"
    },
    "root_cause": {
        "hr1_trainer": ref(R.parent / "human_readiness_hr1_seed87004_v8" / "HR1_TRAIN_REAL.py", "shifted target source"),
        "hr2_trainer": ref(R.parent / "human_readiness_hr2_seed87005_v7" / "HR2_TRAIN.py", "shifted target and unlikelihood source"),
        "shifted_expression": "y[i,1:len(z)] = z[1:]",
        "effect": "current-token copying at positions 1..n and EOS paired with padding"
    },
    "correct_reference": {
        "source": ref(R.parent / "language_compositional_p7.py", "independently validated causal alignment reference"),
        "also_validated": ref(R.parent / "language_sentencebound_p5.py", "independent causal alignment reference"),
        "correct_expression": "y[i,:len(z)-1] = z[1:]",
        "new_source": ref(R / "sources" / "CAUSAL_ALIGNED_TRAIN_REAL.py", "corrected successor trainer")
    },
    "reused_material": [
        ref(R.parent / "human_readiness_hr1_seed87004_v8" / "ENGLISH_TRAIN.jsonl", "source of copied materialized English training records"),
        ref(R.parent / "human_readiness_hr1_seed87004_v8" / "ENGLISH_DEV.jsonl", "source of copied materialized English DEV records"),
        ref(R.parent / "human_readiness_hr1_seed87004_v8" / "ENGLISH_SCHEDULE.json", "source of literal English schedule"),
        ref(R.parent / "human_readiness_hr1_seed87004_v8" / "BINDING_SCHEDULE.json", "source of literal binding schedule"),
        ref(R.parent / "language_pilot_1_early_block_protection_seed8380" / "binding_rehearsal.json", "established binding rehearsal pool"),
        ref(R.parent / "language_pilot_0_tinystories_seed8380" / "binding_dev.json", "nonsacred Pilot 0 binding DEV"),
        ref(R.parent / "language_pilot_1_early_block_protection_seed8380" / "binding_dev.json", "nonsacred Pilot 1 binding DEV")
    ],
    "runtime": {
        "python": "3.12.14",
        "torch": "2.12.0+rocm7.14.0",
        "tokenizers": "0.23.1",
        "cuda_available": True,
        "tokenizer": ref(r"C:\\DaveLM-v0.9\\tokenizer\\v0_7\\davelm_tokenizer.json", "approved tokenizer"),
        "base_model_source": ref(R.parent / "treatment13_model.py", "T13 wrapper"),
        "base_builder_source": ref(R.parent / "DaveLM-v0.9" / "v0_8_2" / "model.py", "authoritative base model builder"),
        "binding_objective_source": ref(R.parent / "fact_supervision_87001_eval_v1" / "PINNED_PILOT1_BINDING_IMPLEMENTATION.py", "pinned binding objective and scope")
    },
    "new_files": {
        "protocol": sha(R / "HR1_CAUSAL_ALIGNED_PROTOCOL.json"),
        "alignment_reference": sha(R / "sources" / "CAUSAL_ALIGNMENT.py"),
        "corrected_trainer": sha(R / "sources" / "CAUSAL_ALIGNED_TRAIN_REAL.py"),
        "preflight_validator": sha(R / "PREFLIGHT_VALIDATE.py")
    },
    "access_constraints": {
        "checkpoint_loaded": False,
        "optimizer_created": False,
        "optimizer_updates": 0,
        "behavioral_inference": False,
        "sealed_final_battery_accessed": False,
        "sacred_material_accessed": False
    }
}
(R / "PROVENANCE.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
