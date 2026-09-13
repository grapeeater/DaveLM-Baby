"""SF2 residual-item autopsy - shared config/loaders (READ-ONLY, no training)."""
from __future__ import annotations
import json, sys, hashlib
from pathlib import Path
import torch

ROOT = Path(r"C:\DaveLM-CADAVER")
BUNDLE = ROOT / "single_fact_acquisition_sf1_seed87011"
SF1_RUN = ROOT / "single_fact_acquisition_sf1_seed87011_run"
SF2_RUN = ROOT / "sf2_kl_parent_retention_run_v2" / "run"
PARENT = ROOT / "language_pilot_1_early_block_protection_seed8380" / "pilot_run" / "checkpoints" / "seed_8380" / "latest.pt"
TOKENIZER_PATH = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")

# Authorized checkpoints (identical weights => SF2 u0 == Pilot1 parent)
CHECKPOINTS = {
    "Pilot1_parent": PARENT,
    "SF1_u100": SF1_RUN / "checkpoint_100.pt",
    "SF2_u100": SF2_RUN / "checkpoint_100.pt",
    "SF2_u200": SF2_RUN / "checkpoint_200.pt",
}

SYSPATH_V09 = r"C:\DaveLM-v0.9"
if SYSPATH_V09 not in sys.path:
    sys.path.insert(0, SYSPATH_V09)
SYSPATH_SRC = str(BUNDLE / "sources")
if SYSPATH_SRC not in sys.path:
    sys.path.insert(0, SYSPATH_SRC)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_base(ckpt: Path, device):
    """Return base_model (DaveLMV082) only, eval, from a Treatment13 checkpoint."""
    from v0_8_2.model import build_model
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = {k[len("base_model."):]: v for k, v in raw["model_state_dict"].items()
          if k.startswith("base_model.")}
    m = build_model("untied")
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    m.eval()
    return m.to(device)


def load_train():
    return json.loads((BUNDLE / "TRAIN.json").read_text(encoding="utf-8"))


def load_schedule():
    return json.loads((BUNDLE / "SCHEDULE.json").read_text(encoding="utf-8"))


def load_raw(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


def record_identity(train):
    """actor, object, predicate, pair, family per id."""
    out = {}
    for r in train:
        out[r["id"]] = {
            "id": r["id"], "actor": r["actor"], "object": r["object"], "predicate": r["predicate"],
            "pair_id": r["pair_id"], "family_id": r["family_id"],
            "correct_index": r["correct_index"], "prompt": r["prompt"],
            "candidates": r["candidates"], "candidate_token_ids": r["candidate_token_ids"],
            "prompt_token_ids": r["prompt_token_ids"],
        }
    return out
