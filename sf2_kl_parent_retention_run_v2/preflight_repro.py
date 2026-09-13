"""SF2 preflight baseline reproduction (M9) - read-only, no optimizer, no training.

Runs under the frozen runtime and validates the harness's update-0 numbers against
the SF1 reference values before the harness is frozen and the run is launched.
Uses eval/inference only (plus a grad-enabled-but-never-backward KL==0 check).
"""
import json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

import SF2_ENGINE as E  # noqa: E402

BUNDLE = Path(r"C:\DaveLM-CADAVER\single_fact_acquisition_sf1_seed87011")
OUT = Path(r"C:\DaveLM-CADAVER\sf2_kl_parent_retention_run_v2\run")
OUT.mkdir(exist_ok=True)

proto = json.loads((HERE / "SF2_PROTOCOL.json").read_text(encoding="utf-8"))
p = E.read(BUNDLE / "PROTOCOL.json")
E.verify(BUNDLE)
E.rt.configure_runtime(p["seed"])
assert os.environ.get("PYTHONHASHSEED") == str(p["seed"])
import platform, tokenizers  # noqa: E402
assert platform.python_version() == "3.12.14" and tokenizers.__version__ == "0.23.1"
print("env ok", platform.python_version(), torch.__version__, torch.cuda.is_available())

device = torch.device("cuda")
m = E.rt.load_model(Path(p["parent"]), device, BUNDLE)
tok = Tokenizer.from_file(p["tokenizer"])
kl_pool, klproto = E.load_kl_pool()
KL_ROWS, KL_ENTRIES = kl_pool["rows"], kl_pool["entries"]

teacher = E.build_teacher(Path(p["parent"]), device)
E.assert_teacher_disjoint(m.base_model, teacher)
print("teacher disjoint OK; teacher trainable count =",
      sum(1 for x in teacher.parameters() if x.requires_grad))

train = E.read(BUNDLE / "TRAIN.json")
base_agg = E.panel(m, BUNDLE, OUT, "update0_acquisition", train, tok, device)
print("update0 acquisition:", {k: base_agg[k] for k in ("correct", "exact", "reversals", "families", "mean_margin")})

chk = E.checks(m, BUNDLE, device)
E.rt.atomic_json(chk, OUT / "update0_checks.json")
ce0 = chk["language"]["loss"]
print("update0 aligned CE", ce0, "PPL", chk["language"]["perplexity"])
print("binding gates:", {n: v["gate"] for n, v in chk["binding"].items()})

d3sel = json.loads(Path(proto["d3"]["selection_manifest"]).read_text(encoding="utf-8"))
d3r0 = E.measure_d3(m, d3sel, tok, device)
s0 = E.d3_summary(d3r0)
E.rt.atomic_json(s0, OUT / "d3_update0_summary.json")
with (OUT / "d3_update0_RAW.jsonl").open("w", encoding="utf-8", newline="\n") as h:
    for r in d3r0:
        h.write(json.dumps(r, ensure_ascii=False) + "\n")
print("update0 D3 mean four-name mass", s0["mean_combined_name_probability"])
print("update0 D3 per-name", {k: round(v, 6) for k, v in s0["per_name_mean_probability"].items()})

sel1 = E.pick_kl_entries(KL_ENTRIES, 1)
m.eval()
kl0 = E.compute_kl(m, teacher, device, KL_ROWS, sel1)
m.train()
print("KL at baseline (update-1 selection)", float(kl0))

# M9 assertions
assert base_agg["correct"] == 9 and base_agg["exact"] == 0, base_agg
assert all(v["gate"] for v in chk["binding"].values())
assert abs(ce0 - 3.3907) / 3.3907 < 1e-4, ce0
assert abs(s0["mean_combined_name_probability"] - 0.000907) < 1e-4, s0
assert float(kl0) < 1e-9, float(kl0)
print("M9 BASELINE REPRODUCTION PASS")
