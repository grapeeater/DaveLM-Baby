"""Pre-seal unit checks for the Phase 2A T2 objective (read-only)."""
import importlib.util
import json
import sys
from pathlib import Path
import torch

BUNDLE = Path(r"C:\DaveLM-CADAVER\baby_vnext_phase2a_t2_ctxbind_v1")
ARCH = Path(r"C:\DaveLM-CADAVER\baby_vnext_60m_design_v1")
sys.path.insert(0, str(ARCH))

spec = importlib.util.spec_from_file_location("t2runner", BUNDLE / "PHASE2A_T2_RUNNER.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

from baby_vnext.config import BabyVNextConfig
from baby_vnext.binding import BabyVNextWithBinding
from tokenizers import Tokenizer

DEV = "cuda"
cfg = json.loads((BUNDLE / "PHASE2A_T2_CONFIG.json").read_text(encoding="utf-8"))
M = cfg["objective"]["M"]
tok = Tokenizer.from_file(cfg["tokenizer"]["path"])

results = {}

# 1. common-additive-shift invariance
torch.manual_seed(0)
s = torch.tensor([0.4, -1.2, 0.1, 0.9], dtype=torch.float64, requires_grad=True)
base = runner.margin_hinge_loss(list(s), 2, M)
shift = runner.margin_hinge_loss(list(s + 3.7), 2, M)
results["shift_invariance"] = float(abs(base - shift)) < 1e-12

# 2. candidate-order invariance
perm = [3, 0, 2, 1]
scores_perm = [s[i] for i in perm]
ci_perm = perm.index(2)
results["order_invariance"] = bool(abs(runner.margin_hinge_loss(list(s), 2, M)
                                       - runner.margin_hinge_loss(scores_perm, ci_perm, M)) < 1e-12)

# load parent model
from baby_vnext.checkpoint import _state_digest
payload = torch.load(cfg["parent"]["checkpoint"], map_location="cpu", weights_only=False)
model = BabyVNextWithBinding(BabyVNextConfig.load(ARCH / "BABY_VNEXT_CONFIG.json"))
model.load_state_dict(payload["model_state_dict"], strict=True)
model.to(DEV)

for n, p in model.named_parameters():
    p.requires_grad_(n.startswith("base_model."))
model.eval()


def score_cands(item):
    prompt = [2] + [int(x) for x in item["prompt_token_ids"]]
    P = len(prompt)
    out = []
    for cand in item["candidate_token_ids"]:
        c = [int(t) for t in cand]
        x = torch.tensor([prompt + c], device=DEV)
        logits, _ = model(x)
        lp = torch.log_softmax(logits[0, P - 1:P - 1 + len(c)], dim=-1)
        idx = torch.tensor(c, device=DEV)
        out.append(lp[torch.arange(len(idx)), idx].mean())
    return out


# first-token-colliding pair (Mia vs Max) constructed on a real DEV A prompt
rows = [json.loads(l) for l in open(BUNDLE / "data" / "qa_dev.jsonl", encoding="utf-8")]
base_item = next(r for r in rows if r["level"] == "A")
item = dict(base_item)
item["candidate_names"] = ["Mia", "Max"]
item["candidate_token_ids"] = [[int(t) for t in tok.encode(" Mia.").ids],
                               [int(t) for t in tok.encode(" Max.").ids]]
item["correct_index"] = 0
results["colliding_first_tokens"] = bool(item["candidate_token_ids"][0][0] == item["candidate_token_ids"][1][0])
scores = score_cands(item)
results["colliding_scores_differ"] = bool(abs(float(scores[0] - scores[1])) > 1e-6)
loss = runner.margin_hinge_loss(scores, item["correct_index"], M)
loss.backward()
results["nonzero_grad"] = any(p.grad is not None and bool((p.grad != 0).any())
                              for p in model.parameters() if p.requires_grad)
results["frozen_binding_no_grad"] = all(p.grad is None for n, p in model.named_parameters()
                                        if not n.startswith("base_model."))

# 3. no EOS/continuation target: appending an EOS token to the input must not change the score
model.zero_grad(set_to_none=True)
item2 = dict(item)
prompt = [2] + [int(x) for x in item2["prompt_token_ids"]]
c = [int(t) for t in item2["candidate_token_ids"][item2["correct_index"]]]
P = len(prompt)
x1 = torch.tensor([prompt + c], device=DEV)
x2 = torch.tensor([prompt + c + [3]], device=DEV)
with torch.no_grad():
    l1, _ = model(x1)
    l2, _ = model(x2)
    lp1 = torch.log_softmax(l1[0, P - 1:P - 1 + len(c)], dim=-1)
    lp2 = torch.log_softmax(l2[0, P - 1:P - 1 + len(c)], dim=-1)
    idx = torch.tensor(c, device=DEV)
    s1 = float(lp1[torch.arange(len(idx)), idx].mean())
    s2 = float(lp2[torch.arange(len(idx)), idx].mean())
results["eos_absent_from_score"] = bool(abs(s1 - s2) < 1e-6)

# 4. exact counts
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
results["counts"] = [trainable, frozen]
results["counts_ok"] = (trainable, frozen) == (60536064, 984321)
results["M"] = M

print(json.dumps(results, indent=2))
assert results["shift_invariance"] and results["order_invariance"]
assert results["colliding_first_tokens"] and results["colliding_scores_differ"]
assert results["nonzero_grad"] and results["frozen_binding_no_grad"]
assert results["eos_absent_from_score"] and results["counts_ok"]
print("UNIT_CHECKS_PASS")
