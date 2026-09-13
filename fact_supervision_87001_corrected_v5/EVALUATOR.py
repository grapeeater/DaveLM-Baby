# Frozen v4 evaluator source; no auto-Primary/Confirmation path.
import torch
def candidate_ll(logits,prompt_len,candidate):
 lp=logits[0,prompt_len:prompt_len+len(candidate)].double().log_softmax(-1); return float(lp[torch.arange(len(candidate)),torch.tensor(candidate)].sum())
def correct(a,b): return a>b
