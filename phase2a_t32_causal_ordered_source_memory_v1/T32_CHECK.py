import json, sys, torch
from pathlib import Path
sys.path.insert(0, r'C:\DaveLM-CADAVER\baby_vnext_60m_design_v1')
sys.path.insert(0, r'C:\DaveLM-CADAVER\phase2a_t32_causal_ordered_source_memory_v1')
from baby_vnext.config import BabyVNextConfig
from baby_vnext.binding import BabyVNextWithBinding
from T32_MEMORY import T32MemoryModel
from T32_RUNNER import load_jsonl, _forward_row, PARENT

device=torch.device('cuda')
c=BabyVNextConfig.load(r'C:\DaveLM-CADAVER\baby_vnext_60m_design_v1\BABY_VNEXT_CONFIG.json')
p=torch.load(PARENT,map_location='cpu',weights_only=False)
base=BabyVNextWithBinding(c); base.load_state_dict(p['model_state_dict'],strict=True); base.to(device).eval()
t=T32MemoryModel(c); miss,un=t.load_state_dict(p['model_state_dict'],strict=False)
assert miss==['memory_gate'] and not un
with torch.no_grad(): t.memory_gate.zero_()
t.to(device).eval()
ids=torch.tensor([[2]+[38,73,69,356,309,271,279,72,281,485,18,225,61,573]],device=device)
with torch.no_grad():
    a=base.base_model.language_head(base.base_model.forward_hidden(ids))
    b=t.base_model.language_head(t.base_model.forward_hidden(ids))
    maxdiff=float((a-b).abs().max().cpu())
rows=load_jsonl(Path(r'C:\DaveLM-CADAVER\phase2a_t3_rebuilt_study_v1\data\qa_train.jsonl'))
r=rows[0]; prompt=[2]+[int(x) for x in r['prompt_token_ids']]; target=[int(x) for x in r['candidate_token_ids'][int(r['correct_index'])]]+[3]; full=prompt+target
logits,stats=_forward_row(t,full[:-1],r,device,len(prompt)-1)
loss=torch.nn.functional.cross_entropy(logits[0,-len(target):],torch.tensor(target,device=device))
loss.backward()
grad=float(t.memory_gate.grad.norm().cpu()) if t.memory_gate.grad is not None else 0.0
pre=float(stats['memory_pre_gate_norm'].max().cpu())
print(json.dumps({'parameter_count':sum(x.numel() for x in t.parameters()),'parent_neutral_max_logit_diff':maxdiff,'memory_pre_gate_norm':pre,'memory_gate_grad_norm':grad,'missing':miss,'unexpected':un},indent=2))
assert maxdiff == 0.0
assert pre > 0.0 and grad > 0.0
