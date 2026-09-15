import json, sys, torch
from pathlib import Path
sys.path.insert(0, r'C:\DaveLM-CADAVER\phase2a_t32_causal_ordered_source_memory_v1')
from T32_RUNNER import load_jsonl, eval_panel, identity_fork_table, lang_ce, read_u16, read_u32, T3_DATA, LANG_DEV, LANG_STARTS, CTX, TOK_PATH
from T32_MEMORY import T32MemoryModel
from baby_vnext.config import BabyVNextConfig
from tokenizers import Tokenizer

def run(seed):
    b=Path(r'C:\DaveLM-CADAVER\phase2a_t32_causal_ordered_source_memory_v1'); ck=b/f'run_seed{seed}'/'checkpoints'/'checkpoint_1000.pt'
    p=torch.load(ck,map_location='cpu',weights_only=False); c=BabyVNextConfig.load(r'C:\DaveLM-CADAVER\baby_vnext_60m_design_v1\BABY_VNEXT_CONFIG.json'); m=T32MemoryModel(c); m.load_state_dict(p['model_state_dict'],strict=True); m.cuda().eval()
    rows=load_jsonl(T3_DATA/'qa_dev.jsonl'); audits=json.loads((T3_DATA/'audits.json').read_text()); names=json.loads((b/'data'/'name_inventory.json').read_text())['names']; forks=identity_fork_table(audits['dev_names'],names)
    tok=Tokenizer.from_file(str(TOK_PATH)); stream=read_u16(LANG_DEV); starts=read_u32(LANG_STARTS,1600)[:1280]
    on=eval_panel(m,rows,torch.device('cuda'),forks); ce=lang_ce(m,stream,starts,torch.device('cuda'),16)
    with torch.no_grad(): m.memory_gate.zero_()
    off=eval_panel(m,rows,torch.device('cuda'),forks); ce_off=lang_ce(m,stream,starts,torch.device('cuda'),16)
    out={'seed':seed,'checkpoint':str(ck),'memory_on':{k:v for k,v in on.items() if k!='per_row'},'memory_off':{k:v for k,v in off.items() if k!='per_row'},'language_ce_on':ce,'language_ce_off':ce_off,'protected':{'T3_TEST':'SEALED_UNOPENED','FINAL':'LOCKED','sacred':'LOCKED'}}
    return out
allout=[run(s) for s in (890001,890002)]; print(json.dumps(allout,indent=2)); Path(r'C:\DaveLM-CADAVER\phase2a_t32_causal_ordered_source_memory_v1\T32_ABLATION.json').write_text(json.dumps(allout,indent=2)+'\n')
