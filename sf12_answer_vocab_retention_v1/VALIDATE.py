"""Bounded SF12 preflight: no checkpoint load, optimizer creation, inference, or update."""
import ast, hashlib, json, os, platform, sys
from collections import Counter
from pathlib import Path

import torch
import tokenizers
from tokenizers import Tokenizer

import CONTROLLER as C
import SF2_ENGINE as E

H = Path(__file__).resolve().parent
SF11 = H.parent / "sf11_narrow_breadth_widening_v1"


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def main():
    checks={}
    assert not (H/'runs').exists() and not (H/'logs').exists()
    assert json.loads((H/'REPORTER_SMOKE.json').read_text())['status']=='PASS'
    checks['reporter_smoke_test']='PASS'
    receipt=json.loads((SF11/'FREEZE_RECEIPT.json').read_text())
    assert sha(SF11/'FREEZE_RECEIPT.json') == (SF11/'FREEZE_RECEIPT.sha256').read_text().split()[0]
    assert sha(SF11/'SHA256SUMS.txt') == receipt['manifest_sha256']
    for line in (SF11/'SHA256SUMS.txt').read_text().splitlines():
        hh,n=line.split('  ',1); assert sha(SF11/n)==hh,n
    checks['sealed_sf11_source_verified']=True

    p=json.loads((H/'PROTOCOL.json').read_text())
    unchanged=['TRAIN.json','SCHEDULE.json','DEV_SURFACE.json','DEV_ORDER.json','TRAIN16_RETENTION.json','KL_POOL.json','D3_SELECTION.json','SF2_ENGINE.py']
    assert all(sha(H/n)==sha(SF11/n) for n in unchanged)
    checks['scientific_payload_byte_identical_to_sf11']=unchanged
    for path,hh in json.loads((H/'EXTERNAL_INPUTS.json').read_text()).items(): assert sha(path)==hh,path
    for r in p['runs']: assert sha(r['parent_checkpoint'])==r['parent_checkpoint_sha256']
    checks['parents_verified']={str(r['seed']):r['parent_checkpoint_sha256'] for r in p['runs']}

    assert platform.python_version()==p['runtime']['python']
    assert torch.__version__==p['runtime']['torch'] and tokenizers.__version__==p['runtime']['tokenizers']
    assert Path(sys.executable).resolve()==Path(p['runtime']['executable']).resolve()
    assert torch.cuda.is_available()
    checks['runtime']={'python':platform.python_version(),'torch':torch.__version__,'tokenizers':tokenizers.__version__,'cuda':True,'device':torch.cuda.get_device_name(0)}

    tok=Tokenizer.from_file(p['tokenizer'])
    assert sha(p['tokenizer'])==p['tokenizer_sha256']
    expected={'Alex':314,'Owen':536,'Mia':925,'Nora':512}
    observed={name:tok.encode(' '+name+'.').ids[0] for name in expected}
    assert observed==expected,(observed,expected)
    assert tuple(expected.values())==C.NAME_TOKEN_IDS
    checks['tokenizer']={'sha256':sha(p['tokenizer']),'name_first_token_ids':observed}

    schedule,items,idx,train16,ds,do,pool,kl,_=C.load_inputs()
    assert len(schedule)==200 and Counter(u['kind'] for u in schedule)=={'english':180,'binding':20}
    assert len(items)==48 and len({r['prompt'] for r in items})==48
    exposure=Counter(i for u in schedule if u['kind']=='english' for i in u['ids'])
    assert Counter(exposure.values())=={180:32,45:16}
    assert all(len(u['ids'])==36 for u in schedule if u['kind']=='english')
    checks['schedule']={'updates':200,'english':180,'binding':20,'batch':36,'presentations':6480,'new_items':32,'preservation_items':16,'new_per_item':180,'preservation_per_item':45}

    src=(H/'CONTROLLER.py').read_text()
    tree=ast.parse(src)
    assert 'LAMBDA_NAME = 1.0' in src and 'NAME_DELTA = 0.001' in src
    assert 'loss = loss + LAMBDA_NAME * name_val' in src
    assert 'E.compute_kl(m, teacher' not in src
    assert src.count('compute_kl_and_name_retention(')==3
    # D3 is invoked only by run_eval; training uses the KL pool alone for R_name.
    calls=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='measure_d3': calls.append(node.lineno)
    assert len(calls)==1 and calls[0] < next(n.lineno for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='main')
    assert all(x not in src for x in ['HELDOUT16','ALTERNATE48','COPY8','COMPETING64','FINAL'])
    checks['contamination_locks']={'D3_training_calls':0,'D3_evaluation_calls':1,'locked_panel_references':0,'final_or_sacred_references':0}

    class Base(torch.nn.Module):
        def __init__(self,bias): super().__init__(); self.bias=bias
        def forward(self,x):
            v=torch.arange(1024,dtype=torch.float32,device=x.device).view(1,1,-1)
            z=torch.sin(v*.003 + x.float().unsqueeze(-1)*.017)
            if self.bias:
                z=z.clone(); z[:,:,list(C.NAME_TOKEN_IDS)] += self.bias
            return z
    class Wrap(torch.nn.Module):
        def __init__(self,b): super().__init__(); self.base_model=Base(b)
    rows=[[1,2,3,4,5],[6,7,8,9]]
    sel=[(i%2,(i%3)+1) for i in range(160)]
    parent=Wrap(0.0); child=Wrap(.25)
    old=E.compute_kl(child,parent.base_model,torch.device('cpu'),rows,sel)
    new,rname,tel=C.compute_kl_and_name_retention(child,parent.base_model,torch.device('cpu'),rows,sel)
    assert torch.equal(old,new), (old,new)
    assert rname.item()>0 and tel['active_fraction']==1.0
    same_kl,same_r,same_tel=C.compute_kl_and_name_retention(parent,parent.base_model,torch.device('cpu'),rows,sel)
    assert same_r.item()==0.0 and same_tel['active_fraction']==0.0
    checks['objective_mechanics']={'full_KL_exactly_equal_to_authoritative_function':True,'positive_excess_activates':True,'teacher_equal_has_zero_R_name':True,'additional_forward':False}

    assert p['answer_vocabulary_retention']['delta']==.001 and p['answer_vocabulary_retention']['lambda_name']==1.0
    assert p['gates']==json.loads((SF11/'PROTOCOL.json').read_text())['gates']
    checks['historical_gates_unchanged']=True
    checks.update({'checkpoint_loaded':False,'optimizer_created':False,'updates':0,'final_accessed':False,'sacred_accessed':False})
    out={'status':'SF12_PROSPECTIVE_PREFLIGHT_PASS','checks':checks,'warnings':['SF12 does not change or test KL corpus coverage/disjointness.','Partial-widening comparison to SF11 is historical, not a concurrent control.']}
    (H/'PREFLIGHT.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    print(out['status'])

if __name__=='__main__': main()
