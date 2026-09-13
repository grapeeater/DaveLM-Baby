"""Build SF18 from sealed SF13 with language-head freezing only on English updates."""
import difflib,hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
H=Path(__file__).resolve().parent; ROOT=H.parent; SF13=ROOT/'sf13_broad_coverage_kl_retention_v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def wj(p,o): p.write_text(json.dumps(o,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
def main():
 assert not any(p.name!='BUILD.py' for p in H.iterdir()); r=json.loads((SF13/'FREEZE_RECEIPT.json').read_text()); assert sha(SF13/'FREEZE_RECEIPT.json')==(SF13/'FREEZE_RECEIPT.sha256').read_text().split()[0] and sha(SF13/'SHA256SUMS.txt')==r['manifest_sha256']
 for line in (SF13/'SHA256SUMS.txt').read_text().splitlines(): hh,n=line.split('  ',1); assert sha(SF13/n)==hh,n
 fs=['D3_SELECTION.json','DEV_ORDER.json','DEV_SURFACE.json','EXTERNAL_INPUTS.json','KL_POOL_MANIFEST.json','KL_POOL.json','SCHEDULE.json','SF13_KL_SCHEDULE.json','SF2_ENGINE.py','SF2_PROTOCOL.json','TRAIN.json','TRAIN16_RETENTION.json']
 for n in fs: shutil.copyfile(SF13/n,H/n)
 for d in ['data','sources']: shutil.copytree(SF13/d,H/d)
 old=(SF13/'CONTROLLER.py').read_text(encoding='utf-8'); new=old.replace('"""SF13: broad-coverage retention-position sampling; identical SF11 loss and all other machinery.\n\nONE scientific variable vs SF11/SF12: retention position coverage / sampling geometry\n(contiguous sequential 160-entry slice -> frozen row-distributed 160-entry schedule).\nAll loss terms, curriculum, optimizer, scope, gates and persistence are unchanged from SF11."""','"""SF18: sealed SF13 full-CE widening with the untied language head frozen on English updates.\n\nThe sole scientific variable is English-update parameter scope. Binding updates restore the unchanged full T13 scope. All objectives, data, optimizer, schedule, gates and evaluation remain SF13."""',1)
 helper=r'''

HEAD_PARAMETER_NAMES = ('base_model.language_head.weight', 'base_model.language_head.bias')

def set_scope_sf18(model, binding):
    E.set_scope(model, binding)
    if not binding:
        for name,param in model.named_parameters():
            if name in HEAD_PARAMETER_NAMES:
                param.grad=None
                param.requires_grad_(False)

def scope_audit_sf18(model):
    out={}
    for binding in [False,True]:
        set_scope_sf18(model,binding)
        on=[]; off=[]
        for name,param in model.named_parameters():
            base_expected=binding or (name.startswith('base_model.') and not any(name.startswith(f'base_model.blocks.{i}.') for i in range(4)))
            expected=base_expected and (binding or name not in HEAD_PARAMETER_NAMES)
            assert param.requires_grad==expected and param.grad is None
            (on if expected else off).append(name)
        assert all(n in off for n in HEAD_PARAMETER_NAMES) if not binding else all(n in on for n in HEAD_PARAMETER_NAMES)
        out['binding' if binding else 'english']={'active':on,'frozen':off}
    return out
'''
 marker='\ndef verify(sealed=True):\n'; assert marker in new; new=new.replace(marker,helper+marker,1).replace('SF13_PROSPECTIVE_PREFLIGHT_PASS','SF18_PROSPECTIVE_PREFLIGHT_PASS').replace('SF13_PRE_PARENT_LOAD_PASS','SF18_PRE_PARENT_LOAD_PASS')
 assert new.count('rt.atomic_json(E.scope_audit(m), out / \'PARAMETER_SCOPE.json\')')==1; new=new.replace('rt.atomic_json(E.scope_audit(m), out / \'PARAMETER_SCOPE.json\')','rt.atomic_json(scope_audit_sf18(m), out / \'PARAMETER_SCOPE.json\')',1)
 assert new.count("E.set_scope(m, u['kind'] == 'binding')")==1; new=new.replace("E.set_scope(m, u['kind'] == 'binding')","set_scope_sf18(m, u['kind'] == 'binding')",1)
 prov="'kl_pool': sha(H / 'KL_POOL.json'), 'margin_M': MARGIN_M, 'lambda_margin': LAMBDA_MARGIN}"
 prov2="'kl_pool': sha(H / 'KL_POOL.json'), 'margin_M': MARGIN_M, 'lambda_margin': LAMBDA_MARGIN,\n                  'english_head_scope': 'frozen', 'binding_head_scope': 'unchanged_full_T13'}"
 assert new.count(prov)==1; new=new.replace(prov,prov2,1)
 (H/'CONTROLLER.py').write_text(new,encoding='utf-8',newline='\n'); (H/'CONTROLLER_DIFF.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='SF13/CONTROLLER.py',tofile='SF18/CONTROLLER.py')),encoding='utf-8',newline='\n')
 p=json.loads((SF13/'PROTOCOL.json').read_text()); ps=[(87047,87017,'9e293af6d16cb642ba8e2bf1aeb5ad392ff4b27399ebbf0b7b8fd2ef7339b81d',0.007036717671962123),(87048,87018,'839f7f5a60b33a1736376c8a68a02985fb05ba3230e56b7aaf20d1eba5b35102',0.006672408242356376),(87049,87019,'eb6a725173ff26516d876914ea4357cce7e3758616c3d659ff059d2354404517',0.007839491795780695)]
 p.update({'study':'SF18_ENGLISH_OUTPUT_HEAD_FREEZE_V1','created_utc':datetime.now(timezone.utc).isoformat(),'seeds':[x[0] for x in ps],'runs':[{'seed':s,'arm':'curriculum','parent_sf8_seed':pa,'parent_checkpoint':str(ROOT/f'sf8_margin_dose_comparison_v1/runs/seed_{pa}_low/checkpoint_200.pt'),'parent_checkpoint_sha256':hh,'d3_reference':d3} for s,pa,hh,d3 in ps],
 'hypothesis':'Because factual improvement is upstream-dominant but the changed untied head amplifies upstream-driven name-mass inflation, freezing only the output head on English updates can preserve strong full-CE widening while containing D3.','manipulated_variable':'Relative to sealed SF13, freeze exactly base_model.language_head.weight and base_model.language_head.bias during English updates. Restore the unchanged full T13 trainable scope, including the head, during binding updates.',
 'scope_change':{'english_frozen_additions':['base_model.language_head.weight','base_model.language_head.bias'],'english_other_scope':'identical SF13','binding_scope':'identical full T13','optimizer':'unchanged persistent AdamW; head optimizer state and parameters remain untouched during English updates and resume on binding updates'},
 'classification':{'priority1':'MECHANICAL_INCOMPLETE if execution/integrity prevents classification','priority2':'RETENTION_REGRESSION if any frozen TRAIN16, language, or binding gate fails','priority3':'OUTPUT_HEAD_CONSTRAINT_SUPPORTED if >=2/3 reach u200 and pass every frozen endpoint gate including D3<=.010','priority4':'OUTPUT_HEAD_CONSTRAINT_INSUFFICIENT if >=2/3 fail D3 while retention and widening movement survive','priority5':'WIDENING_STALLED_WITH_HEAD_FROZEN if >=2/3 keep D3 and retention green through u200 but fail both widening endpoint gates','otherwise':'MIXED_OR_UNRESOLVED'},
 'interpretation':'Success supports sufficiency of English-update untied-head freezing under SF18. Failure does not prove readout constraints generally ineffective or identify the training-time mechanism.','next_action_rule':'Stop and report after this one treatment; no rescue or automatic second treatment.'})
 wj(H/'PROTOCOL.json',p); wj(H/'PROVENANCE.json',{'study':p['study'],'predecessor':str(SF13),'predecessor_receipt_sha256':sha(SF13/'FREEZE_RECEIPT.json'),'predecessor_manifest_sha256':sha(SF13/'SHA256SUMS.txt'),'unchanged_payload_hashes':{n:sha(H/n) for n in fs},'sole_scientific_change':'freeze untied language head during English updates only','fresh_seeds':p['seeds'],'final_sacred':'LOCKED_NOT_ACCESSED'}); print('SF18_BUILD_COMPLETE')
if __name__=='__main__': main()
