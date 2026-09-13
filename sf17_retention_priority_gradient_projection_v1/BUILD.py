"""Build prospective SF17 from sealed SF13 with one retention-priority gradient projection."""
import difflib, hashlib, json, shutil
from datetime import datetime, timezone
from pathlib import Path

H=Path(__file__).resolve().parent; ROOT=H.parent; SF13=ROOT/'sf13_broad_coverage_kl_retention_v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def write_json(p,o): p.write_text(json.dumps(o,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
def main():
 assert not any(p.name!='BUILD.py' for p in H.iterdir())
 r=json.loads((SF13/'FREEZE_RECEIPT.json').read_text()); assert sha(SF13/'FREEZE_RECEIPT.json')==(SF13/'FREEZE_RECEIPT.sha256').read_text().split()[0]; assert sha(SF13/'SHA256SUMS.txt')==r['manifest_sha256']
 for line in (SF13/'SHA256SUMS.txt').read_text().splitlines():
  hh,n=line.split('  ',1); assert sha(SF13/n)==hh,n
 files=['D3_SELECTION.json','DEV_ORDER.json','DEV_SURFACE.json','EXTERNAL_INPUTS.json','KL_POOL_MANIFEST.json','KL_POOL.json','SCHEDULE.json','SF13_KL_SCHEDULE.json','SF2_ENGINE.py','SF2_PROTOCOL.json','TRAIN.json','TRAIN16_RETENTION.json']
 for n in files: shutil.copyfile(SF13/n,H/n)
 for d in ['data','sources']: shutil.copytree(SF13/d,H/d)
 old=(SF13/'CONTROLLER.py').read_text(encoding='utf-8')
 new=old.replace('"""SF13: broad-coverage retention-position sampling; identical SF11 loss and all other machinery.\n\nONE scientific variable vs SF11/SF12: retention position coverage / sampling geometry\n(contiguous sequential 160-entry slice -> frozen row-distributed 160-entry schedule).\nAll loss terms, curriculum, optimizer, scope, gates and persistence are unchanged from SF11."""','"""SF17: SF13 full-CE recipe plus retention-priority conflict projection.\n\nThe sole scientific variable is English-gradient composition: when factual CE+margin and parent-KL gradients conflict globally, remove only the factual component opposing KL.\nAll losses, weights, data, optimizer, scope, schedule, gates and evaluation remain SF13."""',1)
 helper=r'''

def retention_priority_project(factual_grads, retention_grads):
    """Project only factual gradient against fixed retention gradient on global conflict."""
    pairs=[(gf,gr) for gf,gr in zip(factual_grads,retention_grads) if gf is not None and gr is not None]
    dot=sum((gf*gr).sum(dtype=torch.float64) for gf,gr in pairs)
    fact_norm2=sum((gf*gf).sum(dtype=torch.float64) for gf in factual_grads if gf is not None)
    ret_norm2=sum((gr*gr).sum(dtype=torch.float64) for gr in retention_grads if gr is not None)
    conflict=bool(dot.detach().item()<0.0 and ret_norm2.detach().item()>0.0)
    coefficient=(-dot/ret_norm2) if conflict else dot.new_zeros(())
    combined=[]
    for gf,gr in zip(factual_grads,retention_grads):
        if gf is None and gr is None: combined.append(None)
        elif gf is None: combined.append(gr)
        elif gr is None: combined.append(gf)
        else: combined.append(gf + coefficient.to(gf.dtype)*gr + gr if conflict else gf+gr)
    cosine=float((dot/torch.sqrt(fact_norm2*ret_norm2)).detach()) if fact_norm2.detach().item()>0 and ret_norm2.detach().item()>0 else 0.0
    telemetry={'gradient_dot':float(dot.detach()),'gradient_cosine':cosine,'gradient_conflict':conflict,'projection_coefficient':float(coefficient.detach()),'factual_grad_norm':float(torch.sqrt(fact_norm2).detach()),'retention_grad_norm':float(torch.sqrt(ret_norm2).detach())}
    return combined,telemetry
'''
 marker='\ndef verify(sealed=True):\n'; assert marker in new; new=new.replace(marker,helper+marker,1)
 new=new.replace('SF13_PROSPECTIVE_PREFLIGHT_PASS','SF17_PROSPECTIVE_PREFLIGHT_PASS').replace('SF13_PRE_PARENT_LOAD_PASS','SF17_PRE_PARENT_LOAD_PASS')
 target="        ce = kl_val = margin_loss = None\n        if u['kind'] == 'english':"
 repl="        ce = kl_val = margin_loss = None\n        projection_telemetry = None\n        manual_english_gradients = False\n        if u['kind'] == 'english':"
 assert new.count(target)==1; new=new.replace(target,repl,1)
 oldloss="""            loss = ce + E.LAMBDA_KL * kl_val
            margin_loss = margin_hinge_term(logits, u['ids'], idx)
            loss = loss + LAMBDA_MARGIN * margin_loss
        else:
            m.train()
            loss, _ = rt.binding_loss(m, rt.binding_docs_for_batch(pool, u['quartets'], u['documents']), device, pin)
        assert torch.isfinite(loss), 'nonfinite loss'
        loss.backward()"""
 newloss="""            margin_loss = margin_hinge_term(logits, u['ids'], idx)
            factual_loss = ce + LAMBDA_MARGIN * margin_loss
            retention_loss = E.LAMBDA_KL * kl_val
            loss = factual_loss + retention_loss
            active = [x for x in m.parameters() if x.requires_grad]
            factual_grads = torch.autograd.grad(factual_loss, active, allow_unused=True)
            retention_grads = torch.autograd.grad(retention_loss, active, allow_unused=True)
            combined_grads, projection_telemetry = retention_priority_project(factual_grads, retention_grads)
            for param, grad in zip(active, combined_grads): param.grad = grad
            manual_english_gradients = True
        else:
            m.train()
            loss, _ = rt.binding_loss(m, rt.binding_docs_for_batch(pool, u['quartets'], u['documents']), device, pin)
        assert torch.isfinite(loss), 'nonfinite loss'
        if not manual_english_gradients: loss.backward()"""
 assert new.count(oldloss)==1; new=new.replace(oldloss,newloss,1)
 metric="        if margin_loss is not None: rec.update(margin_loss=float(margin_loss.detach()))"
 metric2="        if margin_loss is not None: rec.update(margin_loss=float(margin_loss.detach()), **projection_telemetry)"
 assert new.count(metric)==1; new=new.replace(metric,metric2,1)
 prov="'kl_pool': sha(H / 'KL_POOL.json'), 'margin_M': MARGIN_M, 'lambda_margin': LAMBDA_MARGIN}"
 prov2="'kl_pool': sha(H / 'KL_POOL.json'), 'margin_M': MARGIN_M, 'lambda_margin': LAMBDA_MARGIN,\n                  'gradient_composition': 'retention_priority_global_conflict_projection'}"
 assert new.count(prov)==1; new=new.replace(prov,prov2,1)
 (H/'CONTROLLER.py').write_text(new,encoding='utf-8',newline='\n')
 (H/'CONTROLLER_DIFF.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='SF13/CONTROLLER.py',tofile='SF17/CONTROLLER.py')),encoding='utf-8',newline='\n')
 p=json.loads((SF13/'PROTOCOL.json').read_text()); parent_data=[(87044,87017,'9e293af6d16cb642ba8e2bf1aeb5ad392ff4b27399ebbf0b7b8fd2ef7339b81d',0.007036717671962123),(87045,87018,'839f7f5a60b33a1736376c8a68a02985fb05ba3230e56b7aaf20d1eba5b35102',0.006672408242356376),(87046,87019,'eb6a725173ff26516d876914ea4357cce7e3758616c3d659ff059d2354404517',0.007839491795780695)]
 p.update({'study':'SF17_RETENTION_PRIORITY_GRADIENT_PROJECTION_V1','created_utc':datetime.now(timezone.utc).isoformat(),'seeds':[x[0] for x in parent_data],
 'runs':[{'seed':s,'arm':'curriculum','parent_sf8_seed':ps,'parent_checkpoint':str(ROOT/f'sf8_margin_dose_comparison_v1/runs/seed_{ps}_low/checkpoint_200.pt'),'parent_checkpoint_sha256':hh,'d3_reference':d3} for s,ps,hh,d3 in parent_data],
 'hypothesis':'The replicated widening-versus-D3 tradeoff reflects direct conflict between factual CE+margin and parent-KL gradients; preserving the KL gradient while removing only the opposing factual component can retain widening and D3 simultaneously.',
 'manipulated_variable':'Relative to sealed SF13, replace ordinary summed English gradients with one deterministic global retention-priority conflict projection. If dot(g_factual,g_KL)<0, set g_factual_projected=g_factual-dot(g_factual,g_KL)/||g_KL||^2*g_KL; apply g_factual_projected+g_KL. If dot>=0, apply the unchanged sum. Binding gradients are unchanged.',
 'gradient_groups':{'factual':'full causal answer CE + 0.25 first-answer margin hinge M=1','retention':'1.0 full-vocabulary forward parent KL on frozen broad-coverage 160 positions','priority':'KL gradient is never projected','scope':'all currently trainable English parameters as one global vector','zero_or_unused':'unused component gradients remain absent; parameters used by only one objective receive that objective gradient unchanged'},
 'classification':{'priority1':'MECHANICAL_INCOMPLETE if integrity/execution prevents classification','priority2':'RETENTION_REGRESSION if any frozen TRAIN16, language, or binding gate fails','priority3':'RETENTION_PRIORITY_PROJECTION_SUPPORTED if >=2/3 reach u200 and pass every frozen endpoint gate including D3<=.010','priority4':'RETENTION_PRIORITY_PROJECTION_INSUFFICIENT if >=2/3 fail D3 while retention and widening movement survive','priority5':'WIDENING_STALLED_UNDER_PROJECTION if >=2/3 keep D3 and retention green through u200 but fail both widening endpoint gates','otherwise':'MIXED_OR_UNRESOLVED'},
 'interpretation':'This tests inference-training gradient composition under the SF13 recipe. Success supports sufficiency of the tested retention-priority projection; failure does not rule out all gradient surgery or establish a capacity limit.','next_action_rule':'Stop after this one prospective study; no rescue or second treatment.'})
 write_json(H/'PROTOCOL.json',p)
 write_json(H/'PROVENANCE.json',{'study':p['study'],'predecessor':str(SF13),'predecessor_receipt_sha256':sha(SF13/'FREEZE_RECEIPT.json'),'predecessor_manifest_sha256':sha(SF13/'SHA256SUMS.txt'),'unchanged_payload_hashes':{n:sha(H/n) for n in files},'sole_scientific_change':'retention-priority global conflict projection on English gradients','fresh_seeds':p['seeds'],'final_sacred':'LOCKED_NOT_ACCESSED'})
 print('SF17_BUILD_COMPLETE')
if __name__=='__main__': main()
