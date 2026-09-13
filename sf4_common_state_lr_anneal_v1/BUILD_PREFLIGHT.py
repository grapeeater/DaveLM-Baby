"""Outcome-blind package construction; no checkpoint loading or training."""
import ast, collections, hashlib, json, os, platform, shutil, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=Path(r'C:\DaveLM-CADAVER\single_fact_acquisition_sf1_seed87011')
SF2=Path(r'C:\DaveLM-CADAVER\sf2_kl_parent_retention_run_v2')
import SF2_ENGINE as E
import torch, tokenizers

def put(name,obj):
    (ROOT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')

def main():
    assert not (ROOT/'FREEZE_RECEIPT.json').exists(), 'Never overwrite a sealed study'
    base=E.verify(BASE)
    assert E.sha(SF2/'FREEZE_RECEIPT.json') == (SF2/'FREEZE_RECEIPT.sha256').read_text(encoding='utf-8').strip()
    assert E.sha(SF2/'SHA256SUMS.txt') == json.loads((SF2/'FREEZE_RECEIPT.json').read_text(encoding='utf-8-sig'))['manifest_sha256']
    inputs={}
    for pkg,mname in [(SF2,'SHA256SUMS.txt'),(BASE,'SHA256SUMS.txt')]:
        for line in (pkg/mname).read_text(encoding='utf-8').splitlines():
            h,n=line.split('  ',1); path=pkg/n
            assert E.sha(path)==h,n
            inputs[str(path)]=h
        inputs[str(pkg/mname)]=E.sha(pkg/mname)
    for path,h in base['external_hashes'].items():
        assert E.sha(path)==h; inputs[path]=h
    for path in (sys.executable,torch.__file__,torch._C.__file__,tokenizers.__file__):
        inputs[str(Path(path).resolve())]=E.sha(path)
    inputs[base['parent']]=base['parent_sha256']; inputs[base['tokenizer']]=base['tokenizer_sha256']
    for name in ('SF2_ENGINE.py','SF2_PROTOCOL.json','KL_POOL.json','KL_POOL_MANIFEST.json'):
        assert E.sha(ROOT/name)==E.sha(SF2/name)
    for path in (ROOT/'sources').glob('*.py'):
        assert E.sha(path)==E.sha(SF2/'sources'/path.name)
    d3=Path(r'C:\DaveLM-CADAVER\sf1_readout_selection_forensic_v1\D3_SELECTION.json')
    assert E.sha(d3)=='3ae0f6a748f5545b5d0af2fd89e5beb5482e9e42713ba61567d1fea353db2226'
    shutil.copyfile(d3,ROOT/'D3_SELECTION.json'); inputs[str(d3)]=E.sha(d3)
    evidence=['sf2_residual_item_autopsy_v1/REPORT.md','comparative_model_anatomy_post_sf2_v1/REPORT.md',
              'sf3_prospective_treatment_v1/FINAL_REPORT.md','sf2_update100_replay_mismatch_autopsy_v1/REPORT.md',
              'sf2_update100_replay_mismatch_autopsy_v1/SELF_REPLAY_A_VS_B.json',
              'sf2_kl_parent_retention_run_v2/run/SF2_RESULTS_SUMMARY.json']
    for rel in evidence:
        path=ROOT.parent/rel; inputs[str(path)]=E.sha(path)
    schedule=E.read(BASE/'SCHEDULE.json'); train=E.read(BASE/'TRAIN.json')
    assert len(schedule)==200 and len(train)==16
    idx={r['id']:r for r in train}; lrs=[]; ei=0
    pool=E.read(BASE/'data/binding_rehearsal.json')['quartets']; assert len(pool)==80
    for number,u in enumerate(schedule,1):
        assert u['update']==number
        assert u['kind']==('binding' if number%10==0 else 'english')
        tlr=5e-5
        if u['kind']=='english':
            ei+=1
            assert collections.Counter(u['ids'])==collections.Counter({k:2 for k in idx})
            x,y=E.pad_batch([idx[i] for i in u['ids']],u['pad'])
            assert x.shape==(32,u['pad']) and u['pad']<=256
            for r,xx,yy in zip([idx[i] for i in u['ids']],x,y):
                c=r['candidate_token_ids'][r['correct_index']]
                z=[2]+r['prompt_token_ids']+c+[3]; k=len(r['prompt_token_ids'])
                assert len(c)==4 and xx[:len(z)-1].tolist()==z[:-1]
                assert yy[:k].tolist()==[-100]*k and yy[k:k+5].tolist()==c+[3]
                assert (yy[k+5:]==-100).all()
            if number>100:
                j=ei-90; tlr=5e-5*(90-j)/89
        else:
            assert len(E.rt.binding_docs_for_batch(pool,u['quartets'],u['documents']))==32
        lrs.append({'update':number,'kind':u['kind'],'english_index':ei,'control':5e-5,'treatment':tlr})
    assert ei==180
    assert all(r['control']==r['treatment'] for r in lrs[:100])
    assert [r['treatment'] for r in lrs if r['kind']=='binding']==[5e-5]*20
    put('LR_SCHEDULE.json',lrs)
    kl=E.read(ROOT/'KL_POOL.json')
    dev=[json.loads(l) for l in (BASE/'data/ENGLISH_DEV.jsonl').read_text(encoding='utf-8').splitlines()][:128]
    forbidden={tuple(r['token_ids']) for r in dev}|{tuple(r['prompt_token_ids']) for r in train}
    assert not any(tuple(row) in forbidden for row in kl['rows'])
    km=E.read(ROOT/'KL_POOL_MANIFEST.json'); source=km['rule']['source']
    assert E.sha(source)==km['rule']['source_sha256']; inputs[source]=E.sha(source)
    historical=E.read(BASE/'PREFLIGHT.json')
    for path,r in historical['overlap'].items():
        assert E.sha(path)==r['sha256']; inputs[path]=r['sha256']
    assert historical['actor_object_disjoint'] and historical['independent_semantics']=='PASS'
    put('INPUT_HASHES.json',inputs)
    p={
      'study':'SF4_COMMON_STATE_LATE_ENGLISH_LR_V1','base_bundle':str(BASE),
      'parent':base['parent'],'parent_sha256':base['parent_sha256'],
      'hypothesis':'Reducing late English step size may avoid an Owen-side boundary offset while preserving acquired context-conditioned selection and SF2 retention. Settling/overshoot is a hypothesis, not established dynamics.',
      'single_variable':'English-update learning-rate sequence at global updates 101-200',
      'common_trajectory':'Two independent fresh Pilot1 runs, same current runtime, seed87011, exact SF2 updates1-100; neither continues past100. Compare all model/optimizer/RNG/controller tensors and scalars exactly. First run is predesignated common state, never choose by outcomes.',
      'branch_state':'Preserve complete update100 restart: model, optimizer, Python/CPU/GPU RNG, scope, completed update, English index, protocol/input/schedule provenance. Both branches load identical values; no optimizer reset. Only arm provenance label differs.',
      'branch_freeze':'After common-state exact equality and all common100 gates pass, seal COMMON_STATE.pt, both initial branch states, equality report and branch receipt before either continuation. Both real load-only preflights must restore identical model/optimizer/RNG before any branch update. Since update101 has identical LR, preserve its complete state in both arms and require exact equality before treatment update102 (the first differing LR).',
      'control':'Continuation101-200 at5e-5 for every update.',
      'treatment':'Continuation101-200: English ordinal j=1..90 uses5e-5*(90-j)/89; binding LR remains5e-5. At j90 zero LR advances optimizer moments/step but not parameters, as normal AdamW semantics. No alternative floor/curve or tuning.',
      'interpretation_of_variable':'This changes cumulative late English step size and its temporal profile, including scaled decoupled decay and relative effective binding step pressure. It does not isolate annealing shape from total step size. No claim otherwise.',
      'data':'Exact frozen SF1 TRAIN16 twice/update (32), exact binding schedule32docs/update; exact SF2 KL rows/160 positions/English update; no regenerated data.',
      'schedule':'Global200:20*(9English+1binding), common100 then100/arm. Two common-prefix validation replicas add100 infrastructure-validation updates, not another treatment arm.',
      'objective':'Exact pinned SF2 full-vocabulary mean causal CE over four answer tokens+EOS plus1.0*D_KL(student||Pilot1) on160 ordinary positions, full1024 support,T1,float64 KL accumulation. Teacher eval+no_grad; student KL eval with gradients; factual CE train mode. Binding exact pinned causal answerCE+LAM1.0536573711078283 hard-min localization.',
      'scope':'English train token/position embeddings, blocks4-7, final norm, untied head; blocks0-3 and T13 specialized localizer/retrieval frozen with gradNone and optimizer inactivity. Binding all parameters. Same both arms.',
      'optimizer':'AdamW5e-5 common/control; betas(.9,.999),eps1e-8,wd.05,amsgradFalse,foreachFalse,fusedFalse; clip2.0, one persistent optimizer state restored exactly per branch; no LR scheduler object.',
      'runtime':{'python':'3.12.14','torch':'2.12.0+rocm7.14.0','tokenizers':'0.23.1','seed':87011,'PYTHONHASHSEED':'87011','deterministic_algorithms':True,'TF32':False,'autocast':False,'process':'fresh process per common replica and arm; same RNG state restored at branch; no NumPy RNG used'},
      'evaluations':{'common':[0,100],'arm_full':[100,200],'arm_train_only':[125,150,175],'order':'common_a then common_b; exact comparison+branch freeze; both load-only preflights; control then treatment; after both endpoints committed and classified, eligible-arm-only transfer in HELDOUT,ALTERNATE,COPY,COMPETING order. No interim selection.','rng':'All eval must leave RNG state exactly unchanged.'},
      'gates':{'acquisition':'At fixed200 correct16/16 AND exact16/16 AND reversal8/8 AND complete families4/4; positive four-token LL margin, ties fail; greedy normalEOS,max32, exact candidate+EOS.','binding':'Each80-doc pool independently answer>=76 AND BD>=76 AND collapse0 at0/100/200; never average.','language':'aligned DEV mean of four batchCEs over first128 rows <= own common update0CE+.25 at100/200; preserve historical reduction.','d3':'mean four-name mass<=.01 at100/200; prospective safety ceiling adopted from SF3 and SF2 predicted signature, not retroactive SF2 gate.','common_guard':'Stop if correct<10 AND exact<10 at100; exact common-state equality and exact loaded-branch equality required.','transfer':'Per arm only if ALL endpoint acquisition/retention gates pass. Both arm endpoints must first exist. No pooled qualification; no endpoint early stopping.'},
      'statistics':'Raw item/pair/family counts and paired arm differences; four family-level mean margins and per-pair outcomes; no population-generalization p-values or precision claims from four repeatedly trained families/one seed. Report residual and all other items, not only residual.',
      'stopping':'Any mandatory integrity/baseline/common-state/determinism/safety failure is HARD STOP. No tolerance relaxation, rescue, extra arms, or followup. Fixed endpoint selection only. Resume latest committed state only, no replay; missing scheduled eval only.',
      'classification':'Treatment passes acquisition only at16/16,16/16,8/8,4/4 with retention. Treatment/control contrast is within this state+seed+budget only. If both fail, pair completed acquisition fail. If treatment fails while control passes, no annealing benefit. Historical SF2_FAIL and SF3_HARD_STOP unchanged.',
      'final_access':False,'sacred_access':False,
      'locked_panels':'Do not parse/score before gate. Byte hash integrity checks permitted; inherited semantic/overlap validation used without opening panels.'}
    put('PROTOCOL.json',p)
    from CONTROLLER import equality,state_digest
    mock={'model':{'w':torch.tensor([1.,2.])},'optimizer':{'state':{0:{'step':torch.tensor(1),'exp_avg':torch.zeros(2)}}},'rng':{'python':(1,2)},'completed':100}
    E.rt.atomic_torch_save(mock,ROOT/'MOCK_STATE.pt')
    q=torch.load(ROOT/'MOCK_STATE.pt',weights_only=False)
    assert equality(mock,q) and state_digest(mock)==state_digest(q)
    q['completed']=101; assert not equality(mock,q)
    for n in ('CONTROLLER.py','BUILD_PREFLIGHT.py','BRANCH_FREEZE.py','FINALIZE.py'):
        ast.parse((ROOT/n).read_text(encoding='utf-8'))
    assert torch.cuda.is_available(); E.rt.configure_runtime(87011)
    assert platform.python_version()=='3.12.14' and tokenizers.__version__=='0.23.1'
    put('STATIC_PREFLIGHT.json',{'status':'PASS','input_files':len(inputs),'schedule_resolved':200,'english':180,'binding':20,
        'masking':'PASS','scope_and_objective':'byte-identical pinned SF2 sources; controller checks inactive state each step',
        'KL_full_token_array_disjoint':True,'overlap_limit':'Exact-array/previous decoded-prompt audit; not semantic disjointness or new held-out data.',
        'locked_panels':'UNPARSED/UNSCORED','restart_mock':'PASS with negative changed-state test',
        'runtime':{'python':platform.python_version(),'torch':torch.__version__,'tokenizers':tokenizers.__version__,'GPU':torch.cuda.get_device_name(0)},
        'checkpoint_loaded':False,'optimizer_created':False})
    files=[x for x in ROOT.rglob('*') if x.is_file() and '__pycache__' not in x.parts]
    lines=[E.sha(x)+'  '+x.relative_to(ROOT).as_posix() for x in sorted(files)]
    (ROOT/'PRETRAIN_SHA256SUMS.txt').write_text('\n'.join(lines)+'\n')
    put('FREEZE_RECEIPT.json',{'status':'SF4_PROSPECTIVE_PROTOCOL_FROZEN','manifest_sha256':E.sha(ROOT/'PRETRAIN_SHA256SUMS.txt'),'payload_files':len(lines)})
    (ROOT/'FREEZE_RECEIPT.sha256').write_text(E.sha(ROOT/'FREEZE_RECEIPT.json')+'  FREEZE_RECEIPT.json\n')
    for x in files+[ROOT/'PRETRAIN_SHA256SUMS.txt',ROOT/'FREEZE_RECEIPT.json',ROOT/'FREEZE_RECEIPT.sha256']:
        os.chmod(x,0o444)
    print('FROZEN',E.sha(ROOT/'FREEZE_RECEIPT.json'),flush=True)

if __name__=='__main__': main()
