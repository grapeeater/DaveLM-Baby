"""One-time, outcome-blind SF6 materialization. No checkpoint loading or training."""
import hashlib, json, shutil, sys
from pathlib import Path
from datetime import datetime, timezone

H = Path(__file__).resolve().parent
ROOT = H.parent
S2 = ROOT / 'sf2_kl_parent_retention_run_v2'
S1 = ROOT / 'single_fact_acquisition_sf1_seed87011'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def write(p, v):
    Path(p).write_text(json.dumps(v, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8', newline='\n')

def manifest(p):
    return {n.replace('\\','/'): h for h,n in (l.split('  ',1) for l in p.read_text().splitlines() if l.strip())}

def main():
    assert not (H/'PROTOCOL.json').exists(), 'Do not overwrite a materialized protocol'
    assert sha(S2/'FREEZE_RECEIPT.json') == (S2/'FREEZE_RECEIPT.sha256').read_text().split()[0]
    assert sha(S2/'SHA256SUMS.txt') == read(S2/'FREEZE_RECEIPT.json')['manifest_sha256']
    assert sha(S1/'RECEIPT.json') == (S1/'RECEIPT.sha256').read_text().split()[0]
    assert sha(S1/'SHA256SUMS.txt') == read(S1/'RECEIPT.json')['manifest_sha256']
    hashes2, hashes1 = manifest(S2/'SHA256SUMS.txt'), manifest(S1/'SHA256SUMS.txt')
    for n,h in hashes2.items():
        assert sha(S2/n)==h, n
    reused=[]
    def copy(src, dest, expected):
        assert sha(src)==expected, str(src)
        target=H/dest
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists(), str(target)
        shutil.copyfile(src,target)
        assert sha(target)==expected
        reused.append({'source':str(src),'destination':dest,'sha256':expected})
    for n in ['SF2_ENGINE.py','SF2_PROTOCOL.json','KL_POOL.json','KL_POOL_MANIFEST.json']:
        copy(S2/n,n,hashes2[n])
    for n in hashes2:
        if n.startswith('sources/'):
            assert hashes1[n]==hashes2[n], n
            copy(S2/n,n,hashes2[n])
    for n in ['TRAIN.json','SCHEDULE.json','data/ENGLISH_DEV.jsonl','data/binding_rehearsal.json','data/binding_dev_pilot0.json','data/binding_dev_pilot1.json']:
        copy(S1/n,n,hashes1[n])
    for n in ['PROTOCOL.json','PREFLIGHT.json','SOURCE_PROVENANCE.json']:
        copy(S1/n,'authority/SF1_'+n,hashes1[n])
    d3=ROOT/'sf1_readout_selection_forensic_v1/D3_SELECTION.json'
    copy(d3,'D3_SELECTION.json','3ae0f6a748f5545b5d0af2fd89e5beb5482e9e42713ba61567d1fea353db2226')
    p=read(S1/'PROTOCOL.json')
    external=dict(p['external_hashes'])
    external[p['parent']]=p['parent_sha256']
    external[p['tokenizer']]=p['tokenizer_sha256']
    km=read(H/'KL_POOL_MANIFEST.json')
    external[km['rule']['source']]=km['rule']['source_sha256']
    for n,h in external.items(): assert sha(n)==h,n
    write(H/'EXTERNAL_INPUTS.json',external)
    # Run order is a fixed blocked alternation, chosen before any outcomes.
    runs=[{'seed':s,'arm':a} for s,arms in [(87011,['control','anneal']),(87012,['anneal','control']),(87013,['control','anneal'])] for a in arms]
    schedule=read(H/'SCHEDULE.json')
    lr=[]
    j=0
    for u in schedule:
        if u['kind']=='english' and u['update']>100: j+=1
        lr.append({'update':u['update'],'kind':u['kind'],'control':5e-5,
                   'anneal':5e-5*(90-j)/89 if u['kind']=='english' and u['update']>100 else 5e-5})
    assert j==90
    write(H/'LR_SCHEDULE.json',lr)
    proto={
      'study':'SF6_SAME_SEED_INDEPENDENT_LATE_ENGLISH_LR_ANNEAL_V1',
      'created_utc':datetime.now(timezone.utc).isoformat(),
      'hypothesis':'Reducing late English step size may improve the weakest factual margin/full acquisition while preserving retention. Constant-LR overshoot is a hypothesis, not an established cause.',
      'manipulated_variable':'English AdamW learning rate after global update100 only; LR-scaled weight decay consequently changes as an inseparable part of this LR intervention.',
      'parent':p['parent'],'parent_sha256':p['parent_sha256'],
      'tokenizer':p['tokenizer'],'tokenizer_sha256':p['tokenizer_sha256'],
      'runtime':{'executable':str(ROOT/'sf2_runtime/sf2venv/Scripts/python.exe'),'python':'3.12.14','torch':'2.12.0+rocm7.14.0','tokenizers':'0.23.1','float32':True,'autocast':False,'tf32':False,'deterministic_algorithms':True},
      'runs':runs,'seeds':[87011,87012,87013],
      'pairing':'Same stochastic seed within pair; independent fresh processes/parent/optimizer. No branch, shared mutable state, historical replay, or cross-run tensor-equality prerequisite. Materialized data order is identical across all runs.',
      'schedule':'Consume unchanged SCHEDULE.json: 200 updates=180 English+20 binding; bindings10,20,...,200. Every English batch is all16 TRAIN records twice, in literal ordering.',
      'lr':'Control5e-5 throughout. Anneal English updates1..1005e-5; post100 English ordinal j=1..90:5e-5*(90-j)/89. Thus update1015e-5,update199zero,update200binding5e-5. Consume LR_SCHEDULE.json.',
      'optimizer':{'type':'AdamW','lr':5e-5,'betas':[0.9,0.999],'eps':1e-8,'weight_decay':0.05,'amsgrad':False,'foreach':False,'fused':False,'gradient_clip':2.0,'state':'one fresh optimizer per run; continuous state; inactive grad=None and unchanged parameters/moments/step; zero-lr English update still computes gradients and advances active Adam state exactly once'},
      'english_objective':'Pinned prepare_example/pad_batch full-vocabulary causal CE over four response tokens+EOS only, mean supervised tokens; plus lambda1 D_KL(student||Pilot1), full1024support,T1,mean160positions,float64 accumulation->float32 scalar. Student KL eval with gradients; teacher separate/frozen/eval/no_grad; factual CE train mode. No new replay/objective.',
      'kl_data':'Unchanged KL_POOL.json165363positions/741rows; select160 in frozen rotation per English index1..180. No resampling.',
      'binding':'Unchanged pinned answer causal CE +1.0536573711078283 hard-min permutation-invariant localization. Same frozen rehearsal identities/order. Both nonsacred DEV pools separately.',
      'scope':'PINNED_MASKING.set_scope only: English every base_model parameter exceptblocks0..3; all specializedlocalizer/wq/wk/wv/wo frozen. Binding all parameters active. Do not call HR3 block3 scope.',
      'evaluation_updates':[0,100,200],
      'evaluation':'Pinned SF2 panel/score/checks/measure_d3. TRAIN16 only, aligned first128TinyStoriesDEV with mean4batchlosses, frozenD3positions256, both80documentbindingpools. No intermediate acquisition probes.',
      'baseline':'Native parent per-run9correct/0exact; alignedCE3.3907 relative tolerance1e-4; binding pass; D3.000907 absolute tolerance1e-4; teacherKLabs<1e-9. These are intact parent/output controls, not trajectory comparisons.',
      'gates':{'acquisition200':{'correct':16,'exact':16,'reversals':8,'families':4},'language100_200':'CE<=ownupdate0CE+.25','binding_each_pool100_200':'answer>=76,BD>=76,collapse==0','d3_100_200':'four-name mass<=.01 (explicit SF5/current-request gate; SF2 itself recorded D3 diagnostically)','acquisition_guard100':'STOP arm if correct<10 AND exact<10'},
      'stopping':'No endpoint selection. Each run stops at200 or its frozen100/200guard. Other preregistered independent runs continue after a scientific arm failure. Integrity/provenance/nonfinite/runtime failure halts study for mechanical resolution or report. No extra seeds/retries chosen by outcome/no rescue.',
      'transfer':'All HELDOUT/ALTERNATE/COPY/COMPETING remain LOCKED_UNSCORED throughout this study, including on acquisition success; report acquisition to user first. FINAL/sacred never accessed.',
      'persistence':'Atomic fsync rolling full model/optimizer/Python+CPU+GPU RNG,completed,scope,provenance aftereveryupdate; permanent100/200model+hash. Resume only latest valid committed state, nextcompleted+1. Diagnostics persisted independently; completed raw item results reused. No state forks. Corrupt restart stops. Metrics included in rolling state permit missing post-commit log recovery.',
      'primary_comparison':'Raw paired all-gate endpoint success for eachseed; treatment-only/control-only/both/neither counts. All6fixed before outcomes. Acquisition success is an individual endpoint classification, not proof that LR schedule caused success.',
      'secondary_comparison':'Per-item endpoint sequence margins; mean/minmargin; previously identified residual TRAIN:g0:carried:wooden boat:a0; raw reversal/family/exact counts; language,D3,binding. For all16items report paired delta200,delta100,delta200-delta100 descriptively.',
      'numerical_uncertainty':'Update100 pair differences arise before LR divergence under identical intended conditions and measure observed numerical variability. Report maxabs/meanabs margin offsets and counts. Historical observed<=.00697nat at200 is context, NOT a guaranteed bound. Endpoint changes comparable to prefix offsets are numerically fragile; no equality gate/no exclusions. Prefix-adjusted differences are descriptive, not an estimate with certified confidence. N3 cannot establish statistical certainty or generalization.',
      'study_interpretation':{'INCONCLUSIVE_EARLY_STOP':'Takes priority if any arm stops before200: report all runs and gates, but the planned3complete endpoint pairs were not obtained; no seed replacement or claim that a pre-intervention failure was caused by LR.','SUGGESTIVE_ANNEAL_BENEFIT':'atleast2treatment-only fullgate successes and0control-only successes','CONTROL_FAVORED_OR_MIXED':'anycontrol-only fullgate success; report if treatment-only also occurs','INCONCLUSIVE':'allotherpatterns, including bothpass/bothfail; do not equate inconclusive with equivalence','statistical_limit':'No population pvalue/CI; rawN3pairs and margins only. No claim of a proven optimization mechanism.'},
      'history':'SF1STOP_REGRESSION,SF2ACQUISITION_FAIL,SF3/SF4hardstops,SF5failedprefork remain unchanged; no exactrootcause inferred.'}
    write(H/'PROTOCOL.json',proto)
    evidence=[S2/'FREEZE_RECEIPT.json',S2/'SHA256SUMS.txt',S1/'RECEIPT.json',S1/'SHA256SUMS.txt',S2/'run/SF2_FINAL_REPORT.md',ROOT/'sf2_residual_item_autopsy_v1/REPORT.md',ROOT/'overnight_reproducibility_resolution_v1/REPORT.md',ROOT/'sf5_matched_prefix_anneal_v1/SF5_PROTOCOL.json',Path(r'C:\Users\jdman\Desktop\New folder\DEEPSEEK ANALYSIS.txt'),Path(r'C:\Users\jdman\Desktop\New folder\QWEN ANALYSIS.txt')]
    write(H/'PROVENANCE.json',{'reused_byte_identical':reused,'evidence_hashes':{str(x):sha(x) for x in evidence},'locked_files_not_copied_or_parsed':True,'historical_sources_unmodified':True})
    print('MATERIALIZED',H)

if __name__=='__main__': main()
