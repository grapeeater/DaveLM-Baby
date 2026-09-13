"""Essential mechanical checks only. No checkpoint loading or optimizer updates."""
import ast, json, inspect, os, sys
from collections import Counter
from pathlib import Path
import CONTROLLER as C

def main():
    p=C.verify(sealed=False)
    schedule,lrs,train,idx,pool,kl,old=C.load_inputs()
    tok=C.Tokenizer.from_file(p['tokenizer'])
    for r in train:
        assert tok.encode(r['prompt']).ids==r['prompt_token_ids']
        for text,ids in zip(r['candidates'],r['candidate_token_ids']):
            assert tok.encode(text).ids==ids and len(ids)==4
            assert tok.encode(r['prompt']+text).ids==r['prompt_token_ids']+ids
            assert tok.decode(ids)==text
    assert len(set(r['family_id'] for r in train))==4
    assert len(set(r['pair_id'] for r in train))==8
    counts=Counter(r['candidates'][r['correct_index']] for r in train)
    assert set(counts.values())=={4}
    good={'correct':16,'exact':16,'reversals':8,'families':4}
    checks={'binding':{n:{'gate':True} for n in ['pilot0','pilot1']},'language':{'loss':3.5}}
    d3={'mean_combined_name_probability':.01}
    assert C.gates(good,checks,d3,3.39,200)['endpoint_pass']
    for key in good:
        bad=dict(good); bad[key]-=1
        assert not C.gates(bad,checks,d3,3.39,200)['endpoint_pass']
    assert not C.gates(good,checks,{'mean_combined_name_probability':.01000001},3.39,200)['continue']
    for n in checks['binding']:
        checks['binding'][n]['gate']=False
        assert not C.gates(good,checks,d3,3.39,200)['continue']
        checks['binding'][n]['gate']=True
    checks['language']['loss']=3.6400001
    assert not C.gates(good,checks,d3,3.39,200)['continue']
    checks['language']['loss']=3.5
    assert not C.gates({'correct':9,'exact':9,'reversals':0,'families':0},checks,d3,3.39,100)['continue']
    assert C.gates({'correct':10,'exact':9,'reversals':0,'families':0},checks,d3,3.39,100)['continue']
    src=inspect.getsource(C)
    ast.parse(src)
    assert 'set_scope_block3(' not in src and 'E.main(' not in src
    assert all(x not in src for x in ['HELDOUT.json','ALTERNATE.json','COPY.json','COMPETING.json','NotImplementedError'])
    # All state equality checks concern one run's frozen parameter/optimizer invariants.
    assert 'snapshot_at_100' not in src and 'historical_checkpoint' not in src
    assert "lr=lrs[u['update']-1][a.arm]" in src
    assert 'ce+E.LAMBDA_KL*kl_val' in src and 'E.set_scope(m,' in src
    assert 'rt.binding_loss(m,' in src and 'E.compute_kl(m,' in src
    assert 'E.panel(' in src and 'E.checks(' in src and 'E.measure_d3(' in src
    scopesource=inspect.getsource(C.E.set_scope)
    assert 'range(4)' in scopesource
    assert C.E.LAMBDA_KL==1 and C.E.KL_POS_PER_UPDATE==160
    pin=C.rt.pinned_binding(C.H)
    assert pin.LAM==1.0536573711078283
    for fp in C.H.glob('*.py'): ast.parse(fp.read_text(encoding='utf-8'))
    result={'status':'PASS','checkpoint_loaded':False,'optimizer_created':False,'updates':0,
      'counts':{'updates':200,'english':180,'binding':20,'training_records':16,'families':4,'pairs':8,'batch':32,'kl_positions_per_update':160,'kl_pool':kl['count']},
      'checks':{k:'PASS' for k in ['SF2_source_identity','Pilot1_identity','tokenizer_identity_and_boundaries','literal_schedule_and_balancing','masking_alignment','LR_endpoints_and_binding_LR','single_arm_dependent_training_value_is_LR','same_seed_pairing','pinned_scope_and_objectives','frozen_gate_negative_tests','no_replay_or_fork_gate','locked_panels_not_referenced','external_source_identity','runtime']},
      'runtime':{'python':C.platform.python_version(),'torch':C.torch.__version__,'tokenizers':C.tokenizers.__version__,'gpu':C.torch.cuda.get_device_name(0),
                 'executable':sys.executable,'hip':C.torch.version.hip,'default_dtype':str(C.torch.get_default_dtype()),
                 'relevant_environment':{k:os.environ.get(k) for k in ['TORCH_BLAS_PREFER_HIPBLASLT','CUBLAS_WORKSPACE_CONFIG','HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','HSA_OVERRIDE_GFX_VERSION']}},
      'baseline_policy':'Real native Pilot1 output controls run independently before first optimizer creation in every run; no trajectory equality gates.',
      'source_hashes':{str(m.__file__):C.sha(m.__file__) for m in [C,C.E,C.rt]},
      'scope_function_sha256':C.sha(C.H/'sources/PINNED_MASKING.py')}
    C.rt.atomic_json(result,C.H/'PREFLIGHT.json')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
