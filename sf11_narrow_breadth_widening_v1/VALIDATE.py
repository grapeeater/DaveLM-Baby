"""Independent final-text semantic/parser and mechanical preflight. No model/optimizer."""
import re,json,inspect,subprocess,sys,copy,ast
from collections import Counter,defaultdict
from pathlib import Path
import CONTROLLER as C
def norm(s): return ' '.join(s.casefold().split())

def parse(r):
    context,cue=r['prompt'].split('\n')
    parts=context.split('. ')
    actor=verb=obj=card=None
    for part in parts:
        part=part.removesuffix('.')
        if part.startswith('A card says '):
            assert card is None; card=part.removeprefix('A card says ')
            assert card in ['Alex','Owen','Mia','Nora']
        else:
            assert actor is None
            if ' was ' in part and ' by ' in part:
                obj,tail=part.split(' was '); verb,actor=tail.split(' by ')
            else:
                actor,verb,obj=part.split(' ',2)
            assert actor in ['Alex','Owen','Mia','Nora'] and obj.startswith('the ')
    assert actor is not None and ' ' not in verb
    if cue=='Copy the name on the card:':
        assert card is not None and card!=actor; answer=card
    else:
        if cue.startswith('Who ') and cue.endswith('?'): relation=cue[4:-1]
        else:
            assert cue.startswith('The person who ') and cue.endswith(' was')
            relation=cue[len('The person who '):-4]
        assert relation==verb+' '+obj
        answer=actor
    assert ' '+answer+'.'==r['candidates'][r['correct_index']]
    if 'actor' in r:
        stored_object=('the '+r['object']) if r['id'].startswith('TRAIN:') else r['object']
        assert (actor,verb,obj)==(r['actor'],r['predicate'],stored_object)
    return actor,verb,obj,card

def validate():
    p=C.verify(sealed=False)
    schedule,items,idx,ret,ds,do,pool,kl,_=C.load_inputs()
    tok=C.Tokenizer.from_file(p['tokenizer'])
    allrows=items+ds+do
    assert len(set(norm(r['prompt']) for r in allrows))==80
    for r in allrows:
        parse(r)
        assert tok.encode(r['prompt']).ids==r['prompt_token_ids']
        assert tok.decode(r['prompt_token_ids'])==r['prompt']
        for c,ids in zip(r['candidates'],r['candidate_token_ids']):
            assert len(ids)==4 and tok.encode(c).ids==ids and tok.decode(ids)==c
            assert tok.encode(r['prompt']+c).ids==r['prompt_token_ids']+ids
            assert 1+len(r['prompt_token_ids'])+len(ids)+1<=256
    # Pair completeness and answer balance independently checked from final text.
    for rows in [items,ds,do]:
        pairs=defaultdict(list)
        for r in rows: pairs[r['pair_id']].append(r)
        assert all(len(v)==2 and {r['correct_index'] for r in v}=={0,1} for v in pairs.values())
        for v in pairs.values():
            a,b=map(parse,v); assert a[0]!=b[0] and a[1:3]==b[1:3]
        counts=Counter(r['candidates'][r['correct_index']] for r in rows)
        assert len(counts)==4 and len(set(counts.values()))==1
    for rows in [ds,do]:
        assert len(rows)==16 and set(Counter(r['subgroup'] for r in rows).values())=={4}
        for sg in {r['subgroup'] for r in rows}:
            assert set(Counter(r['candidates'][r['correct_index']] for r in rows if r['subgroup']==sg).values())=={1}
    old=C.read(C.H.parent/'sf10_reduced_density_widening_v1/TRAIN.json'); oldidx={r['id']:r for r in old}
    assert all(r==oldidx[r['id']] for r in items)
    oldsched=C.read(C.H.parent/'sf10_reduced_density_widening_v1/SCHEDULE.json')
    exposure=Counter()
    for u,o in zip(schedule,oldsched):
        assert u['pad']==o['pad']
        if u['kind']=='binding': assert u==o; continue
        assert [i for i in u['ids'] if i.startswith('TRAIN:')]==[i for i in o['ids'] if i.startswith('TRAIN:')]
        exposure.update(u['ids'])
        counts=Counter(idx[i]['candidates'][idx[i]['correct_index']] for i in u['ids'])
        assert set(counts.values())=={9}
        assert Counter(idx[i]['subgroup'] for i in u['ids'])==Counter(oldidx[i]['subgroup'] for i in o['ids'])
        x,y=C.E.pad_batch([idx[i] for i in u['ids']],u['pad'])
        for n,rid in enumerate(u['ids']):
            r=idx[rid]; prefix=[2]+r['prompt_token_ids']; ids=r['candidate_token_ids'][r['correct_index']]
            assert x[n,:len(prefix)+4].tolist()==prefix+ids
            assert y[n,len(prefix)-1:len(prefix)+4].tolist()==ids+[3]
            assert (y[n,:len(prefix)-1]==-100).all() and (y[n,len(prefix)+4:]==-100).all()
    assert sum(exposure.values())==6480
    assert sum(v for k,v in exposure.items() if k.startswith('TRAIN:'))==720
    # New DEV lexical content disjoint from SF9/SF10 training + prior development.
    oldeval=C.read(C.H.parent/'sf10_reduced_density_widening_v1/DEV_SURFACE.json')+C.read(C.H.parent/'sf10_reduced_density_widening_v1/DEV_ORDER.json')
    historicaltext='\n'.join(r['prompt'] for r in old+oldeval)
    for r in ds+do:
        assert r['predicate'] not in historicaltext and r['object'] not in historicaltext
    # No locked panel opening: training is a verified subset of previously validated SF9 material.
    # SF9's frozen generator documents SF1 locked facts use found/carried and these four objects only.
    locked_objects=['small drum','wooden boat','soft scarf','round plate']
    for r in [r for r in items if r['id'].startswith('SF9:')]+ds+do:
        assert all(o not in r['prompt'] for o in locked_objects)
    # Full decoded-text overlap with the authorized TinyStories corpus; no matching JSON serialization.
    corp=Path(C.read(C.H/'KL_POOL_MANIFEST.json')['rule']['source'])
    texts=[]
    for line in corp.read_text(encoding='utf-8').splitlines():
        v=json.loads(line)
        if 'text' in v: texts.append(v['text'])
        else: texts.append(tok.decode(v['token_ids']))
    normalized='\n'.join(norm(t) for t in texts)
    overlap=[r['id'] for r in allrows if norm(r['prompt']) in normalized]
    assert not overlap,overlap
    tests={}
    def rejected(label,r):
        try: parse(r)
        except (AssertionError,ValueError): tests[label]='REJECTED'; return
        raise AssertionError(label+' not rejected')
    bad=copy.deepcopy(ds[0]); bad['correct_index']=1-bad['correct_index']; rejected('wrong_key',bad)
    bad=copy.deepcopy(ds[0]); bad['predicate']='invented'; rejected('wrong_predicate',bad)
    bad=copy.deepcopy(ds[0]); bad['prompt']=bad['prompt'].replace('Who','What') if 'Who' in bad['prompt'] else bad['prompt'].replace('person who','person where'); rejected('wrong_cue',bad)
    bad=copy.deepcopy(ds[0]); bad['prompt']=bad['prompt'].replace('the paper kite','the other object',1); rejected('mismatched_query_object',bad)
    bad=copy.deepcopy(do[0]); bad['correct_index']=1-bad['correct_index']; rejected('broken_assignment_reversal',bad)
    assert C.MARGIN_M==1 and C.LAMBDA_MARGIN==.25 and C.E.LAMBDA_KL==1 and C.E.KL_POS_PER_UPDATE==160
    assert C.rt.pinned_binding(C.H).LAM==1.0536573711078283
    src=inspect.getsource(C); prior=(C.H.parent/'sf10_reduced_density_widening_v1/CONTROLLER.py').read_text()
    assert src[src.index('    for u in schedule[completed:]:'):]==prior[prior.index('    for u in schedule[completed:]:'):]
    for func in ['margin_gap','margin_hinge_term','retention_acq','run_eval','append_metric']:
        a=ast.parse(src); b=ast.parse(prior)
        assert ast.dump(next(n for n in a.body if isinstance(n,ast.FunctionDef) and n.name==func))==ast.dump(next(n for n in b.body if isinstance(n,ast.FunctionDef) and n.name==func))
    assert 'set_scope_block3(' not in src and 'E.main(' not in src
    assert not any(s in src for s in ['HELDOUT.json','ALTERNATE.json','COPY.json','COMPETING.json'])
    goodret={'correct':16,'exact':16,'reversals':8,'families':4}
    goodds={'correct':15,'exact':13,'subgroups':{s:{'exact':3} for s in ['cloze','active_qa','passive_cloze','passive_qa']}}
    goodds['subgroups']['cloze']['exact']=4
    gooddo={'exact':14,'subgroups':{s:{'exact':v} for s,v in zip(['order0:fact','order0:copy','order1:fact','order1:copy'],[4,3,3,4])}}
    ch={'binding':{s:{'gate':True} for s in ['pilot0','pilot1']},'language':{'loss':3.6}}
    d3={'mean_combined_name_probability':.01}
    assert C.gates(goodret,goodds,gooddo,ch,d3,3.4,200)['endpoint_pass']
    bad=copy.deepcopy(gooddo); bad['subgroups']['order1:fact']['exact']=2
    assert not C.gates(goodret,goodds,bad,ch,d3,3.4,200)['endpoint_pass']
    assert not C.gates(goodret,goodds,gooddo,ch,{'mean_combined_name_probability':.01001},3.4,200)['continue']
    bad=copy.deepcopy(ch);bad['language']['loss']=3.65001
    assert not C.gates(goodret,goodds,gooddo,bad,d3,3.4,200)['continue']
    for poolname in ['pilot0','pilot1']:
        bad=copy.deepcopy(ch);bad['binding'][poolname]['gate']=False
        assert not C.gates(goodret,goodds,gooddo,bad,d3,3.4,200)['continue']
    assert C.rt.binding_gate({'answer_exact':76,'both_distinct':76,'slot_collapse':0})
    assert not C.rt.binding_gate({'answer_exact':75,'both_distinct':76,'slot_collapse':0})
    assert not C.rt.binding_gate({'answer_exact':76,'both_distinct':75,'slot_collapse':0})
    assert not C.rt.binding_gate({'answer_exact':80,'both_distinct':80,'slot_collapse':1})
    for k in goodret:
        bad=dict(goodret); bad[k]-=1; assert not C.gates(bad,goodds,gooddo,ch,d3,3.4,200)['continue']
    report={'status':'PASS','checkpoint_loaded':False,'optimizer_created':False,'updates':0,'counts':{'train':48,'new':32,'retention':16,'dev_surface':16,'dev_order':16,'batch':36,'presentations':6480},
      'semantic_validation':'all80 final rendered prompts independently parsed; answer keys/reversals validated','negative_tests':tests,
      'overlap':{'train_dev':0,'dev_historical_sf9_sf10':0,'TinyStories_full_prompt_substrings':overlap,'TinyStories_source_sha256':C.sha(corp),'locked_fence':'verified SF9-subset provenance + disjoint known original object/predicate domain; locked panels not opened; no semantic/near-duplicate independence claim'},
      'mechanical':'all200schedule records; name/category/batchbalance; exactSF10trainingloop; objectives/scopes; masking; tokenizer; gates; binding; runtime PASS',
      'runtime':{'python':C.platform.python_version(),'torch':C.torch.__version__,'tokenizers':C.tokenizers.__version__,'gpu':C.torch.cuda.get_device_name(0)},
      'parent_hashes':{str(r['seed']):r['parent_checkpoint_sha256'] for r in p['runs']}}
    smoke=subprocess.run([sys.executable,'-B',str(C.H/'REPORTER_SMOKE_TEST.py')],capture_output=True,text=True)
    assert smoke.returncode==0 and 'ALL_SMOKE_TESTS_PASS' in smoke.stdout,smoke.stdout+smoke.stderr
    report['reporter_smoke']=smoke.stdout
    C.rt.atomic_json(report,C.H/'PREFLIGHT.json')
    print(json.dumps(report,indent=2))

if __name__=='__main__': validate()
