"""Synthetic end-to-end report rendering and classification; no checkpoint/optimizer/inference."""
import copy,json,tempfile
from pathlib import Path
import REPORTER as R
def put(p,x): p.write_text(json.dumps(x),encoding='utf-8')
def main():
    with tempfile.TemporaryDirectory() as tmp:
        h=Path(tmp); original=R.C.H; R.C.H=h
        put(h/'FREEZE_RECEIPT.json',{}); (h/'SHA256SUMS.txt').write_text('synthetic')
        sg={k:{'exact':4,'correct':4,'n':4} for k in ['cloze','active_qa','passive_cloze','passive_qa']}
        og={k:{'exact':4,'correct':4,'n':4} for k in ['order0:fact','order0:copy','order1:fact','order1:copy']}
        ret={'correct':16,'exact':16,'reversals':8,'families':4}
        surf={**ret,'subgroups':sg}; order={**ret,'subgroups':og}
        checks={'language':{'loss':3.6,'perplexity':36.6},'binding':{k:{'gate':True,'summary':{'answer_exact':80,'both_distinct':80,'slot_collapse':0}} for k in ['pilot0','pilot1']}}
        gates={'retention_acquisition':True,'language':True,'d3':True,'binding':{'pilot0':True,'pilot1':True},'endpoint_pass':True}
        results={}; runs=[]
        for seed in [1,2,3]:
            d=h/f'run{seed}';d.mkdir()
            for u in [0,100,200]:
                for label,v in [('train16',ret),('devsurface',surf),('devorder',order)]: put(d/f'update{u}_{label}_RESULT.json',v)
                put(d/f'update{u}_checks.json',checks);put(d/f'd3_update{u}_summary.json',{'mean_combined_name_probability':.007})
                if u: put(d/f'update{u}_GATES.json',gates)
            t=R.load_trajectory(d); assert t['0']['gates'] is None
            st={'completed':200,'status':'ACQUISITION_SUCCESS','checkpoint':'synthetic.pt','checkpoint_sha256':'synthetic','gates':copy.deepcopy(gates)}
            results[str(seed)]={'seed':seed,'status':st,'trajectory':t};runs.append({'seed':seed,'parent_checkpoint':'parent.pt','parent_checkpoint_sha256':'fake'})
        assert R.classify(results)==('BREADTH_HYPOTHESIS_SUPPORTED',3)
        text=R.render(results,{'runs':runs},'BREADTH_HYPOTHESIS_SUPPORTED',3)
        assert text.endswith('evaluation.\n') and '## All preregistered trajectories' in text
        for x in results.values():
            x['status'].update(completed=100,status='STOP_REGRESSION');x['status']['gates'].update(endpoint_pass=False,d3=False)
            x['trajectory'].pop('200');x['trajectory']['100']['d3']['mean_combined_name_probability']=.019
        assert R.classify(results)==('BREADTH_HYPOTHESIS_SUBSTANTIALLY_WEAKENED',0)
        text=R.render(results,{'runs':runs},*R.classify(results)); assert 'Actual English presentations: 3240' in text
        results['1']['status']['gates']['language']=False
        assert R.classify(results)[0]=='RETENTION_REGRESSION'
        results['1']['status']['gates']['language']=True
        for x in results.values():
            x['status'].update(completed=200,status='ACQUISITION_FAIL');x['status']['gates']['d3']=True
            x['trajectory']['200']=copy.deepcopy(x['trajectory']['100']);x['trajectory']['200']['d3']['mean_combined_name_probability']=.007
        assert R.classify(results)[0]=='NARROW_CURRICULUM_INSUFFICIENT'
        assert R.classify({})[0]=='MECHANICAL_INCOMPLETE'
        R.C.H=original
    print('ALL_SMOKE_TESTS_PASS: missingu0gates, partial/full trajectories, full report rendering, classification priority, no model operations')
if __name__=='__main__': main()
