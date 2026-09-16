"""Apply S1's preregistered criteria without changing the experiment."""
import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.baby_v010 import selection_s1 as s

def check_finite(obj):
    if isinstance(obj,float):assert math.isfinite(obj),'nonfinite metric'
    elif isinstance(obj,dict):
        for v in obj.values():check_finite(v)
    elif isinstance(obj,list):
        for v in obj:check_finite(v)

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=110001);p.add_argument('--step',type=int,default=800);a=p.parse_args()
    s.verify()
    base=json.loads((s.OUT/f'treatment_{a.seed}/eval_0000.json').read_text())
    control=json.loads((s.OUT/f'control_{a.seed}/eval_{a.step:04d}.json').read_text())
    end=json.loads((s.OUT/f'treatment_{a.seed}/eval_{a.step:04d}.json').read_text())
    for x in (base,control,end):check_finite(x)
    b,c,t=[x['summary'] for x in (base,control,end)]
    regression={}
    regression['language_ce']=end['language_dev_ce']-base['language_dev_ce']>.10
    regression['rest_lock']=b['rest_lock']-t['rest_lock']>.05
    for name,bs in base['frozen']['summaries'].items():
        if name in ('broken_context','broken_order','context_lift_proxy'):continue
        for metric in ('first_top1','free_exact'):
            regression[name+'_'+metric]=bs[metric]-end['frozen']['summaries'][name][metric]>.05
    for name in ('broken_context','broken_order'):
        regression[name+'_exact']=end['frozen']['summaries'][name]['free_exact']>.05
    for name,summary in end['isolation']['summaries'].items():
        if name.startswith('value_absent'):regression[name+'_exact']=summary['free_exact']>.05
    differences=[x-y for x,y in zip(t['body_accuracies'],c['body_accuracies'])]
    rng=random.Random(110300)
    boot=sorted(statistics.mean(rng.choices(differences,k=len(differences))) for _ in range(10000))
    query='query_swap_same_surface_novel'
    criteria={
        'body_accuracy_ge_070':t['body_accuracy']>=.70,
        'gain_parent_ge_020':t['body_accuracy']-b['body_accuracy']>=.20,
        'gain_control_ge_015':t['body_accuracy']-c['body_accuracy']>=.15,
        'all_K_above_chance_plus_015':all(t['per_K'][str(k)]>=1/k+.15 for k in (2,3,4)),
        'query_swap_gain_ge_015':end['isolation']['summaries'][query]['first_top1']-base['isolation']['summaries'][query]['first_top1']>=.15,
        'same_surface_exact_gain_ge_015':end['frozen']['summaries']['same_surface_novel']['free_exact']-base['frozen']['summaries']['same_surface_novel']['free_exact']>=.15,
        'query_logit_effect_gain_ge_050':t['query_logit_effect']-b['query_logit_effect']>=.50,
        'bootstrap_control_gain_lower_positive':boot[249]>0,
    }
    futility=a.step==400 and t['body_accuracy']-b['body_accuracy']<.05 and c['body_accuracy']-b['body_accuracy']<.05 and t['mean_margin']-b['mean_margin']<.25
    hard_stop=end['language_dev_ce']-base['language_dev_ce']>.20
    for arm in ('control','treatment'):
        for path in (s.OUT/f'{arm}_{a.seed}').glob('RECEIPT_*.json'):
            hard_stop |= json.loads(path.read_text())['reason'].startswith('hard_stop')
    status='HARD_STOP' if hard_stop else 'REGRESSION' if any(regression.values()) else 'SUCCESS' if all(criteria.values()) else 'PARTIAL' if t['body_accuracy']-b['body_accuracy']>=.10 and t['body_accuracy']-c['body_accuracy']>=.05 else 'FAILURE'
    result={'seed':a.seed,'step':a.step,'status':status,'futility':futility,'success_criteria':criteria,'regression_flags':regression,'baseline':b,'control':c,'treatment':t,'language_ce':{k:v['language_dev_ce'] for k,v in [('baseline',base),('control',control),('treatment',end)]},'paired_body_bootstrap_95':[boot[249],boot[9749]],'original_frozen_summaries':{k:v['frozen']['summaries'] for k,v in [('baseline',base),('control',control),('treatment',end)]},'isolation_summaries':{k:v['isolation']['summaries'] for k,v in [('baseline',base),('control',control),('treatment',end)]},'manifest_sha256':s.digest(s.OUT/'MANIFEST.json'),'protected_material_opened':False}
    dest=s.OUT/f'ADJUDICATION_{a.seed}_{a.step}.json'
    if dest.exists():raise RuntimeError('refuse overwrite')
    s.write(dest,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('baseline','control','treatment','original_frozen_summaries','isolation_summaries')}),flush=True)

if __name__=='__main__':main()
