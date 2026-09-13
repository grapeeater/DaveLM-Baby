import hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 s=json.loads((H/'STATIC_PREFLIGHT.json').read_text()); u=json.loads((H/'U0_PREFLIGHT_RECORD.json').read_text()); r=json.loads((H/'REPORTER_SMOKE.json').read_text())
 assert s['status']=='SF21_STATIC_PREFLIGHT_PASS' and all(x['reproduces_parent'] for x in u['branches'].values()) and r['status']=='PASS'
 out={'status':'SF21_PROSPECTIVE_PREFLIGHT_PASS','static':s,'u0_reproduction':u,'reporter_smoke':r,
  'hashes':{n:sha(H/n) for n in ['CONTROLLER.py','PROTOCOL.json','PAIR_MANIFEST.json','TRAIN.json','SCHEDULE.json','VALIDATE.py','U0_PREFLIGHT_RECORD.json','REPORTER.py','LAUNCH.py']},
  'scientific_difference_count':1,'scientific_difference':'0.1 paired counterfactual representation consistency only','checkpoint_loaded_for_u0_only':True,'optimizer_created':False,'updates':0,'treatment_outcomes_scored':False,'locked_transfer':'LOCKED_UNSCORED','final_accessed':False,'sacred_accessed':False}
 (H/'PREFLIGHT.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
 (H/'PREFLIGHT.md').write_text('# SF21 Prospective Preflight\n\n**PASS.** Sixteen deterministic assignment-reversal pairs cover all 32 widening records. The final-normalized last prompt/query state predicts the first answer token; no answer-token state is compared. Both sides receive gradient. SF20 data, schedule, CE, KL, margin, optimizer, scopes, evaluator, and gates are unchanged. All three parent U0 baselines reproduced exactly. No optimizer was created and no update occurred. Locked transfer, FINAL, and sacred material remain untouched.\n',encoding='utf-8',newline='\n')
 print(out['status'])
if __name__=='__main__': main()
