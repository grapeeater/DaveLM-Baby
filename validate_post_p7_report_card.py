import hashlib,json,re
from pathlib import Path
from tokenizers import Tokenizer

ROOT=Path(r"C:\DaveLM-CADAVER")
OUT=ROOT/"post_p7_language_report_card_v1_seed8380"
TOK=Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")
P7=ROOT/"language_compositional_p7"
TRAIN=ROOT/"language_pilot_0_tinystories_seed8380"/"language_train.jsonl"
DEV=ROOT/"language_pilot_0_tinystories_seed8380"/"language_dev.jsonl"
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def norm(s): return re.sub(r'\s+',' ',s.casefold()).strip()
tok=Tokenizer.from_file(str(TOK))
rows=[json.loads(x) for x in (OUT/'ITEMS.jsonl').read_text(encoding='utf-8').splitlines()]
controlled=[x for x in rows if x['role']=='controlled']; natural=[x for x in rows if x['role']=='generation']
families=json.loads((OUT/'FAMILIES.json').read_text(encoding='utf-8'))['families']
assert len(controlled)==288 and len(natural)==24 and len(families)==36
assert len({x['prompt'] for x in rows})==312
assert all(len(x['candidate_token_ids'])==2 and len(x['candidate_token_ids'][0])==len(x['candidate_token_ids'][1])==3 for x in controlled)
for x in controlled:
 for c,ids in zip(x['candidates'],x['candidate_token_ids']):
  assert tok.encode(c).ids==ids and tok.decode(ids)==c
  assert tok.encode(x['prompt']+c).ids==tok.encode(x['prompt']).ids+ids
for f in families:
 rr=[x for x in controlled if x['family_id']==f['family_id']]
 assert len(rr)==8
 for k in ('assignment','query_index','fact_order','correct_index'):
  assert all(sum(x[k]==v for x in rr)==4 for v in (0,1))
p7=norm((P7/'TRAIN_TEXT.txt').read_text(encoding='utf-8'))
ts=norm(TRAIN.read_text(encoding='utf-8')+'\n'+DEV.read_text(encoding='utf-8'))
assert all(norm(x['prompt']) not in p7 and norm(x['prompt']) not in ts for x in rows)
assert sha(P7/'latest.pt')=='d41ed1186cc945aa05dbd2ba3086fac08ff3b97035c9532e4fa70e78b149a20e'
lengths=[len(tok.encode(x['prompt']).ids)+1+sum(len(i) for i in x['candidate_token_ids'][:1]) for x in controlled]
result={'status':'PASS_PREEXECUTION_ONLY','controlled_items':288,'naturalistic_items':24,'families':36,'tokenizer_sha256':sha(TOK),'p7_checkpoint_sha256':sha(P7/'latest.pt'),'candidate_lengths':[3,3],'unique_prompts':312,'exact_prompt_overlap_count':0,'max_completed_input_tokens':max(lengths),'model_loaded':False,'inference_run':False,'training_run':False,'sacred_exam_accessed':False}
(OUT/'INDEPENDENT_VALIDATION.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n',encoding='utf-8',newline='')
print(json.dumps(result,indent=2))
