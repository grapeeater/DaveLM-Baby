import json, math
from pathlib import Path
ROOT=Path(r'C:\\DaveLM-CADAVER')
srcs=[('baseline',ROOT/'generation_pathology_hr123.json'),('constraints',ROOT/'generation_pathology_constraints.json')]
allout={}
def one(q):
 rows=q.get('rows',[]); ids=[int(r['token_id']) for r in rows]; toks=[r.get('token','') for r in rows]; n=len(ids)
 def onset_run(k):
  for i in range(k-1,n):
   if len(set(ids[i-k+1:i+1]))==1:return i+1
  return None
 def first_ng(k):
  seen={}
  for i in range(k-1,n):
   g=tuple(ids[i-k+1:i+1])
   if g in seen:return i+1
   seen[g]=i
  return None
 longest=0; cur=0; prev=None
 for x in ids:
  cur=cur+1 if x==prev else 1; longest=max(longest,cur); prev=x
 punct=set(['.','!','?',',',';',' :','\n'])
 p_on=None; p_run=0
 for i,t in enumerate(toks):
  if t.strip() in punct:p_run+=1
  else:p_run=0
  if p_run>=3 and p_on is None:p_on=i+1
 onset=next((x for x in [onset_run(2),onset_run(3),p_on,first_ng(2),first_ng(3)] if x is not None),None)
 if onset is None: typ='none'
 elif p_on is not None and p_on==onset:typ='punctuation'
 elif onset_run(3) is not None and onset_run(3)==onset:typ='token-run'
 elif onset_run(2) is not None and onset_run(2)==onset:typ='adjacent-repeat'
 elif first_ng(3) is not None and first_ng(3)==onset:typ='trigram-repeat'
 else:typ='ngram-repeat'
 def mean(key, sel=None):
  vals=[float(r[key]) for i,r in enumerate(rows) if sel is None or sel(i+1)]
  return sum(vals)/len(vals) if vals else None
 return {'prompt':q.get('prompt'),'length':q.get('length'),'immediate_eos':q.get('immediate_eos'),
  'first_adjacent_repeat_step':onset_run(2),'first_triple_repeat_step':onset_run(3),
  'first_repeated_bigram_step':first_ng(2),'first_repeated_trigram_step':first_ng(3),
  'first_punctuation_run_step':p_on,'max_same_token_run':longest,'degeneration_type':typ,
  'entropy_first5':mean('entropy',lambda i:i<=5),'entropy_last5':mean('entropy',lambda i:i>max(0,n-5)),
  'top1_prob_first5':mean('top1_prob',lambda i:i<=5),'top1_prob_last5':mean('top1_prob',lambda i:i>max(0,n-5)),
  'top1_margin_first5':mean('top1_margin',lambda i:i<=5),'top1_margin_last5':mean('top1_margin',lambda i:i>max(0,n-5)),
  'first8_tokens':toks[:8],'decoded':q.get('decoded')}
for label,p in srcs:
 d=json.loads(p.read_text(encoding='utf-8')); allout[label]={}
 for model,modes in d.items():
  allout[label][model]={}
  for mode,qs in modes.items():
   ms=[one(q) for q in qs]; on=[x['first_triple_repeat_step'] for x in ms if x['first_triple_repeat_step'] is not None]
   allout[label][model][mode]={'items':ms,'n':len(ms),
    'immediate_eos_rate':sum(bool(x['immediate_eos']) for x in ms)/len(ms),
    'mean_length':sum(x['length'] for x in ms)/len(ms),
    'triple_repeat_rate':sum(x['first_triple_repeat_step'] is not None for x in ms)/len(ms),
    'mean_first_triple_repeat_step':sum(on)/len(on) if on else None,
    'degeneration_type_counts':{t:sum(x['degeneration_type']==t for x in ms) for t in sorted(set(x['degeneration_type'] for x in ms))},
    'mean_entropy_first5':sum(x['entropy_first5'] for x in ms)/len(ms),
    'mean_entropy_last5':sum(x['entropy_last5'] for x in ms)/len(ms),
    'mean_top1_prob_first5':sum(x['top1_prob_first5'] for x in ms)/len(ms),
    'mean_top1_prob_last5':sum(x['top1_prob_last5'] for x in ms)/len(ms),
    'mean_margin_first5':sum(x['top1_margin_first5'] for x in ms)/len(ms),
    'mean_margin_last5':sum(x['top1_margin_last5'] for x in ms)/len(ms)}
(ROOT/'generation_pathology_analysis.json').write_text(json.dumps(allout,indent=2,ensure_ascii=False),encoding='utf-8')
for label,models in allout.items():
 print('===',label,'===')
 for model,modes in models.items():
  for mode,s in modes.items(): print(model,mode,'len',s['mean_length'],'triple',s['triple_repeat_rate'],'onset',s['mean_first_triple_repeat_step'],'types',s['degeneration_type_counts'],'H',s['mean_entropy_first5'],s['mean_entropy_last5'],'p',s['mean_top1_prob_first5'],s['mean_top1_prob_last5'])
