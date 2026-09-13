import json,glob,os
for f in glob.glob(r'C:\DaveLM-CADAVER\language_*\RESULTS.json'):
 try:j=json.load(open(f,encoding='utf8'))
 except:continue
 print('\n',os.path.dirname(f).split('\\')[-1])
 for k,v in j.items():
  if k in ('parent_sha256','updates','seed','lr','tiny_train_sha256','tokenizer_sha256','language_perplexity','binding','post','pre'): continue
  if isinstance(v,(str,int,float,bool)): print(k,v)
  elif k=='generations':
   for x in v[:3]: print('GEN',x)
