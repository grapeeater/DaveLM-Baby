from __future__ import annotations
import hashlib,json,sys,re
from pathlib import Path
from collections import Counter
from tokenizers import Tokenizer
ROOT=Path(r'C:\DaveLM-CADAVER'); OUT=ROOT/'language_pilot_0_tinystories_seed8380'; SRC=OUT/'TinyStories-valid.txt'; TOK=Path(r'C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json'); T13=ROOT/'treatment13_learned_mapping_row_localization_seed8380'; TRAINOLD=T13/'treatment13_training_quartet_pool.json'
sys.path.insert(0,str(ROOT)); sys.path.insert(0,r'C:\DaveLM-v0.9')
from treatment13_generate import load_key_value_pools, build_pool
from treatment10_common import load_tokenizer
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def main():
 tok=Tokenizer.from_file(str(TOK)); raw=SRC.read_text(encoding='utf-8'); stories=[x.strip() for x in raw.split('<|endoftext|>') if x.strip()]; norm=[s.replace('\r\n','\n').replace('\r','\n') for s in stories]; uniq={hashlib.sha256(s.encode('utf-8')).hexdigest():s for s in norm}; assert len(uniq)>=10000; selected=[uniq[k] for k in sorted(uniq)[:10000]]; tr,dev=selected[:9000],selected[9000:]
 def recs(xs): return [{'id':i,'text':s,'token_ids':tok.encode(s).ids} for i,s in enumerate(xs)]
 train=recs(tr); val=recs(dev); (OUT/'language_train.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in train),encoding='utf-8'); (OUT/'language_dev.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in val),encoding='utf-8')
 def stats(xs):
  words=sum(len(re.findall(r"\b\w+[']?\w*\b",s['text'])) for s in xs); chars=sum(len(s['text']) for s in xs); toks=sum(len(s['token_ids']) for s in xs); lens=sorted(len(s['token_ids']) for s in xs); pct=lambda p:lens[int(round((len(lens)-1)*p))]; return {'stories':len(xs),'words':words,'characters':chars,'tokens':toks,'tokens_per_word':toks/words,'characters_per_token':chars/toks,'token_length_quantiles':{'p10':pct(.1),'p25':pct(.25),'p50':pct(.5),'p75':pct(.75),'p90':pct(.9),'p99':pct(.99)}}
 filler=__import__('experiments.two_mapping_contextual_binding.run',fromlist=['_identity_pool'])._identity_pool(tok)['filler']; filler_words=[str(x['value']) for x in filler]; keys,vals=load_key_value_pools(); combos_r=[(40,4),(42,5),(44,6),(46,7),(48,8),(50,9),(52,10),(54,11)]; combos_d=[(56,12),(58,13)]; bos=tok.token_to_id('<bos>'); rehearsal=build_pool(tok,keys,vals,filler_words,combos_r,10,'train',bos); devpool=build_pool(tok,keys,vals,filler_words,combos_d,10,'dev',bos)
 def rewrite(pool,axis):
  for qi,q in enumerate(pool):
   q['quartet_id']=f'pilot_{axis}:qt_{qi:06d}'; q['axis']=f'pilot_{axis}'
   for d in q['docs']: d['quartet_id']=q['quartet_id']; d['doc_id']=f"{q['quartet_id']}:{d['member']}"; d['axis']=f'pilot_{axis}'
 rewrite(rehearsal,'rehearsal'); rewrite(devpool,'dev')
 (OUT/'binding_rehearsal.json').write_text(json.dumps({'artifact_type':'pilot_binding_rehearsal','quartets':rehearsal},indent=1),encoding='utf-8'); (OUT/'binding_dev.json').write_text(json.dumps({'artifact_type':'pilot_binding_dev','quartets':devpool},indent=1),encoding='utf-8')
 old=json.loads(TRAINOLD.read_text()); oldarr={tuple(d['full_document_token_ids']) for q in old['quartets'] for d in q['docs']}; newarr=[tuple(d['full_document_token_ids']) for q in rehearsal+devpool for d in q['docs']]; assert len(newarr)==400 and len(set(newarr))==400 and not oldarr.intersection(newarr)
 manifest={'source':{'url':'https://huggingface.co/datasets/roneneldan/TinyStories','revision':'f54c09fd23315a6f9c86f9dc80f725de7d8f9c64','file':'TinyStories-valid.txt','source_sha256':sha(SRC)},'tokenizer':{'path':str(TOK),'sha256':sha(TOK),'vocab_size':tok.get_vocab_size()},'selection':'normalize line endings; sha256 each complete story; sort hashes; first 10000 unique; first 9000 train and remaining 1000 dev','language_train':stats(train),'language_dev':stats(val),'binding_rehearsal':{'quartets':80,'documents':320,'sha256':sha(OUT/'binding_rehearsal.json'),'layout_combos':combos_r},'binding_dev':{'quartets':20,'documents':80,'sha256':sha(OUT/'binding_dev.json'),'layout_combos':combos_d},'prior_t13_training_pool_sha256':sha(TRAINOLD),'sacred_retention_opened':False}
 (OUT/'CORPUS_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
