from __future__ import annotations
import hashlib,json,math,random,sys,time,statistics
from collections import defaultdict,Counter
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
ROOT=Path(r'C:\DaveLM-CADAVER'); PILOT=ROOT/'language_pilot_1_early_block_protection_seed8380'; PILOT0=ROOT/'language_pilot_0_tinystories_seed8380'; GRAD=ROOT/'treatment13_orthogonal_shared_unbounded_seed8380'; CK=GRAD/'checkpoints/orthogonal_shared_unbounded/seed_8380/latest.pt'; TOK=Path(r'C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json'); SRC=ROOT/'treatment13_learned_mapping_row_localization_seed8380'; OUT=PILOT/'pilot_run'; sys.path.insert(0,str(ROOT)); sys.path.insert(0,r'C:\DaveLM-v0.9')
from treatment13_model import Treatment13Model
from v0_2.tokenizer_tools import encode_documents
ROW_VALUE_OFFSET=4; SEED=8380; LAM=1.0536573711078283
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def basis(u):
 un=u/torch.linalg.vector_norm(u); p=int(torch.argmax(torch.abs(un))); e=torch.zeros_like(un); e[p]=1. if un[p]>=0 else -1.; w=un-e; H=torch.eye(un.numel(),device=u.device,dtype=u.dtype)-2*torch.outer(w,w)/torch.dot(w,w); return H[:,[i for i in range(un.numel()) if i!=p]]
class OrthoLocalizer(nn.Module):
 def __init__(self,u,q,bs,ba): super().__init__(); self.u=nn.Parameter(u); self.q=nn.Parameter(q); self.bs=nn.Parameter(bs); self.ba=nn.Parameter(ba)
 def forward(self,h):
  v=basis(self.u)@self.q; S=h@self.u+self.bs; R=h@v+self.ba; return torch.stack((S+R,S-R),-1)
def load_model(dev):
 raw=torch.load(CK,map_location=dev,weights_only=False); m=Treatment13Model().to(dev); sd=dict(raw['model_state_dict']); u=sd.pop('localizer.u'); q=sd.pop('localizer.q'); bs=sd.pop('localizer.bs'); ba=sd.pop('localizer.ba'); sd.pop('localizer.scorer.weight',None); sd.pop('localizer.scorer.bias',None); miss,un=m.load_state_dict(sd,strict=False); assert set(miss)=={'localizer.scorer.weight','localizer.scorer.bias'} and not un; m.localizer=OrthoLocalizer(u,q,bs,ba).to(dev); return m
def rowpos(d): return (int(d['query_key_clause_pos']),int(d['other_key_pos'])) if int(d['query_slot'])==0 else (int(d['other_key_pos']),int(d['query_key_clause_pos']))
def loc_loss(att,docs):
 vals=[]
 for i,d in enumerate(docs):
  s0,s1=rowpos(d); x=att[i].clamp_min(1e-12); vals.append(-torch.maximum(x[s0-1,0].log()+x[s1-1,1].log(),x[s1-1,0].log()+x[s0-1,1].log()))
 return torch.stack(vals).mean()
def lang_eval(m,stream,dev,seed):
 g=torch.Generator().manual_seed(seed); vals=[]; m.eval();
 with torch.no_grad():
  for _ in range(20):
   starts=torch.randint(0,len(stream)-257,size=(64,),generator=g); offs=torch.arange(256); x=stream[starts[:,None]+offs[None,:]].to(dev); y=stream[starts[:,None]+offs[None,:]+1].to(dev); vals.append(float(F.cross_entropy(m.base_model(x).reshape(-1,1024),y.reshape(-1))))
 return sum(vals)/len(vals)
def generate(m,tok,prompt):
 bos=tok.token_to_id('<bos>'); eos=tok.token_to_id('<eos>'); ids=tok.encode(prompt).ids; g=[bos]+ids; m.eval()
 with torch.no_grad():
  for _ in range(32):
   z=torch.tensor([g[-256:]],device=next(m.parameters()).device); n=int(m.base_model(z)[:,-1,:].argmax()); g.append(n)
   if n==eos: break
 return {'prompt':prompt,'prompt_ids':ids,'generated_ids':g,'raw_decoded_output':tok.decode(g,skip_special_tokens=True),'stopped_on_eos':g[-1]==eos}
def binding_eval(m,docs,dev):
 rows=[]; m.eval()
 with torch.no_grad():
  for st in range(0,len(docs),32):
   ch=docs[st:st+32]; x=torch.tensor([d['full_document_token_ids'] for d in ch],device=dev); q=torch.tensor([d['qdp'] for d in ch],device=dev); a=torch.tensor([d['answer_causal_position'] for d in ch],device=dev); out,ex=m(x,q,a); ix=torch.arange(len(ch),device=dev)
   for i,d in enumerate(ch):
    s0,s1=rowpos(d); at=ex['localization_attention'][i]; p0=int(at[:,0].argmax())+1;p1=int(at[:,1].argmax())+1; T={s0,s1}; c='BOTH_DISTINCT' if p0!=p1 and {p0,p1}==T else 'SLOT_COLLAPSE' if p0==p1 else 'EXACTLY_ONE' if sum(z in T for z in (p0,p1))==1 else 'NEITHER'; v=out[i,a[i]]; t=int(d['target_value_token']); di=int(d['distractor_value_token']); sr=qr=None
    if c=='BOTH_DISTINCT':
     sm={0:s0 if p0==s0 else s1,1:s0 if p1==s0 else s1}; sel=int(ex['row_weights'][i].argmax()); sr=sm[sel] in T; qr=sm[sel]==(s0 if int(d['query_slot'])==0 else s1)
    rows.append({'doc_id':d['doc_id'],'quartet_id':d['quartet_id'],'member':d['member'],'query_slot':int(d['query_slot']),'orientation':int(d['orientation']),'layout_combo':d['layout_combo'],'answer_correct':bool(int(v.argmax())==t),'predicted':int(v.argmax()),'pos0':p0,'pos1':p1,'true0':s0,'true1':s1,'localization_category':c,'selected_row_correct':sr,'selected_query_row_correct':qr,'target_probability':float(v.softmax(-1)[t]),'distractor_probability':float(v.softmax(-1)[di])})
 def sm(v): return {'n':len(v),'answer_exact':sum(r['answer_correct'] for r in v),'answer_accuracy':sum(r['answer_correct'] for r in v)/len(v),'BD':sum(r['localization_category']=='BOTH_DISTINCT' for r in v),'EO':sum(r['localization_category']=='EXACTLY_ONE' for r in v),'collapse':sum(r['localization_category']=='SLOT_COLLAPSE' for r in v)}
 byq=defaultdict(list); bycat=defaultdict(list); bylay=defaultdict(list)
 for r in rows: byq[r['quartet_id']].append(r); bycat[r['localization_category']].append(r); bylay[r['layout_combo']].append(r)
 rev=bc=0
 for q in byq.values():
  bm={r['member']:r for r in q}
  for k in ('k0','k1'):
   a,b=bm.get('o1_'+k),bm.get('o2_'+k)
   if a and b: rev+=1; bc+=a['answer_correct'] and b['answer_correct']
 return {'documents':len(rows),'overall':sm(rows),'by_category':{k:sm(v) for k,v in bycat.items()},'by_layout':{k:sm(v) for k,v in sorted(bylay.items())},'complete_quartets':sum(all(r['answer_correct'] for r in q) for q in byq.values()),'reversal_pairs':rev,'reversal_both_correct':bc,'reversal_rate':bc/rev,'rows':rows}
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(TOK))
    assert sha(CK) == 'fed298748e62def6f1751ef1bb7df0cdec51d53fac6e4c86e8c00a95dccdd430'
    assert sha(TOK) == 'e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b'
    man = json.loads((PILOT / 'PILOT1_MATERIAL_MANIFEST.json').read_text())
    assert man['language_train_sha256'] == sha(PILOT / 'language_train.jsonl')
    assert man['language_dev_sha256'] == sha(PILOT / 'language_dev.jsonl')
    assert man['binding_rehearsal_sha256'] == sha(PILOT / 'binding_rehearsal.json')
    assert man['binding_dev_sha256'] == sha(PILOT / 'binding_dev.json')
    tr = [json.loads(x) for x in (PILOT / 'language_train.jsonl').read_text(encoding='utf-8').splitlines()]
    dv = [json.loads(x) for x in (PILOT / 'language_dev.jsonl').read_text(encoding='utf-8').splitlines()]
    rp = json.loads((PILOT / 'binding_rehearsal.json').read_text())['quartets']
    dp = json.loads((PILOT / 'binding_dev.json').read_text())['quartets']
    rehearsal = [d for q in rp for d in q['docs']]
    binddev = [d for q in dp for d in q['docs']]
    assert len(rp) == 80 and len(rehearsal) == 320 and len(dp) == 20 and len(binddev) == 80
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(dev)
    stream_tr = torch.tensor(encode_documents(tok, [{'text': x['text']} for x in tr]), dtype=torch.long)
    stream_dv = torch.tensor(encode_documents(tok, [{'text': x['text']} for x in dv]), dtype=torch.long)
    prompts = ['Hello', 'Hello, my name is', 'What is your name?', 'The dog is', 'The girl went to', 'Yesterday I', 'I am happy because', 'Why did the dog run?', 'Tell me a short story about a dog.', 'he womputeld']

    def set_scope(scope):
        for p in model.parameters():
            p.requires_grad_(scope == 'binding')
        if scope == 'language':
            for p in model.base_model.parameters(): p.requires_grad_(True)
            for i in range(4):
                for p in model.base_model.blocks[i].parameters(): p.requires_grad_(False)
            for p in model.localizer.parameters(): p.requires_grad_(False)
            for module in (model.wq, model.wk, model.wv, model.wo):
                for p in module.parameters(): p.requires_grad_(False)

    # Baseline measurements are frozen before optimizer creation.
    model.eval()
    pre = {'language_loss': lang_eval(model, stream_dv, dev, SEED + 1001), 'raw_generations': [generate(model, tok, p) for p in prompts], 'binding': binding_eval(model, binddev, dev)}
    pre['language_perplexity'] = math.exp(pre['language_loss'])
    (OUT / 'BASELINE_RESULTS.json').write_text(json.dumps(pre, indent=2), encoding='utf-8')

    # Zero-update scope/gradient preflight; no optimizer exists here.
    preflight = {'status': 'PILOT1_PRETREATMENT_PASS_PENDING', 'optimizer_created': False, 'optimizer_steps': 0, 'retention_evaluated': False, 'canonical_checkpoint_sha256': sha(CK), 'tokenizer_sha256': sha(TOK), 'language_train_sha256': sha(PILOT / 'language_train.jsonl'), 'language_dev_sha256': sha(PILOT / 'language_dev.jsonl'), 'binding_rehearsal_sha256': sha(PILOT / 'binding_rehearsal.json'), 'binding_dev_sha256': sha(PILOT / 'binding_dev.json'), 'schedule': '9 language then 1 binding, repeated for exactly 1000 updates', 'protected_language_blocks': [0, 1, 2, 3]}
    # Representative language batch: protected blocks/localizer/retrieval must have no gradients.
    set_scope('language'); model.train(); test_lg = torch.Generator().manual_seed(SEED); starts = torch.randint(0, len(stream_tr) - 257, size=(64,), generator=test_lg); off = torch.arange(256); x = stream_tr[starts[:, None] + off[None, :]].to(dev); y = stream_tr[starts[:, None] + off[None, :] + 1].to(dev); model.zero_grad(set_to_none=True); ltest = F.cross_entropy(model.base_model(x).reshape(-1, 1024), y.reshape(-1)); ltest.backward()
    def nz(name):
        p = dict(model.named_parameters())[name]; return p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.norm()) > 0
    preflight['language_gradient_scope'] = {'block0_zero': not nz('base_model.blocks.0.attention.projection.weight'), 'block3_zero': not nz('base_model.blocks.3.attention.projection.weight'), 'block4_nonzero': nz('base_model.blocks.4.attention.projection.weight'), 'language_head_nonzero': nz('base_model.language_head.weight'), 'localizer_zero': all(p.grad is None for p in model.localizer.parameters()), 'retrieval_zero': all(p.grad is None for module in (model.wq, model.wk, model.wv, model.wo) for p in module.parameters())}
    model.zero_grad(set_to_none=True)
    # Representative binding batch: all model groups must be trainable/receive gradients.
    set_scope('binding'); ch = rehearsal[:32]; bx = torch.tensor([d['full_document_token_ids'] for d in ch], device=dev); bq = torch.tensor([d['qdp'] for d in ch], device=dev); ba = torch.tensor([d['answer_causal_position'] for d in ch], device=dev); by = torch.tensor([d['target_value_token'] for d in ch], device=dev); bout, bex = model(bx, bq, ba); bix = torch.arange(32, device=dev); bla = F.cross_entropy(bout[bix, ba], by); bli = loc_loss(bex['localization_attention'], ch); (bla + LAM * bli).backward()
    preflight['binding_gradient_scope'] = {'block0_nonzero': nz('base_model.blocks.0.attention.projection.weight'), 'block3_nonzero': nz('base_model.blocks.3.attention.projection.weight'), 'block4_nonzero': nz('base_model.blocks.4.attention.projection.weight'), 'localizer_nonzero': all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.norm()) > 0 for p in model.localizer.parameters()), 'retrieval_nonzero': all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.norm()) > 0 for module in (model.wq, model.wk, model.wv, model.wo) for p in module.parameters())}
    model.zero_grad(set_to_none=True); set_scope('binding')
    preflight['status'] = 'PILOT1_PRETREATMENT_PASS'
    assert all(preflight['language_gradient_scope'].values()) and all(preflight['binding_gradient_scope'].values())
    (PILOT / 'PREFLIGHT.json').write_text(json.dumps(preflight, indent=2), encoding='utf-8')
    print('PILOT1_PRETREATMENT_PASS', flush=True)

    spec = {'pilot': 'Language Pilot 1', 'seed': SEED, 'starting_checkpoint_sha256': sha(CK), 'tokenizer_sha256': sha(TOK), 'language_train_sha256': sha(PILOT / 'language_train.jsonl'), 'language_dev_sha256': sha(PILOT / 'language_dev.jsonl'), 'binding_rehearsal_sha256': sha(PILOT / 'binding_rehearsal.json'), 'binding_dev_sha256': sha(PILOT / 'binding_dev.json'), 'optimizer': 'AdamW', 'learning_rate': 3e-4, 'weight_decay': .05, 'gradient_clip': 2.0, 'language_batch_size': 64, 'language_context': 256, 'binding_batch_size': 32, 'total_updates': 1000, 'language_updates': 900, 'binding_updates': 100, 'schedule': '9 language updates then 1 binding update', 'lr_schedule': 'constant 3e-4 (Pilot 0 precedent)', 'objective_binding': 'answer CE + existing hard-min permutation-invariant localization loss', 'language_scope': 'token/position embeddings, transformer blocks 4-7, final norm, language head; blocks 0-3 and T13 localizer/retrieval frozen', 'binding_scope': 'full model', 'language_sampling': 'torch Generator seed 8380 random windows', 'binding_sampling': 'precomputed deterministic 100-step schedule, 8 quartets per step, seed 8380'}
    spectext = json.dumps(spec, sort_keys=True, indent=2); (OUT / 'TRAINING_SPEC.json').write_text(spectext, encoding='utf-8'); spech = hashlib.sha256(spectext.encode()).hexdigest()
    torch.manual_seed(SEED); random.seed(SEED); opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.05); bind_order = list(range(len(rp))); rng = random.Random(8380); bsched = []
    for ep in range(10):
        rng.shuffle(bind_order)
        for j in range(0, len(bind_order), 8): bsched.append([bind_order[k] for k in bind_order[j:j+8]])
    bqmap = {q['quartet_id']: q for q in rp}; lg = torch.Generator().manual_seed(SEED); metrics = []; bi = 0
    for step in range(1, 1001):
        is_bind = step % 10 == 0
        if is_bind:
            docs = []
            for qi in bsched[bi]: docs.extend(bqmap[f'pilot1_rehearsal:qt_{qi:06d}']['docs'])
            bi += 1; set_scope('binding'); x = torch.tensor([d['full_document_token_ids'] for d in docs], device=dev); q = torch.tensor([d['qdp'] for d in docs], device=dev); a = torch.tensor([d['answer_causal_position'] for d in docs], device=dev); y = torch.tensor([d['target_value_token'] for d in docs], device=dev); out, ex = model(x, q, a); ix = torch.arange(32, device=dev); la = F.cross_entropy(out[ix, a], y); li = loc_loss(ex['localization_attention'], docs); loss = la + LAM * li; scope = 'binding'
        else:
            set_scope('language'); st = torch.randint(0, len(stream_tr) - 257, size=(64,), generator=lg); off = torch.arange(256); x = stream_tr[st[:, None] + off[None, :]].to(dev); y = stream_tr[st[:, None] + off[None, :] + 1].to(dev); out = model.base_model(x); loss = la = F.cross_entropy(out.reshape(-1, 1024), y.reshape(-1)); li = torch.tensor(0., device=dev); scope = 'language'
        opt.zero_grad(set_to_none=True); loss.backward(); gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step(); metrics.append({'step': step, 'scope': scope, 'loss': float(loss), 'answer_loss': float(la), 'binding_loss': float(li), 'grad_norm': float(gn)})
        if step % 100 == 0: print(metrics[-1], flush=True)
    model.eval(); final = OUT / 'checkpoints' / 'seed_8380' / 'latest.pt'; final.parent.mkdir(parents=True, exist_ok=True); torch.save({'model_state_dict': model.state_dict(), 'step': 1000, 'seed': SEED, 'objective': 'language_base_ce_with_blocks_0_3_protected_alternated_with_t13_binding', 'language_updates': 900, 'binding_updates': 100}, final)
    post = {'language_loss': lang_eval(model, stream_dv, dev, SEED + 1001), 'raw_generations': [generate(model, tok, p) for p in prompts], 'binding': binding_eval(model, binddev, dev)}; post['language_perplexity'] = math.exp(post['language_loss'])
    (OUT / 'training_metrics.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in metrics), encoding='utf-8')
    result = {'status': 'LANGUAGE_PILOT_1_COMPLETE', 'spec_sha256': spech, 'starting_checkpoint_sha256': sha(CK), 'final_checkpoint_sha256': sha(final), 'tokenizer_sha256': sha(TOK), 'language_train_sha256': sha(PILOT / 'language_train.jsonl'), 'language_dev_sha256': sha(PILOT / 'language_dev.jsonl'), 'binding_rehearsal_sha256': sha(PILOT / 'binding_rehearsal.json'), 'binding_dev_sha256': sha(PILOT / 'binding_dev.json'), 'updates': 1000, 'language_updates': 900, 'binding_updates': 100, 'optimizer_steps': 1000, 'sacred_retention_evaluations': 0, 'preflight': preflight, 'pre': pre, 'post': post}
    (OUT / 'RESULTS.json').write_text(json.dumps(result, indent=2), encoding='utf-8'); (OUT / 'REPORT.md').write_text('# Language Pilot 1 report\n\n' + json.dumps(result, indent=2), encoding='utf-8'); print('LANGUAGE_PILOT_1_COMPLETE', flush=True)
if __name__=='__main__': main()
