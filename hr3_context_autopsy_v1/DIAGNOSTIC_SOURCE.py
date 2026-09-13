"""Bounded nonsacred autopsy. No optimizer, weight updates, or new treatment."""
import json, hashlib, sys, math, re
from pathlib import Path
from collections import Counter, defaultdict
import statistics as st

ROOT=Path(r'C:\DaveLM-CADAVER')
B=ROOT/'human_readiness_hr3_block3_causal_seed87006_v7'
OUT=ROOT/'hr3_context_autopsy_v1'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p): return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def save(name,obj):
    p=OUT/name
    assert not p.exists(),str(p)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
def stats(xs):
    return {'n':len(xs),'mean':st.mean(xs),'min':min(xs),'median':st.median(xs),'max':max(xs)} if xs else {}

def main():
    OUT.mkdir(exist_ok=False)
    save('DIAGNOSTIC_PLAN.json',{
        'scope':'Existing nonsacred readiness facts and frozen scheduled training records only; no optimizer or weight changes.',
        'selection':'All 24 saved fact scores on Pilot1, corrected HR1, HR3-500. Gradient/representation diagnostics: first lexicographic fact family, all four members; first literal English and binding batches.',
        'measurements':['canonical candidate preference vs matched reversal changes','word-only vs period contribution','absolute candidate sequence mass','training exposure counts','last-cue residual/logit changes','eval-mode gradients of existing sentence CE, binding objective, factual answer NLL, and candidate-pair logistic loss'],
        'limitations':'Diagnostic gradient losses do not authorize a training objective. Single-family/single-batch gradients cannot establish global interference or representation sufficiency. No trained probe or gradient update.',
        'decision':'Train only if one smallest treatment is uniquely justified. Otherwise report scientific fork.',
        'final_access':False,'sacred_access':False,'source_sha256':sha(Path(__file__))})
    sys.path.insert(0,str(B/'sources'))
    import hr3_block3_runtime as rt
    import torch
    import torch.nn.functional as F
    from tokenizers import Tokenizer
    rt.verify_integrity(B)
    protocol=read(B/'HR3_PROTOCOL.json')
    assert sha(Path(protocol['tokenizer_path']))==rt.TOKENIZER_SHA256
    rawpath=ROOT/'human_readiness_hr3_causal_seed87006_execution_v7/NONSACRED_DEV_COMPARISON.json'
    assert sha(rawpath)=='d22fc52652936a9fc55e4dad2fb0222121ac31d06e48325202cf8b59795a8b65'
    comparison=read(rawpath)
    selected=[c for c in comparison['checkpoints'] if c['name'] in ['Pilot1_parent','HR1_causal_aligned_500','HR3_block3_update_500']]
    train=rows(B/protocol['data']['english_train']); index={r['id']:r for r in train}
    schedule=read(B/protocol['data']['english_schedule'])['batches']
    presentations=[index[i] for batch in schedule for i in batch['record_ids']]
    counts=Counter(r['id'] for r in presentations)
    builder=ROOT/'human_readiness_hr1_seed87004/HR1_BUILD_SOURCE.py'
    split_re=r'(?<=[.!?])\s+'
    exposure={'pool_records':len(train),'presentations':len(presentations),'unique_presented_records':len(counts),'presentation_multiplicity':dict(Counter(counts.values())),
      'supervised_content_tokens':sum(len(r['token_ids']) for r in presentations),'supervised_eos_tokens':len(presentations),
      'max_content_length':max(len(r['token_ids']) for r in presentations),
      'records_with_internal_sentence_boundary':sum(bool(re.search(split_re,r['text'])) for r in presentations),
      'records_with_question_mark':sum('?' in r['text'] for r in presentations),
      'scheduled_multi_record_contexts':0,'explicit_matched_english_counterfactual_families':0,
      'english_updates':450,'binding_updates':50,'binding_document_presentations':1600,
      'source_builder':str(builder),'source_builder_sha256':sha(builder),
      'objective':'English mean CE per supervised token, one sentence per independent batch row, BOS and final EOS; binding separate mean answer CE + 1.0536573711078283 localization. No English candidate contrast or factual task mixer.'}
    save('EXPOSURE.json',exposure)
    saved={}
    for c in selected:
        rr=c['controlled_rows']['fact']; pairs=defaultdict(list)
        for r in rr: pairs[r['reversal_pair_id']].append(r)
        profiles=[]
        for pid,ms in sorted(pairs.items()):
            ms=sorted(ms,key=lambda x:x['assignment']); assert len(ms)==2
            names=sorted(ms[0]['candidates'])
            def d(r,word=False):
                sc=[sum(ts[:-1]) if word else sum(ts) for ts in r['candidate_token_log_scores']]
                return sc[r['candidates'].index(names[0])]-sc[r['candidates'].index(names[1])]
            d0,d1=map(d,ms); q=1 if ms[0]['candidates'][ms[0]['correct_index']]==names[0] else -1
            bias=(d0+d1)/2; signal=q*(d0-d1)/2
            profiles.append({'pair':pid,'canonical_candidates':names,'d_assignment0':d0,'d_assignment1':d1,
              'context_odd_component_correct_direction':signal,'assignment_even_bias':bias,
              'unchanged_chosen_candidate':(d0>0)==(d1>0),
              'both_correct':all(r['correct'] for r in ms),
              'word_only_both_correct':all((d(r,True)>0)==(r['candidates'][r['correct_index']]==names[0]) and d(r,True)!=0 for r in ms)})
        chosen_names=Counter(); first=0
        for r in rr:
            if r['chosen_index'] is not None:
                chosen=r['candidates'][r['chosen_index']].strip().rstrip('.')
                chosen_names[chosen]+=1; first+=r['prompt'].startswith(chosen+' ')
        saved[c['name']]={'pairs':profiles,'unchanged_choice_pairs':sum(p['unchanged_chosen_candidate'] for p in profiles),
          'positive_context_direction_pairs':sum(p['context_odd_component_correct_direction']>0 for p in profiles),
          'both_correct_pairs':sum(p['both_correct'] for p in profiles),'word_only_both_correct_pairs':sum(p['word_only_both_correct'] for p in profiles),
          'bias_absolute':stats([abs(p['assignment_even_bias']) for p in profiles]),'context_signed':stats([p['context_odd_component_correct_direction'] for p in profiles]),
          'chosen_names':dict(chosen_names),'first_mentioned_choice_items':first,
          'absolute_candidate_pair_mass':stats([sum(math.exp(s) for s in r['candidate_log_scores']) for r in rr]),
          'correct_answer_nll_per_token':stats([-r['candidate_log_scores'][r['correct_index']]/len(r['candidate_token_ids'][r['correct_index']]) for r in rr])}
    save('SAVED_SCORE_AUTOPSY.json',saved)
    rt.configure_runtime(87006); device=torch.device('cuda'); tok=Tokenizer.from_file(protocol['tokenizer_path'])
    facts=rows(Path(protocol['readiness_dev']['path'])/'DEV_FACTS.jsonl')
    family=sorted({r['family_id'] for r in facts})[0]; probe=sorted([r for r in facts if r['family_id']==family],key=lambda r:r['id'])
    def group(name):
        if name.startswith('base_model.blocks.'): return 'block_'+name.split('.')[2]
        if name.startswith('base_model.'): return name.split('.')[1]
        return 'specialized'
    for c in selected:
        cp=Path(c['checkpoint']); before=sha(cp); assert before==c['checkpoint_sha256']
        model=rt.load_model(cp,device,B).eval()
        weights={k:v.detach().clone() for k,v in model.state_dict().items()}
        params=list(model.named_parameters())
        for _,p in params: p.requires_grad_(True)
        gradients={}; losses={}; hidden=[]
        def get_grad(label,loss):
            gg=torch.autograd.grad(loss,[p for _,p in params],allow_unused=True)
            grouped=defaultdict(list)
            for (n,p),g in zip(params,gg): grouped[group(n)].append((torch.zeros_like(p) if g is None else g).detach().flatten().cpu())
            gradients[label]={g:torch.cat(v) for g,v in grouped.items()}; losses[label]=float(loss.detach())
        xx,yy=rt.aligned_tensors([index[i] for i in schedule[0]['record_ids']],device)
        get_grad('scheduled_sentence_ce',F.cross_entropy(model.base_model(xx).reshape(-1,1024),yy.reshape(-1),ignore_index=-100))
        def scores(row):
            prompt=tok.encode(row['prompt']).ids; out=[]
            for cand in row['candidate_token_ids']:
                seq=[2]+prompt+cand
                lp=model.base_model(torch.tensor([seq],device=device))[0].log_softmax(-1)
                out.append(torch.stack([lp[len(prompt)+j,t] for j,t in enumerate(cand)]).sum())
            return torch.stack(out)
        for objective in ['factual_answer_nll','factual_pair_logistic']:
            vals=[]
            for r in probe:
                ss=scores(r); ci=r['correct_index']
                vals.append(-ss[ci]/len(r['candidate_token_ids'][ci]) if objective=='factual_answer_nll' else F.softplus(ss[1-ci]-ss[ci]))
            get_grad(objective,torch.stack(vals).mean())
        bs=read(B/protocol['data']['binding_schedule'])['batches'][0]
        qs=rt.flatten_quartets(read(B/protocol['data']['binding_rehearsal']))
        docs=rt.binding_docs_for_batch(qs,bs['quartet_ids'],bs['documents'])
        loss,_=rt.binding_loss(model,docs,device,rt.pinned_binding(B)); get_grad('scheduled_binding_objective',loss)
        for r in probe:
            captures={}; handles=[]
            for i,block in enumerate(model.base_model.blocks):
                handles.append(block.register_forward_hook(lambda m,inp,out,i=i: captures.__setitem__('block_'+str(i),out[0,-1].detach().cpu())))
            with torch.no_grad():
                logits=model.base_model(torch.tensor([[2]+tok.encode(r['prompt']).ids],device=device))[0,-1]
                captures['final_norm']=model._captured[0,-1].detach().cpu()
                captures['logits']=logits.detach().cpu()
            for h in handles:h.remove()
            hidden.append((r,captures))
        contrasts=[]
        for q in [0,1]:
            members=sorted([(r,h) for r,h in hidden if r['query']==q],key=lambda z:z[0]['assignment'])
            contrasts.append({'query':q,'layers':{g:{'relative_l2':float((members[0][1][g]-members[1][1][g]).norm()/members[0][1][g].norm().clamp_min(1e-12)),
               'cosine':float(F.cosine_similarity(members[0][1][g],members[1][1][g],dim=0))} for g in members[0][1]}})
        norms={label:{g:float(v.norm()) for g,v in gs.items()} for label,gs in gradients.items()}
        cos={}
        for other in ['scheduled_sentence_ce','scheduled_binding_objective','factual_answer_nll']:
            cos[other]={g:float(F.cosine_similarity(gradients['factual_pair_logistic'][g],v,dim=0)) if v.norm()>0 and gradients['factual_pair_logistic'][g].norm()>0 else None for g,v in gradients[other].items()}
        assert all(torch.equal(v,weights[k]) for k,v in model.state_dict().items())
        assert all(p.grad is None for _,p in params)
        assert sha(cp)==before
        save(c['name']+'_GRADIENTS_REPRESENTATIONS.json',{'checkpoint':str(cp),'sha256':before,'family':family,'losses':losses,'gradient_norms':norms,'cosine_with_pair_logistic_gradient':cos,'assignment_hidden_contrasts':contrasts,'weights_bitwise_unchanged':True,'checkpoint_hash_unchanged':True,'optimizer_created':False,'note':'All-parameter eval-mode derivative diagnostics. Protected groups were enabled for differentiation only; no optimizer or .grad accumulation. Gradients are local derivatives, not AdamW update predictions.'})
        print(c['name']+' diagnostic persisted',flush=True)
        del model,weights,gradients; torch.cuda.empty_cache()
    save('PROVENANCE.json',{'status':'BOUNDED_AUTOPSY_COMPLETE','input_comparison_sha256':sha(rawpath),'bundle_receipt_sha256':sha(B/'FREEZE_RECEIPT.json'),'source_sha256':sha(Path(__file__)),'python':sys.version,'torch':torch.__version__,'checkpoint_updates':0,'optimizer_created':False,'final_accessed':False,'sacred_accessed':False})

if __name__=='__main__': main()
