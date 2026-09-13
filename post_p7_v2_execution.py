"""Single-use P7/v2 executor. Never regenerates items or imports a training runner."""
import hashlib, importlib.util, json, math, platform, stat, sys, time
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent
F=ROOT/'post_p7_language_report_card_v2_seed8391'
OUT=ROOT/'post_p7_language_report_card_v2_seed8391_execution_p7'
CP=ROOT/'archive/DAVELM_P7_MILESTONE_seed8380/davelm_p7_latest.pt'
TOK=Path(r'C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json')
CP_HASH='d41ed1186cc945aa05dbd2ba3086fac08ff3b97035c9532e4fa70e78b149a20e'
SUM_HASH='e3ab16daeb7c1f54e136db68daecf348383fa0f830810cc5e200d10f84dcc151'
TOK_HASH='e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b'
SECTIONS=('near_distribution','counterfactual','surface_form','distractor')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def save(name,obj):
    p=OUT/name
    with p.open('xb') as f:f.write((json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())
def read_jsonl(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines()]
def describe(xs):
    import statistics
    return {'n':len(xs),'min':min(xs),'median':statistics.median(xs),'mean':statistics.fmean(xs),'max':max(xs)}
def verify():
    assert sha(CP)==CP_HASH and sha(TOK)==TOK_HASH
    assert sha(F/'SHA256SUMS.txt')==SUM_HASH
    hashes={}
    for line in (F/'SHA256SUMS.txt').read_text().splitlines():
        h,n=line.split('  ');assert Path(n).name==n and sha(F/n)==h,n
        hashes[n]=h
    assert {p.name for p in F.iterdir()}==set(hashes)|{'SHA256SUMS.txt'}
    for p in F.iterdir():
        assert not p.stat().st_mode & stat.S_IWRITE
        assert b'\r' not in p.read_bytes()
    for n,m in json.loads((F/'FREEZE_RECEIPT.json').read_text())['artifacts'].items():
        assert hashes[n]==m['sha256'] and (F/n).stat().st_size==m['bytes']
    spec=importlib.util.spec_from_file_location('sealed_semantic_validator',F/'post_p7_v2_independent_validator.py')
    validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
    rows=read_jsonl(F/'ITEMS.jsonl');validation=validator.validate(rows)
    assert len(rows)==376 and len({r['item_id'] for r in rows})==376
    assert len({' '.join(r['prompt'].casefold().split()) for r in rows})==376
    manifest=json.loads((F/'MANIFEST.json').read_text());assert dict(Counter(r['section'] for r in rows))==manifest['counts']
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(TOK))
    for r in rows:
        p=r['prompt'];pids=tok.encode(p).ids;assert pids==r['prompt_token_ids'] and tok.decode(pids)==p
        for c,ids in zip(r.get('candidates',[]),r.get('candidate_token_ids',[])):
            assert len(ids)==3 and tok.encode(c).ids==ids and tok.decode(ids)==c
            assert tok.encode(p+c).ids==pids+ids
        assert 1+len(pids)+(32 if r['section']=='naturalistic' else 3)<=256
    for entry in json.loads((F/'OVERLAP_AUDIT.json').read_text())['sources']:
        assert sha(entry['path'])==entry['sha256'],entry['path']
    stop_tokens={i:tok.decode([i]) for i in range(tok.get_vocab_size()) if any(c in tok.decode([i]) for c in '.!?')}
    assert stop_tokens=={5:'!',18:'.',35:'?',901:'?"'}
    return rows,tok,{'status':'PASS_BEFORE_MODEL_LOAD','checkpoint_sha256':CP_HASH,'tokenizer_sha256':TOK_HASH,'detached_sha256':SUM_HASH,'frozen_artifacts':hashes,'independent_semantics':validation,'punctuation_stop_tokens':stop_tokens,'punctuation_note':'Token 901 contains a question mark followed by a quote. Stop on it and retain the entire token verbatim.','read_only_flags':True,'source_hashes':True,'token_boundaries':True,'no_sacred_material':True}

def summarize(scores):
    groups=defaultdict(list)
    for row in scores:groups[row['family_id']].append(row)
    families=[]
    for fid,rs in sorted(groups.items()):
        pairs=defaultdict(list)
        for r in rs:pairs[(r['query'],r['fact_order'],r['distractor_position'])].append(r)
        ps=[]
        for (q,o,d),pair in sorted(pairs.items()):
            pair.sort(key=lambda r:r['assignment']);assert len(pair)==2 and [r['assignment'] for r in pair]==[0,1]
            assert pair[0]['correct_index']!=pair[1]['correct_index']
            ps.append({'query':q,'fact_order':o,'distractor_position':d,'item_ids':[r['item_id'] for r in pair],'both_correct':all(r['correct'] for r in pair),'margins':[r['margin'] for r in pair]})
        n=sum(p['both_correct'] for p in ps)
        families.append({'family_id':fid,'section':rs[0]['section'],'items':len(rs),'item_correct':sum(r['correct'] for r in rs),'ties':sum(r['tie'] for r in rs),'complete':all(r['correct'] for r in rs),'reversals_correct':n,'reversals_total':len(ps),'reversal_fraction':n/len(ps),'margin':describe([r['margin'] for r in rs]),'reversal_pairs':ps})
    sections={}
    for s in SECTIONS:
        fs=[f for f in families if f['section']==s];rs=[r for r in scores if r['section']==s]
        sections[s]={'items':len(rs),'correct':sum(r['correct'] for r in rs),'accuracy':sum(r['correct'] for r in rs)/len(rs),'ties':sum(r['tie'] for r in rs),'complete_families':sum(f['complete'] for f in fs),'families':len(fs),'complete_fraction':sum(f['complete'] for f in fs)/len(fs),'reversal_profile':dict(sorted(Counter(f['reversals_correct'] for f in fs).items())),'reversals_correct':sum(f['reversals_correct'] for f in fs),'reversals_total':sum(f['reversals_total'] for f in fs),'mean_within_family_reversal':sum(f['reversal_fraction'] for f in fs)/len(fs),'margin':describe([r['margin'] for r in rs])}
    paired=[]
    for i in range(8):
        found={s:next(f for f in families if f['family_id']==f'{s}:lex{i:02d}') for s in ('near_distribution','surface_form','distractor')}
        paired.append({'lexical_id':i,'outcomes':{s:{k:f[k] for k in ('items','item_correct','complete','reversals_correct','reversals_total','reversal_fraction','margin')} for s,f in found.items()}})
    return {'sections':sections,'families':families,'matched_family_comparisons':paired,'no_pass_fail_gate':True,'naturalistic_rubric':'User review required for semantic fields; no automatic success count'}

def report(summary,scores,generations):
    lines=['# P7 on prospective report card v2','', 'Single authorized execution. Controlled scores are matched name likelihood comparisons, not generated answers. Naturalistic text is saved verbatim; JSON strings additionally expose whitespace and empty outputs.','', '## Section results','', '| Section | Items correct | Ties | Complete families | Reversals both correct | Mean family reversal | Margin min / median / mean / max |','|---|---:|---:|---:|---:|---:|---|']
    for s,v in summary['sections'].items():
        m=v['margin'];lines.append(f"| {s} | {v['correct']}/{v['items']} | {v['ties']} | {v['complete_families']}/{v['families']} | {v['reversals_correct']}/{v['reversals_total']} | {v['mean_within_family_reversal']:.6f} | {m['min']:.6f} / {m['median']:.6f} / {m['mean']:.6f} / {m['max']:.6f} |")
    lines+=['','## Every family','', '| Family | Correct items | Complete | Reversals | Mean margin | Minimum margin |','|---|---:|---|---:|---:|---:|']
    for f in summary['families']:lines.append(f"| {f['family_id']} | {f['item_correct']}/{f['items']} | {f['complete']} | {f['reversals_correct']}/{f['reversals_total']} | {f['margin']['mean']:.6f} | {f['margin']['min']:.6f} |")
    lines+=['','## Matched diagnostic comparisons','', 'Only lexical families 0–7 are matched across active, surface and distractor conditions. Distractor families have twice as many items and reversal pairs. Surface changes both fact and query wording.','']
    for p in summary['matched_family_comparisons']:lines+=['```json',json.dumps(p,ensure_ascii=False,sort_keys=True),'```']
    lines+=['','## Frozen representative sample','', 'Every item of the lexicographically first family in each controlled section. No selection based on outcomes.','']
    for s in SECTIONS:
        fid=min(r['family_id'] for r in scores if r['section']==s)
        for r in scores:
            if r['family_id']!=fid:continue
            ci=r['correct_index'];lines += [f"### {r['item_id']}",'','PROMPT:','```text',r['prompt'],'```',f"Correct candidate (JSON): {json.dumps(r['candidates'][ci])}",f"Competing candidate (JSON): {json.dumps(r['candidates'][1-ci])}",f"Candidate log-likelihoods [Sam, Tom]: {r['candidate_log_likelihoods']}",f"Margin: {r['margin']!r}",f"Selected candidate (JSON): {json.dumps(r['selected_candidate'])}",f"Outcome: {'tie' if r['tie'] else 'correct' if r['correct'] else 'incorrect'}",'']
    lines+=['## Every naturalistic prompt and raw continuation','']
    for g in generations:
        lines += [f"### {g['item_id']}",'','PROMPT:','```text',g['prompt'],'```','BABY:','```text',g['continuation'],'```','Raw continuation as JSON: '+json.dumps(g['continuation'],ensure_ascii=False),'RUBRIC: HUMAN REVIEW REQUIRED for grammatical completeness, subject/object consistency, relation/action appropriateness, repetition and contradiction.',f"Mechanical truncation: {g['truncated']}; stop reason: {g['stop_reason']}; generated tokens: {len(g['generated_token_ids'])}",'']
    lines+=['## Limits','', 'This battery contains only two answer names and two factual predicates. Active factual QA is more demanding than the original P7 single-story-prompt panel. Complete-family/reversal outcomes, not isolated correct answers, assess reliable selection. Surface facts and query style change together; distractor comparisons use matched families. Exact non-overlap excludes neither paraphrases nor semantic contamination. No binding-retention test was authorized. No broad English, conversational, general-reasoning, architecture, capacity, or causal-training claim follows from this single checkpoint evaluation. V1 controlled scores remain withdrawn.','']
    return '\n'.join(lines)

def main():
    assert not OUT.exists(),'No retry, reuse, or overwrite permitted'
    rows,tok,preflight=verify()
    OUT.mkdir()
    save('PREFLIGHT_RECEIPT.json',preflight)
    (OUT/'RUNNER.py').write_bytes(Path(__file__).read_bytes())
    import torch, tokenizers
    sys.path.insert(0,r'C:\DaveLM-v0.9')
    from v0_8_2.model import build_model
    sources={}
    for name,module in sorted(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if path and name.startswith(('v0_8_2','v0_7','v0_2_1')) and Path(path).is_file():sources[str(Path(path))]=sha(path)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    raw=torch.load(CP,map_location='cpu',weights_only=True)
    # Extract only the ordinary base state. No wrapper or specialized modules instantiated.
    state={k[len('base_model.'):]:v for k,v in raw['model_state_dict'].items() if k.startswith('base_model.')}
    assert state and all(not t.is_floating_point() or t.dtype==torch.float32 for t in state.values())
    model=build_model('untied');model.load_state_dict(state,strict=True)
    assert model.context_size==256 and sum(p.numel() for p in model.parameters())==10594944
    for k,v in model.state_dict().items():assert torch.equal(v,state[k])
    def digest(m):
        h=hashlib.sha256()
        for k,t in sorted(m.state_dict().items()):
            h.update(k.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    before=digest(model)
    model.to(device).eval();model.requires_grad_(False)
    assert digest(model)==before
    runtime={'device':device,'device_name':torch.cuda.get_device_name(0) if device.startswith('cuda') else platform.processor(),'python':platform.python_version(),'torch':str(torch.__version__),'tokenizers':tokenizers.__version__,'model_dtype':'float32','score_dtype':'CPU float64 log_softmax and sum','autocast':False,'tf32':False,'batch_size':1,'padding':False,'truncation':False,'base_only_strict_load':True,'model_sources':sources,'runner_sha256':sha(Path(__file__)),'state_digest_before':before,'punctuation_stop_ids':[5,18,35,901]}
    save('RUNTIME_AND_LOAD_RECEIPT.json',runtime)
    # One-shot marker precedes first forward; failures are preserved, never auto-resumed.
    save('INFERENCE_STARTED.json',{'time':time.time(),'checkpoint_sha256':CP_HASH,'frozen_sha256':SUM_HASH})
    start=time.monotonic();scores=[];gens=[];forward_count=0
    with (OUT/'RAW_CONTROLLED_SCORES.jsonl').open('x',encoding='utf-8',newline='\n') as out:
        for r in rows:
            if r['section']=='naturalistic':continue
            lp=[];L=len(r['prompt_token_ids'])
            for c in r['candidate_token_ids']:
                ids=[2]+r['prompt_token_ids']+c
                with torch.inference_mode():
                    logits=model(torch.tensor([ids],device=device))[0,L:L+3,:].detach().to(device='cpu',dtype=torch.float64)
                    logp=torch.log_softmax(logits,dim=-1)
                    selected=logp[torch.arange(3),torch.tensor(c)]
                assert bool(torch.isfinite(selected).all())
                lp.append(selected.tolist());forward_count+=1
            ll=[sum(v) for v in lp];ci=r['correct_index'];margin=ll[ci]-ll[1-ci]
            selected=None if ll[0]==ll[1] else r['candidates'][0 if ll[0]>ll[1] else 1]
            score={**r,'candidate_token_log_probabilities':lp,'candidate_log_likelihoods':ll,'margin':margin,'correct':margin>0,'tie':margin==0,'selected_candidate':selected}
            scores.append(score);out.write(json.dumps(score,ensure_ascii=False,allow_nan=False)+'\n');out.flush()
            if len(scores)%64==0:print(f'Controlled {len(scores)}/352',flush=True)
    with (OUT/'RAW_GENERATIONS.jsonl').open('x',encoding='utf-8',newline='\n') as out:
        for r in rows:
            if r['section']!='naturalistic':continue
            ids=[2]+r['prompt_token_ids'];new=[];stop='token_limit'
            for _ in range(32):
                with torch.inference_mode():next_id=int(model(torch.tensor([ids],device=device))[0,-1,:].argmax().item())
                forward_count+=1;new.append(next_id);ids.append(next_id)
                if next_id==3:stop='eos';break
                if next_id in (5,18,35,901):stop='punctuation';break
            g={'item_id':r['item_id'],'prompt':r['prompt'],'generated_token_ids':new,'continuation':tok.decode(new,skip_special_tokens=True),'stop_reason':stop,'truncated':stop=='token_limit','rubric':{k:'HUMAN REVIEW REQUIRED' for k in ('grammatical_completeness','subject_consistency','object_consistency','relation_action_appropriateness','repetition_failure','context_contradiction')}}
            gens.append(g);out.write(json.dumps(g,ensure_ascii=False,allow_nan=False)+'\n');out.flush()
            print(f'Naturalistic {len(gens)}/24',flush=True)
    assert len(scores)==352 and len(gens)==24
    assert digest(model)==before and sha(CP)==CP_HASH
    _,_,post=verify();assert post==preflight
    assert all(sha(p)==h for p,h in sources.items())
    summary=summarize(scores);save('SUMMARY.json',summary)
    with (OUT/'REPORT.md').open('xb') as f:f.write(report(summary,scores,gens).encode())
    save('POST_EXECUTION_RECEIPT.json',{'status':'COMPLETE','checkpoint_sha256_after':sha(CP),'state_digest_after':digest(model),'checkpoint_and_memory_unchanged':True,'frozen_material_unchanged':True,'forward_calls':forward_count,'controlled_items':352,'naturalistic_items':24,'elapsed_seconds':time.monotonic()-start,'training':False,'sacred_material_accessed':False,'model_source_hashes_unchanged':True})
    files=sorted(OUT.iterdir());save('OUTPUT_MANIFEST.json',{'artifacts':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in files},'note':'Manifest excludes itself; detached checksums include manifest. No circular hash.'})
    files=sorted(OUT.iterdir())
    with (OUT/'SHA256SUMS.txt').open('xb') as f:f.write(''.join(sha(p)+'  '+p.name+'\n' for p in files).encode())
    for p in OUT.iterdir():p.chmod(p.stat().st_mode & ~stat.S_IWRITE)
    print(json.dumps({'status':'COMPLETE','output':str(OUT),'checksum_sha256':sha(OUT/'SHA256SUMS.txt'),'sections':summary['sections']},indent=2),flush=True)

if __name__=='__main__':main()
