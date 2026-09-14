from __future__ import annotations
import hashlib, json, math
from array import array
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(r"C:\DaveLM-CADAVER")
TD=ROOT/"phase2a_post_t28_tokenizer_diagnostic_v1"
SD=ROOT/"phase2a_post_t28_suffix_state_trace_v1"
items=json.loads((TD/"ITEMS.json").read_text(encoding="utf-8"))
trace=json.loads((SD/"TRACE.json").read_text(encoding="utf-8"))

def rate(rows,key): return sum(bool(x[key]) for x in rows)/len(rows) if rows else 0
def subseq_count(seq, pat):
    return sum(1 for i in range(len(seq)-len(pat)+1) if seq[i:i+len(pat)]==pat)

raw=array('H')
lp=ROOT/"baby_vnext_phase1g_language_v1"/"data"/"LANGUAGE_TRAIN_STREAM.u16"
with lp.open('rb') as f: raw.fromfile(f,lp.stat().st_size//2)
stream=list(raw)

name_ids={x['name']:x['name_token_ids'] for x in items}
name_rows=[]
for name,ids in sorted(name_ids.items()):
    rows=[x for x in items if x['name']==name]
    shared=max((next((i for i,(a,b) in enumerate(zip(ids,o)) if a!=b),min(len(ids),len(o))) for n,o in name_ids.items() if n!=name),default=0)
    # If another sequence is a prefix, identity is unresolved through the shorter sequence.
    disamb=shared+1 if shared < len(ids) else None
    name_rows.append({'name':name,'token_ids':ids,'name_tokens':len(ids),'longest_shared_prefix':shared,
                      'first_identity_disambiguating_position_1based':disamb,
                      'full_string_count_phase1g':subseq_count(stream,ids),'n':len(rows),
                      'exact_rate':rate(rows,'exact'),'first_correct_diverge_rate':sum(x['mode']=='first_token_correct_then_diverge' for x in rows)/len(rows),
                      'min_token_frequency':min(x['min_train_frequency'] for x in rows)})

freq_sorted=sorted(items,key=lambda x:x['min_train_frequency']); q=len(freq_sorted)//4
freq_groups={'rarest_quartile':freq_sorted[:q],'most_common_quartile':freq_sorted[-q:]}
length_groups={str(k):[x for x in items if x['name_tokens']==k] for k in sorted({x['name_tokens'] for x in items})}

tok_summary=json.loads((TD/'SUMMARY.json').read_text(encoding='utf-8'))
tok_summary['name_table']=name_rows
tok_summary['frequency_effects']={k:{'n':len(v),'exact_rate':rate(v,'exact'),'diverge_rate':sum(x['mode']=='first_token_correct_then_diverge' for x in v)/len(v),
                                         'frequency_range':[min(x['min_train_frequency'] for x in v),max(x['min_train_frequency'] for x in v)]} for k,v in freq_groups.items()}
tok_summary['length_effects']={k:{'n':len(v),'exact_rate':rate(v,'exact'),'diverge_rate':sum(x['mode']=='first_token_correct_then_diverge' for x in v)/len(v)} for k,v in length_groups.items()}
(TD/'DETAILED_SUMMARY.json').write_text(json.dumps(tok_summary,indent=2)+'\n',encoding='utf-8')

# Pair teacher/free rows by seed,item,timestep.
idx={(r['seed'],r['id'],r['step'],r['mode']):r for r in trace}
steps=defaultdict(lambda:{'n':0,'teacher_top1':0,'free_top1':0,'teacher_rank':0,'free_rank':0,'prefix_same':0})
seeds=defaultdict(lambda:{'n':0,'teacher_top1':0,'free_top1':0,'teacher_rank':0,'free_rank':0})
for key,r in idx.items():
    seed,rid,step,mode=key
    if mode!='teacher': continue
    f=idx[(seed,rid,step,'free')]; tl=r['layers'][-1]; fl=f['layers'][-1]
    d=steps[step]; d['n']+=1; d['teacher_top1']+=tl['target_top1']; d['free_top1']+=fl['target_top1']; d['teacher_rank']+=tl['target_rank']; d['free_rank']+=fl['target_rank']; d['prefix_same']+=r['target_prefix']==f['generated_prefix']
    s=seeds[seed]; s['n']+=1; s['teacher_top1']+=tl['target_top1']; s['free_top1']+=fl['target_top1']; s['teacher_rank']+=tl['target_rank']; s['free_rank']+=fl['target_rank']

def finish(d): return {'n':d['n'],'teacher_top1_rate':d['teacher_top1']/d['n'],'free_top1_rate':d['free_top1']/d['n'],'teacher_mean_rank':d['teacher_rank']/d['n'],'free_mean_rank':d['free_rank']/d['n'],**({'same_prefix_fraction':d['prefix_same']/d['n']} if 'prefix_same' in d else {})}
derived={'population':'first-token-correct-then-diverge only (frozen)',
         'exact_success_trace':'NOT_RUN_PROTOCOL_NOT_FROZEN','wrong_first_trace':'NOT_RUN_PROTOCOL_NOT_FROZEN',
         'by_step':{str(k):finish(v) for k,v in sorted(steps.items())},
         'by_seed':{str(k):finish(v) for k,v in sorted(seeds.items())},
         'layer_summary':json.loads((SD/'SUMMARY.json').read_text(encoding='utf-8'))}
(SD/'DETAILED_SUMMARY.json').write_text(json.dumps(derived,indent=2)+'\n',encoding='utf-8')

def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
for d in (TD,SD):
    lines=[]
    for p in sorted(x for x in d.rglob('*') if x.is_file() and x.name!='SHA256SUMS.txt'):
        lines.append(f"{h(p)}  {p.relative_to(d)}")
    (d/'SHA256SUMS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')

print(json.dumps({'tokenizer':tok_summary,'suffix':derived},indent=2))
