"""Prospectively sealed SF11 reporter. No model inference; compact report and complete raw references."""
import json,stat
import CONTROLLER as C

def load_trajectory(d):
    result={}
    for u in [0,100,200]:
        if not (d/f'update{u}_train16_RESULT.json').exists(): continue
        ch=C.read(d/f'update{u}_checks.json')
        result[str(u)]={'train16':C.read(d/f'update{u}_train16_RESULT.json'),
          'surface':C.read(d/f'update{u}_devsurface_RESULT.json'),'order':C.read(d/f'update{u}_devorder_RESULT.json'),
          'language':ch['language'],'binding':{k:{'gate':v['gate'],'summary':v.get('summary',{})} for k,v in ch['binding'].items()},
          'd3':C.read(d/f'd3_update{u}_summary.json'),'gates':C.read(d/f'update{u}_GATES.json') if u else None}
    return result

def classify(results):
    success=sum(x['status']['gates']['endpoint_pass'] for x in results.values())
    if len(results)!=3: return 'MECHANICAL_INCOMPLETE',success
    if any(x['status']['status'] not in ['STOP_REGRESSION','ACQUISITION_FAIL','ACQUISITION_SUCCESS'] for x in results.values()): return 'MECHANICAL_INCOMPLETE',success
    for x in results.values():
        g=x['status']['gates']
        if not g['retention_acquisition'] or not g['language'] or not all(g['binding'].values()): return 'RETENTION_REGRESSION',success
    if success>=2: return 'BREADTH_HYPOTHESIS_SUPPORTED',success
    d3bad=sum(not x['status']['gates']['d3'] for x in results.values())
    severe=sum(x['trajectory'][str(x['status']['completed'])]['d3']['mean_combined_name_probability']>=.018 for x in results.values())
    if severe>=2: return 'BREADTH_HYPOTHESIS_SUBSTANTIALLY_WEAKENED',success
    if d3bad>=2: return 'BREADTH_HYPOTHESIS_WEAKENED',success
    if d3bad==0 and success==0: return 'NARROW_CURRICULUM_INSUFFICIENT',success
    return 'MIXED_OR_UNRESOLVED',success

def subgroup(row,key): return row['subgroups'][key]['exact']

def render(results,p,classification,n):
    lines=['# SF11 — narrower breadth, unchanged presentation dose','',f'**{classification}** — complete endpoints: **{n}/3**.','',
      '## Curriculum and exposure','',
      '48 unique input prompts: 16 unchanged preservation items + 32 widening items. The latter cross four answer names with four surface categories (16) and two orders × two query sources (16). Each category includes both assignments for each name pair. This is the minimum complete name-balanced coverage within the inherited SF9 template structure.','',
      'Selection was fixed before outcomes: SF10’s first canonical NEW shard (Alex/Owen: painted the tin cup; Mia/Nora: bought the green ball). All16 preservation items keep SF10’s rotation. Per update:9 correct answers per name;4 items per new surface or order/query category. Across unique prompts:12 correct answers per name.','',
      'Excluding the query/cue yields32 distinct context prefixes; the comparable SF9/SF10 breadth unit is the complete input prompt, yielding48. New surface counts:cloze4, activeQA4, passivecloze4, passiveQA4. Competing counts:each order8, each source8, each order×source4. Including preservation, unique fact-answer prompts40 and card-answer prompts8; query-source balance is exact within the competing arm.','',
      '| Study | Unique prompts | Batch | Planned English updates | Planned presentations | Presentations/item |',
      '|---|---:|---:|---:|---:|---|',
      '| SF8 |16|32|180|5,760|360|','| SF9 |144|144|180|25,920|180|','| SF10 |144|36|180|6,480|45|',
      '| SF11 |48|36|180|6,480|new32:180; preservation16:45|','',
      'SF11 preserves SF10’s total dose, name/category mix, preservation dose, padding, binding schedule and full training step. Narrowing inevitably changes which lexical contexts recur and repetition per surviving item; cardinality is not isolated from subset composition.','',
      '## Frozen gates','',
      '- Surface16: forced-choice ≥15; greedy exact ≥13; every surface ≥3/4 exact.',
      '- Order16: exact ≥13; fact and copy each ≥7/8; each order ≥7/8; each order×query ≥3/4.',
      '- Both new panels: ≥4 additional exact responses versus that parent’s u0 (25 percentage points), at u200.',
      '- TRAIN16:16 correct,16 exact,8 reversals,4 families. Language ≤ownu0+0.25nat. D3≤0.01.',
      '- Binding: EACH pool answer≥76/80,BD≥76/80,collapse0. No averages. Stop on retention failure at100/200; endpoint200only.','',
      '## Parent mapping and checkpoint provenance','',
      '| New seed | SF8 parent | Parent SHA-256 |','|---|---|---|']
    for r in p['runs']: lines.append(f"|{r['seed']}|{r['parent_checkpoint']}|{r['parent_checkpoint_sha256']}|")
    lines+=['','## All preregistered trajectories','',
      '|Seed|Update|TRAIN16 c/x/r/f|Surface c/x|Order exact fact/copy|CE / PPL|D3|','|---|---:|---|---|---|---|---:|']
    for key,x in results.items():
        for u,z in x['trajectory'].items():
            a=z['train16']; s=z['surface']; o=z['order']; l=z['language']
            fact=sum(subgroup(o,k+':fact') for k in ['order0','order1']); card=sum(subgroup(o,k+':copy') for k in ['order0','order1'])
            lines.append(f"|{x['seed']}|{u}|{a['correct']}/{a['exact']}/{a['reversals']}/{a['families']}|{s['correct']}/{s['exact']}|{o['exact']} ({fact}/{card})|{l['loss']:.6f} / {l['perplexity']:.3f}|{z['d3']['mean_combined_name_probability']:.8f}|")
    lines+=['','## New development subgroups and retention','']
    for key,x in results.items():
        lines += [f"### Seed {x['seed']}",'', '|Update|cloze|active QA|passive cloze|passive QA|fact→card fact/copy|card→fact fact/copy|','|---:|---:|---:|---:|---:|---|---|']
        for u,z in x['trajectory'].items():
            s=z['surface'];o=z['order']
            vals=[subgroup(s,k) for k in ['cloze','active_qa','passive_cloze','passive_qa']]
            lines.append(f"|{u}|{vals[0]}/4|{vals[1]}/4|{vals[2]}/4|{vals[3]}/4|{subgroup(o,'order0:fact')}/4, {subgroup(o,'order0:copy')}/4|{subgroup(o,'order1:fact')}/4, {subgroup(o,'order1:copy')}/4|")
        st=x['status']; final=x['trajectory'][str(st['completed'])]
        lines+=['',f"**{st['status']} at update {st['completed']}.** Gates: `{json.dumps(st['gates'])}`",'']
        for u,z in x['trajectory'].items():
            for pool,v in z['binding'].items(): lines.append(f"- u{u} {pool}: `{json.dumps(v['summary'])}`; PASS={v['gate']}.")
        lines+=['',f"Final checkpoint: `{st['checkpoint']}`",f"SHA-256: `{st['checkpoint_sha256']}`",f"Actual English presentations: {(st['completed']-st['completed']//10)*36}; no unrun u200 result is inferred.",'']
    safe=sum(x['status']['gates']['d3'] for x in results.values())
    surf_improved=sum(x['trajectory'][str(x['status']['completed'])]['surface']['exact']>x['trajectory']['0']['surface']['exact'] for x in results.values())
    order_improved=sum(x['trajectory'][str(x['status']['completed'])]['order']['exact']>x['trajectory']['0']['order']['exact'] for x in results.values())
    retained=sum(x['status']['gates']['retention_acquisition'] for x in results.values())
    lines+=['## Evidence ledger and Dave-coded interpretation','',
      f'- D3 safe at the stopping endpoint: **{safe}/3**. Complete frozen widening+retention endpoint: **{n}/3**.',
      f'- Surface exact improved versus parent: **{surf_improved}/3**; order/source exact improved: **{order_improved}/3**. Improvement alone is not passage of the subgroup gates.',
      f'- SF8 TRAIN16 acquisition retained: **{retained}/3**.',
      '- The new panels test the same surface/source structures on two previously unused verb/object combinations. They do not establish unconstrained language, general reasoning, or broad transfer. The 32 new DEV items were frozen before parent/candidate scoring.',
      '- Historical SF9/SF10 development and SF1 transfer panels were not scored. FINAL/sacred remained untouched. There was no outcome-driven change, extra seed, rescue, or follow-up experiment.',
      '- Three independent SF8 parent continuations give bounded replication. Comparison with SF10 is historical; it does not prove that the number of distinct contexts alone caused D3 behavior.',
      '- Dave-coded: fewer examples count as a win only if Baby both learns the wider lesson and stays safe. Moving a few DEV answers while D3 breaks is still a failed treatment. Calling it general English would be bullshit.','',
      '## Integrity','',f"Receipt: `{C.sha(C.H/'FREEZE_RECEIPT.json')}`",f"Payload manifest: `{C.sha(C.H/'SHA256SUMS.txt')}`",'Raw per-item scores/generations, D3 traces, binding rows and update metrics are in runs/. The result classification is committed before any potential future panel decision. No historical-panel reevaluation is included.','']
    if 'SUPPORTED' in classification: action='Review the frozen SF11 success before authorizing any historical transfer-panel evaluation.'
    elif 'WEAKENED' in classification: action='Preregister one retention-objective review before authorizing further widening training.'
    elif classification=='NARROW_CURRICULUM_INSUFFICIENT': action='Review the frozen SF11 widening failures before selecting a new treatment.'
    else: action='Review the frozen SF11 safety and subgroup results before authorizing another treatment.'
    lines.append('NEXT ACTION: '+action)
    return '\n'.join(lines)+'\n'

def main():
    p=C.verify(); results={}
    for r in p['runs']:
        key=f"seed_{r['seed']}_curriculum"; d=C.H/'runs'/key
        st=C.read(d/'STATUS.json'); assert C.sha(st['checkpoint'])==st['checkpoint_sha256']
        assert C.sha(r['parent_checkpoint'])==r['parent_checkpoint_sha256']
        results[key]={'seed':r['seed'],'status':st,'trajectory':load_trajectory(d)}
    label,n=classify(results)
    C.rt.atomic_json({'classification':label,'endpoint_passes':n,'historical_panels':'LOCKED_UNSCORED','final_sacred_accessed':False},C.H/'CLASSIFICATION.json')
    C.rt.atomic_json({'classification':label,'endpoint_passes':n,'results':results},C.H/'RESULTS.json')
    (C.H/'FINAL_REPORT.md').write_text(render(results,p,label,n),encoding='utf-8',newline='\n')
    paths=[C.H/x for x in ['CLASSIFICATION.json','RESULTS.json','FINAL_REPORT.md','RUN_LEDGER.json']]
    paths+=sorted(f for f in (C.H/'runs').rglob('*') if f.is_file() and f.name!='restart.pt')
    paths+=list((C.H/'logs').glob('*.log'))
    text=''.join(f'{C.sha(f)}  {f.relative_to(C.H).as_posix()}\n' for f in paths)
    (C.H/'OUTPUT_SHA256SUMS.txt').write_text(text,encoding='utf-8',newline='\n')
    for line in text.splitlines():
        h,f=line.split('  ',1); assert C.sha(C.H/f)==h
    for f in paths+[C.H/'OUTPUT_SHA256SUMS.txt']: f.chmod(stat.S_IREAD)
    print('STUDY_COMPLETE',label,'OUTPUT_SHA256',C.sha(C.H/'OUTPUT_SHA256SUMS.txt'),flush=True)

if __name__=='__main__': main()
