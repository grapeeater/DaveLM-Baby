"""Prospective SF11: one narrow subset, unchanged total SF10 exposure. No model access."""
import json,hashlib,shutil,re,difflib
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
from tokenizers import Tokenizer
H=Path(__file__).resolve().parent; ROOT=H.parent; S10=ROOT/'sf10_reduced_density_widening_v1'; S9=ROOT/'sf9_surface_order_curriculum_v1'; S8=ROOT/'sf8_margin_dose_comparison_v1'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v): Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def norm(s): return ' '.join(s.casefold().split())

# Fixed before construction; no paraphrase search or collision-driven resampling.
DEV_CONTENT={'A':('lifted','the paper kite'),'B':('washed','the red basket')}
NAMES={'A':['Alex','Owen'],'B':['Mia','Nora']}
SURFACES=['cloze','active_qa','passive_cloze','passive_qa']

def render(surface,name,verb,obj):
    fact=f'{name} {verb} {obj}.' if not surface.startswith('passive') else f'{obj} was {verb} by {name}.'
    cue=f'Who {verb} {obj}?' if surface.endswith('qa') else f'The person who {verb} {obj} was'
    return fact+'\n'+cue

def main():
    assert not (H/'PROTOCOL.json').exists()
    assert sha(S10/'FREEZE_RECEIPT.json')==(S10/'FREEZE_RECEIPT.sha256').read_text().split()[0]
    assert sha(S10/'SHA256SUMS.txt')==read(S10/'FREEZE_RECEIPT.json')['manifest_sha256']
    for l in (S10/'SHA256SUMS.txt').read_text().splitlines():
        h,n=l.split('  ',1); assert sha(S10/n)==h,n
    # Verify all completed SF10 outputs, without scoring or touching any locked panel.
    for l in (S10/'OUTPUT_SHA256SUMS.txt').read_text().splitlines():
        h,n=l.split('  ',1); assert sha(S10/n)==h,n
    old=read(S10/'PROTOCOL.json'); provenance=[]
    def copy(n):
        dest=H/n; dest.parent.mkdir(parents=True,exist_ok=True); assert not dest.exists()
        shutil.copyfile(S10/n,dest); assert sha(dest)==sha(S10/n)
        provenance.append({'source':str(S10/n),'destination':n,'sha256':sha(dest)})
    for n in ['SF2_ENGINE.py','SF2_PROTOCOL.json','KL_POOL.json','KL_POOL_MANIFEST.json','D3_SELECTION.json','EXTERNAL_INPUTS.json','TRAIN16_RETENTION.json']:
        copy(n)
    for sub in ['sources','data','authority']:
        for f in (S10/sub).glob('*'):
            if f.is_file(): copy(f.relative_to(S10).as_posix())
    parents=[]
    for seed,r in zip([87026,87027,87028],old['runs']):
        rr=dict(r,seed=seed); assert sha(rr['parent_checkpoint'])==rr['parent_checkpoint_sha256']
        st=read(Path(rr['parent_checkpoint']).parent/'STATUS.json')
        assert st['status']=='ACQUISITION_SUCCESS' and st['checkpoint_sha256']==rr['parent_checkpoint_sha256']
        parents.append(rr)
    external=read(H/'EXTERNAL_INPUTS.json')
    external.update({r['parent_checkpoint']:r['parent_checkpoint_sha256'] for r in parents})
    for f,h in external.items(): assert sha(f)==h,f
    write(H/'EXTERNAL_INPUTS.json',external)
    # EXTERNAL_INPUTS gains the three required SF8 hashes; mark this explicitly as an extension.
    provenance=[r for r in provenance if r['destination']!='EXTERNAL_INPUTS.json']
    allitems=read(S10/'TRAIN.json')
    # First canonical NEW shard, entirely structural: A painted/tin cup; B bought/green ball.
    new=[r for r in allitems if r['id'].startswith('SF9:') and r['shard']==0]
    retained=[r for r in allitems if r['id'].startswith('TRAIN:')]
    assert len(new)==32 and len(retained)==16
    items=retained+new
    assert len(set(norm(r['prompt']) for r in items))==48
    oldschedule=read(S10/'SCHEDULE.json'); schedule=[]
    for u in oldschedule:
        v=dict(u)
        if u['kind']=='english':
            ids=[i for i in u['ids'] if i.startswith('TRAIN:')]+[r['id'] for r in new]
            assert len(ids)==36
            v['ids']=ids
        schedule.append(v)
    write(H/'TRAIN.json',items); write(H/'SCHEDULE.json',schedule)
    tok=Tokenizer.from_file(old['tokenizer'])
    cand={r['candidates'][i]:r['candidate_token_ids'][i] for r in retained for i in [0,1]}
    ds=[]; do=[]
    def item(group,name,verb,obj,prompt,answer,subgroup,kind):
        names=NAMES[group]; candidates=[' '+n+'.' for n in names]
        pair=f'SF11DEV:{kind}:{group}:{subgroup}'
        return {'id':pair+':'+name,'family_id':f'SF11DEV:{kind}:{group}','pair_id':pair,'subgroup':subgroup,'arm':'factual',
          'prompt':prompt,'prompt_token_ids':tok.encode(prompt).ids,'candidates':candidates,'candidate_token_ids':[cand[c] for c in candidates],
          'correct_index':names.index(answer),'actor':name,'predicate':verb,'object':obj}
    for group,names in NAMES.items():
        verb,obj=DEV_CONTENT[group]
        for i,name in enumerate(names):
            other=names[1-i]
            for surface in SURFACES:
                ds.append(item(group,name,verb,obj,render(surface,name,verb,obj),name,surface,'surface'))
            for order in ['order0','order1']:
                fact=f'{name} {verb} {obj}.'; card=f'A card says {other}.'
                context=fact+' '+card if order=='order0' else card+' '+fact
                for query in ['fact','copy']:
                    cue=f'The person who {verb} {obj} was' if query=='fact' else 'Copy the name on the card:'
                    do.append(item(group,name,verb,obj,context+'\n'+cue,name if query=='fact' else other,order+':'+query,'order'))
    write(H/'DEV_SURFACE.json',ds); write(H/'DEV_ORDER.json',do)
    exposure=Counter(i for u in schedule if u['kind']=='english' for i in u['ids'])
    balance={'unique_items':48,'new_items':32,'preservation_items':16,'names':dict(Counter(r['candidates'][r['correct_index']] for r in items)),
      'new_categories':dict(Counter(r['subgroup'] for r in new)), 'presentations':sum(exposure.values()),'presentations_by_item':dict(exposure),
      'per_update':36,'new_per_update':32,'preservation_per_update':4,'selection_ids':[r['id'] for r in new],
      'unique_contexts_excluding_cues':len({norm(r['prompt'].split('\n')[0]) for r in items}),
      'note':'Breadth unit is unique complete rendered input prompt, including cue, exactly as SF9/SF10 count items. Shared factual clauses are not independent contexts.'}
    write(H/'CURRICULUM_MANIFEST.json',balance)
    proto={k:old[k] for k in ['parent_anchor','parent_sha256','tokenizer','tokenizer_sha256','runtime','optimizer','kl_data','binding','scope','evaluation_updates']}
    proto.update(study='SF11_NARROW_BREADTH_WIDENING_V1',created_utc=datetime.now(timezone.utc).isoformat(),runs=parents,seeds=[87026,87027,87028],
      hypothesis='At fixed SF10 presentation total/category/name mix, narrowing distinct widening prompts may retain widening while avoiding D3 regression. SF10 weakened density-only remediation, not proved breadth causation.',
      manipulated_variable='Distinct widening curriculum breadth: 128 ->32 new items; total including original preservation16:144->48. Same batch36, new32+preservation4, same6480total. Reduced lexical breadth and increased repetition per surviving new item are inseparable from narrowing at fixed total exposure.',
      curriculum='Choose exactly SF10 new shard0 (first canonical object/verb combination per name pair), retain all four surfaces and both orders x both queries for both actor assignments. No outcome-based selection. TRAIN16 retention shard rotation/order and padding unchanged from SF10.',
      minimum_breadth='Four answer identities x four surface categories=16; four identities x two source orders x two query sources=16 competing items. These32 are the minimum for complete name-counterbalanced SF9-category coverage. Add all16 historical preservation items=48. This is minimum within the inherited two-arm-name-pair template structure, not a universal combinatorial lower bound.',
      sampling='200updates,20cycles9English+1binding. 180English batches of36: fixed32new + originalSF10 rotating4TRAIN records. 5760new+720preservation=6480total. Eachnew180presentations; eachTRAIN1645presentations. Binding entries and shared pad values unchanged.',
      margin={'lambda_margin':.25,'M':1.0,'mechanics':'Pinned SF10 first-answer full-vocabulary logsoftmax hinge mean over36; unchanged.'},
      english_objective='Unchanged SF10 mean causal answer4tokens+EOS CE +1.0 student||Pilot1 KL over160frozenpositions +.25mean max(0,1-firstanswergap). No optimizer state inherited; fresh AdamW5e-5, unchanged settings.',
      dev={'surface':16,'order':16,'content':DEV_CONTENT,'rule':'Same SF9 renderers, disjoint verbs/objects from SF9/SF10 training/dev and originalTRAIN16. One complete namepair x surface/order-query family per group; 8reversal pairs and2complete8-item families per panel. Freeze before any parent output. No historical DEV scored in SF11.'},
      gates={'retention_acquisition':old['gates']['retention_acquisition'],'language':old['gates']['language'],'d3':old['gates']['d3'],'binding_each_pool':old['gates']['binding_each_pool'],
       'dev_surface':'correct>=15/16 AND exact>=13/16 AND each surface exact>=3/4',
       'dev_order':'exact>=13/16 AND fact>=7/8 AND copy>=7/8 AND each order>=7/8 AND each order:query cell>=3/4',
       'development_gain':'At200 exact surface>=ownu0+4 AND exact order>=ownu0+4 (25 percentage point gain per panel). No adaptation to baseline results.',
       'continue':old['gates']['continue'],'endpoint_pass':'u200 AND allretention AND dev_surface AND dev_order AND development_gain'},
      evaluation='0/100/200: unchangedTRAIN16, newDEV_SURFACE16, newDEV_ORDER16, identical128alignedTinyStoriesDEV, frozenD3256, both80docbindingpools. Exact candidate4token likelihood excludingEOS; margin>0; normalgreedy32 exactcandidate+EOS. Family/pair summaries descriptive; no broader English claim.',
      stopping='Stop eachrun at100/200 on any retention failure; no widening early stop. Full endpoint200only. Other independent runs continue after valid scientificfailure. No rescue, no replacementseed, no postresultcurriculumchange.',
      transfer='All historicalSF1transfer and historicalSF9/SF10DEV remain unscored in SF11. No optional reevaluation is authorized by this package. FINAL/sacred never accessed.',
      persistence='Inherited SF10 full atomic rolling restart eachupdate plus permanent100/200. Sequential blocking child execution; explicit durable ledger before launch/PID/end/status. Exactlyone launchperseed; no auto-restart/replay of valid failures. Mechanical interruption reported with committedstate preserved.',
      classification={'priority1':'MECHANICAL_INCOMPLETE if any run lacks classified endpoint dueexecutionfailure','priority2':'RETENTION_REGRESSION if anyTRAIN16/language/binding failure','priority3':'BREADTH_HYPOTHESIS_SUPPORTED if>=2/3 reach200andallendpointgates','priority4':'BREADTH_HYPOTHESIS_WEAKENED if>=2D3fails; SUBSTANTIALLY_WEAKENED if>=2finalD3>=.018 (matching historicalmagnitude)', 'otherwise':'IfallD3safeand0completeendpoints:NARROW_CURRICULUM_INSUFFICIENT; otherpatternsMIXED_OR_UNRESOLVED'},
      limits='Three parent replications, no population certainty. Comparison to SF10 is historical, not concurrent same-seed breadthrandomization; a selectedsubset cannot isolate cardinality from lexicalcontent/topology. Support is bounded practical evidence, never proof of a general breadth mechanism.',
      next_action_rule='If supported:review frozen result before any transfer authorization. If D3weakened:preregister one retention-objective review before any further training. If safeinsufficient:review the frozen widening failures before choosing a new treatment. If otherretentionfailure:review that safetyfailure. No action executed afterreport.')
    write(H/'PROTOCOL.json',proto)
    # Minimum mechanical controller changes: counts, same-total schedule, stronger fresh-panel gates; training body byte-identical.
    src=(S10/'CONTROLLER.py').read_text(); out=src.replace('SF10','SF11')
    out=out[out.index('import argparse'):]
    out='"""SF11: narrower curriculum; identical SF10 loss, optimizer, scope and persistence."""\n'+out
    out=out.replace('len(items) == len(idx) == 144','len(items) == len(idx) == 48')
    out=out.replace('assert set(exposure.values()) == {45} and len(exposure) == 144',"assert len(exposure) == 48\n    assert all(exposure[r['id']] == (45 if r['id'].startswith('TRAIN:') else 180) for r in items)")
    out=out.replace("return agg['correct'] >= 15 and agg['exact'] >= 13","return agg['correct'] >= 15 and agg['exact'] >= 13 and all(agg['subgroups'][s]['exact'] >= 3 for s in ['cloze','active_qa','passive_cloze','passive_qa'])")
    out=out.replace("return agg['exact'] >= 13 and fact >= 7 and copy >= 7", "return agg['exact'] >= 13 and fact >= 7 and copy >= 7 and all(sg[o+':fact']['exact']+sg[o+':copy']['exact']>=7 for o in ['order0','order1']) and all(sg[o+':'+q]['exact']>=3 for o in ['order0','order1'] for q in ['fact','copy'])")
    out=out.replace("g = gates(ret, dsurf, dord, ch, d3, base_loss, u)","g = gates(ret, dsurf, dord, ch, d3, base_loss, u)\n        g['development_gain'] = dsurf['exact'] >= read(out/'update0_devsurface_RESULT.json')['exact']+4 and dord['exact'] >= read(out/'update0_devorder_RESULT.json')['exact']+4\n        g['endpoint_pass'] = g['endpoint_pass'] and g['development_gain']")
    # Preserve all rest; no model operation or target math edited.
    assert out[out.index('    for u in schedule[completed:]:'):]==src[src.index('    for u in schedule[completed:]:'):]
    (H/'CONTROLLER.py').write_text(out,encoding='utf-8',newline='\n')
    (H/'CONTROLLER_DIFF.patch').write_text(''.join(difflib.unified_diff(src.splitlines(True),out.splitlines(True),fromfile='SF10/CONTROLLER.py',tofile='SF11/CONTROLLER.py')),encoding='utf-8')
    write(H/'PROVENANCE.json',{'byte_identical':provenance,'sf10_receipt_sha256':sha(S10/'FREEZE_RECEIPT.json'),'sf10_outputs_sha256':sha(S10/'OUTPUT_SHA256SUMS.txt'),
      'source_controller_sha256':sha(S10/'CONTROLLER.py'),'sf9_builder_sha256':sha(S9/'CURRICULUM.py'),'sf8_final_report_sha256':sha(S8/'FINAL_REPORT.md'),
      'extended_external_inputs':'Three SF8 parent hashes added. All preexisting external hashes unchanged.','locked_panels_not_opened':True,'no_checkpoint_loaded':True})
    print('BUILT',len(items),'items; exposure',sum(exposure.values()))

if __name__=='__main__': main()
