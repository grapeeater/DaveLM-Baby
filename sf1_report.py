import json,hashlib,collections,statistics,math,shutil
from pathlib import Path
R=Path(r'C:\DaveLM-CADAVER');B=R/'single_fact_acquisition_sf1_seed87011';O=R/'single_fact_acquisition_sf1_seed87011_run'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=read(O/'STATUS.json');p=read(B/'PROTOCOL.json');assert sha(p['parent'])==p['parent_sha256']
assert sha(B/'RECEIPT.json')==(B/'RECEIPT.sha256').read_text().split()[0]
assert sha(B/'SHA256SUMS.txt')==read(B/'RECEIPT.json')['manifest_sha256']
for line in (B/'SHA256SUMS.txt').read_text().splitlines():
 h,n=line.split('  ',1);assert sha(B/n)==h
metrics=[json.loads(l) for l in (O/'TRAIN_METRICS.jsonl').read_text().splitlines()];assert [m['update'] for m in metrics]==list(range(1,len(metrics)+1))
completed=len(metrics);assert completed==s.get('completed',s.get('update'));bs=sum(x['kind']=='binding' for x in metrics)
results={};checks={};cps={}
for u in [0,100,200]:
 path=O/f'update{u}_acquisition_RESULT.json'
 if path.exists():results[str(u)]=read(path);checks[str(u)]=read(O/f'update{u}_checks.json')
 if u and (O/f'checkpoint_{u}.pt').exists():
  h=sha(O/f'checkpoint_{u}.pt');assert h==(O/f'checkpoint_{u}.sha256').read_text().strip();cps[str(u)]={'path':str(O/f'checkpoint_{u}.pt'),'sha256':h}
transfer={}
for label in ['heldout','alternate','copy','competing']:
 path=O/(label+'_RESULT.json')
 if path.exists():assert s['transfer_opened'];transfer[label]=read(path)
if not s['transfer_opened']:assert not transfer
scope=read(O/'PARAMETER_SCOPE.json');assert not any(n.startswith(f'base_model.blocks.{i}.') for n in scope['english']['active'] for i in range(4));assert all(n.startswith('base_model.') for n in scope['english']['active']);assert not scope['binding']['frozen']
freq={}
for name in ['TRAIN','HELDOUT','ALTERNATE','COPY','COMPETING']:
 rs=read(B/(name+'.json'));freq[name]={k:dict(collections.Counter(r[k] for r in rs)) for k in ['actor','object','predicate','correct_index','subgroup']}
first10=[x['loss'] for x in metrics if x['kind']=='english'][:10];last10=[x['loss'] for x in metrics if x['kind']=='english'][-10:]
lastu=max(map(int,results));rawlast=[json.loads(l) for l in (O/f'update{lastu}_acquisition_RAW.jsonl').read_text(encoding='utf-8').splitlines()];byfamily=collections.defaultdict(list)
for r in rawlast:byfamily[r['family_id']].append(r)
diag={'correct_target_nll_by_response_position':[statistics.mean(-r['token_scores_including_eos'][r['correct_index']][j] for r in rawlast) for j in range(5)],'family_profiles':{n:{'correct':sum(r['correct'] for r in rs),'n':len(rs),'generated_text_counts':dict(collections.Counter(r['generated_text'] for r in rs))} for n,rs in byfamily.items()},'new_behavioral_inference':False,'source':'saved acquisition raw scores only'}
out={'classification':s,'completed_updates':completed,'english_updates':completed-bs,'binding_updates':bs,'checkpoint_hashes':cps,'acquisition':results,'checks':checks,'transfer':transfer,'training_first10_mean_loss':statistics.mean(first10),'training_last10_mean_loss':statistics.mean(last10),'saved_output_diagnosis':diag,'balance_audit':freq,'parent_hash_reverified':True,'bundle_hashes_reverified':True,'final_accessed':False,'sacred_accessed':False}
(O/'SUMMARY.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
lines=['# SF1 single-fact acquisition study','',f"Classification: **{s['status']}**.",'',f"Completed {completed}/200 prospectively fixed updates ({completed-bs} factual English, {bs} binding). No readiness or v1.0 claim. No FINAL or sacred material accessed.",'',
'## Frozen intervention and provenance','',
'Parent: Pilot1 SHA256 `'+p['parent_sha256']+'`. Seed87011. English/binding scope: blocks0-3 and all specialized components frozen on English; all T13 parameters active on binding. Exact named parameter lists are in PARAMETER_SCOPE.json. Inactive parameter values and every existing optimizer-state field were compared after every English update and remained unchanged.','',
'16 training items, four complete families; two actor pairs, two predicates, two objects per pair, balanced actor swaps. Heldout uses the opposite object pool per actor pair, so actor/object combinations are absent from training while vocabulary remains familiar. Candidates all four tokens. All 152 prompts across training and diagnostics are unique. Completed input length <=50 tokens. Final rendered semantic parser, negative tests, causal labels, token boundaries, repeated-batch identities, and restart round trip passed before training. Full prompts had zero exact/substring matches in the allowlisted nonsacred corpora/evaluations; this does not exclude semantic/constituent overlap.','',
'Fixed 200 updates, 20 cycles of nine factual plus one binding; English batch32 is two copies of every training record. No sentence rehearsal was introduced. AdamW5e-5, wd.05, betas(.9,.999), eps1e-8, clipping2; correct causal answer-plus-EOS CE, no pairwise loss or normalization. Objective and masking pinned from the earlier v8 harness; binding implementation pinned from validated HR3/Pilot1 code. Binding pool unchanged (320 documents/80 quartets), each quartet scheduled twice. Readiness gates remain untouched.','',
'Endpoint acquisition gate: 16/16 restricted correctness AND16/16 exact candidate+EOS AND8/8 reversals AND4/4 complete families. Fixed endpoint selection only. Regression checks at100/200: each pool >=76/80 answers and BD, zero collapse; aligned sentence DEV loss no more than parent+0.25 nats. Stop on either regression. No transfer inference before successful endpoint gates.','',
'Bundle: `'+str(B)+'`; receipt SHA256 `'+sha(B/'RECEIPT.json')+'`; payload manifest SHA256 `'+sha(B/'SHA256SUMS.txt')+'`.','',
'## Acquisition and language monitoring','',
'| Update | Correct | Reversals | Families | Exact+EOS | Mean margin | Candidate mass | Aligned loss | PPL |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for u,v in results.items():
 l=checks[u]['language'];lines.append(f"| {u} | {v['correct']}/16 | {v['reversals']}/8 | {v['families']}/4 | {v['exact']}/16 | {v['mean_margin']:.5f} | {v['mean_mass']:.5g} | {l['loss']:.5f} | {l['perplexity']:.3f} |")
last=results[str(max(map(int,results)))];lines+=['',f"Mean supervised English loss, first10 updates: {statistics.mean(first10):.5f}; last10: {statistics.mean(last10):.5f}. These response/EOS training losses differ from the nonsacred sentence-language loss. Raw margins, candidate token scores including separate EOS, absolute candidate mass, top1/EOS diagnostics, and every unmodified generated response are durably preserved in the per-panel RAW files.",'']
if s['status']=='SF1_ACQUISITION_FAIL':lines+=['The endpoint failed its own-training-family gate. This is an acquisition failure on familiar controlled examples, not evidence that only transfer failed. No heldout, alternate-cue, copy or competing-name diagnostic was scored. No extra training was attempted.','']
elif s['status']=='STOP_REGRESSION':lines += ['The preregistered regression guard stopped the run. Intermediate training-item performance does not satisfy the fixed endpoint gate. Transfer panels remained locked. The result is preserved without extending training or relaxing the cost/preservation guard.','']
else:lines+=['All own-training acquisition criteria passed at the fixed endpoint. This demonstrates the narrow trained response interface; it does not by itself distinguish factual use from copying the only actor name. The already-frozen diagnostics below were evaluated in order after the endpoint was committed and hashed.','']
lines+=['## Both nonsacred binding pools — independently gated','','| Update | Pool | Answer | BD | Collapse | Quartets | Strict reversal | Gate |','|---|---|---:|---:|---:|---:|---:|---|']
for u,q in checks.items():
 for n,v in q['binding'].items():
  a=v['summary'];lines.append(f"| {u} | {n} | {a['answer_exact']}/80 | {a['both_distinct']}/80 | {a['slot_collapse']} | {a['complete_quartets']}/20 | {a['strict_reversal_both_correct']}/40 | {'PASS' if v['gate'] else 'FAIL'} |")
lines+=['','Queried-row and downstream correctness conditional on BD are preserved in each checks JSON. Pool scores are never averaged.','',
'## Frozen transfer and copy diagnostics','']
if transfer:
 lines+=['| Panel/subgroup | Correct | Reversals | Families | Exact+EOS | Mean margin | Candidate mass |','|---|---:|---:|---:|---:|---:|---:|']
 for name,v in transfer.items():
  for sub,z in [('TOTAL',v)]+list(v['subgroups'].items()):lines.append(f"| {name}/{sub} | {z['correct']}/{z['n']} | {z['reversals']}/{z['pairs']} | {z['families']}/{z['family_count']} | {z['exact']}/{z['n']} | {z['mean_margin']:.5f} | {z['mean_mass']:.5g} |")
 lines+=['','Heldout tests familiar actor/object recombination; alternate panels change voice/cue with heldout semantics. Isolated card-name copying tests response formatting under a non-factual copying instruction. Competing contexts include a factual actor and a different name on a card; changing only the query from factual cloze to card-copy requests different answers, with both mention orders balanced. These are diagnostic outcomes, not new readiness gates.','']
else:lines+=['HELDOUT16: NOT OPENED. ALTERNATE48: NOT OPENED. COPY8: NOT OPENED. COMPETING64: NOT OPENED. These panels remain frozen and unscored because the prerequisite did not pass. No copy/transfer conclusion is inferred from training items.','']
lines+=['## Interpretation limits and next boundary','',
'The study isolates a simpler factual response interface with the existing causal answer objective and protected Pilot1 scope. No naked discrimination loss, decoder penalty, extra blocks, architecture change, or broader corpus strategy was introduced. Success on one-name facts may be an identity-copying shortcut; the matched control diagnostics must be considered before any stronger claim. Failure would not establish architectural impossibility, a capacity ceiling or a unique need for discrimination/unfreezing. A materially different objective, scope or curriculum requires a new scientific decision; none is selected here.','',
'Results on training families are not independent heldout evidence and have no generalization confidence interval. Transfer items have correlated family structure; raw family/reversal profiles and subgroup counts are the evidence. No gates were adjusted after outcomes. Parameter-scope monitoring and inactive optimizer-state comparisons completed on every executed update. This is a bounded single-fact study only.','',
'## Saved-output failure diagnosis','',
f"At update{lastu}, mean correct-target NLL at response positions1..4 then EOS is: {diag['correct_target_nll_by_response_position']}. The training objective averages these five positions; this explains why a small response/EOS loss is not equivalent to perfect selection. It does not establish that the averaging rule is causally responsible for the failure.",'']
for n,v in diag['family_profiles'].items():lines.append(f"- {n}: {v['correct']}/{v['n']} correct; exact decoded response frequencies {v['generated_text_counts']}.")
lines+=['',
'These are measurements on already-seen training records. They do not establish heldout semantic transfer or distinguish copying the sole name from factual understanding. No confirmed implementation defect emerged: next-token alignment, identities, schedules, parameter scopes and frozen optimizer inactivity were checked. The run hit a scientific cost guard, not a mechanical execution failure.','',
'The next boundary is language retention while teaching the single-fact interface. Adding language rehearsal, changing factual exposure or learning rate, or changing response weighting would be scientifically different interventions. The present result does not uniquely choose among them, so no corrective training is authorized by the mechanical-defect exception. The 200-update endpoint was not reached; this result cannot show what unchanged continued training would have done. Gates were not weakened to find out.','',
'## Checkpoint identities','']
for u,c in cps.items():lines += [f"- Update{u}: `{c['path']}`; SHA256 `{c['sha256']}`."]
lines+=['','Parent hash and every frozen payload checksum were reverified after completion. No mechanical execution corrections were required. The permanent checkpoints and one rolling restart retain provenance; no history was overwritten.']
(O/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n');shutil.copyfile(Path(__file__),O/'REPORT_SOURCE.py')
files=sorted(q for q in O.iterdir() if q.is_file());(O/'OUTPUT_SHA256SUMS.txt').write_text(''.join(sha(q)+'  '+q.name+'\n' for q in files),encoding='utf-8',newline='\n')
receipt={'status':s['status'],'bundle_receipt':sha(B/'RECEIPT.json'),'manifest':sha(O/'OUTPUT_SHA256SUMS.txt'),'report':sha(O/'REPORT.md'),'checkpoints':cps,'parent_unchanged':True,'final_access':False,'sacred_access':False}
(O/'OUTPUT_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');(O/'OUTPUT_RECEIPT.sha256').write_text(sha(O/'OUTPUT_RECEIPT.json')+'  OUTPUT_RECEIPT.json\n',encoding='utf-8')
for q in O.iterdir():
 if q.is_file():q.chmod(0o444)
assert sha(O/'OUTPUT_RECEIPT.json')==(O/'OUTPUT_RECEIPT.sha256').read_text().split()[0]
assert sha(O/'OUTPUT_SHA256SUMS.txt')==read(O/'OUTPUT_RECEIPT.json')['manifest']
for line in (O/'OUTPUT_SHA256SUMS.txt').read_text().splitlines():h,n=line.split('  ',1);assert sha(O/n)==h
print(json.dumps({'status':s,'receipt':sha(O/'OUTPUT_RECEIPT.json'),'manifest':sha(O/'OUTPUT_SHA256SUMS.txt'),'checkpoints':cps},indent=2))
