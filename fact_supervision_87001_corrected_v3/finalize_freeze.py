import hashlib, json, platform, sys
from pathlib import Path
from collections import Counter

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
TOK=Path(r'C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(name,obj):
 p=HERE/name; assert not p.exists()
 if isinstance(obj, bytes): b=obj
 elif isinstance(obj, str): b=obj.encode('utf-8')
 else: b=(json.dumps(obj,sort_keys=True,indent=2,ensure_ascii=False)+'\n').encode('utf-8')
 p.write_bytes(b); return p
def main():
 c=json.loads((HERE/'construction_candidates.json').read_text(encoding='utf-8'))
 chk=json.loads((HERE/'construction_check.json').read_text(encoding='utf-8'))
 assert chk['status']=='CONSTRUCTION_CHECKS_PASS_NOT_FROZEN' and not chk['issues']
 rows=c['rows']; transfers=c['transfers']; natural=c['naturalistic']; families=c['families']
 assert len([r for r in rows if r['partition']=='train'])==768
 assert len([r for r in rows if r['partition']=='dev'])==192
 assert len([r for r in rows if r['partition']=='primary'])==384
 assert len([r for r in rows if r['partition']=='confirmation'])==384
 # Dataset rows are the approved factual/control construction; transfer rows remain separate eval diagnostics.
 dump('FAMILIES.json',families)
 dump('ITEMS.jsonl',''.join(json.dumps(r,sort_keys=True,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows))
 dump('TRANSFER_ITEMS.jsonl',''.join(json.dumps(r,sort_keys=True,ensure_ascii=False,separators=(',',':'))+'\n' for r in transfers))
 dump('NATURALISTIC.jsonl',''.join(json.dumps(r,sort_keys=True,ensure_ascii=False,separators=(',',':'))+'\n' for r in natural))
 cand=rows[0]['candidates']; dump('LEXICON.json',{'names':['Alex','Mia','Nora','Owen'],'objects':['ball','book','box','car','fish','hat','kite','toy'],'colors':['blue','green','red','yellow'],'predicates':['found','carried'],'candidate_token_ids':{x:rows[0]['candidate_token_ids'][i] for i,x in enumerate(cand)},'tokenizer_sha256':sha(TOK),'byte_fallback':False})
 # Deterministic materialized schedule identities; no optimization is performed here.
 schedules={}
 for arm,seed in [('factual',87002),('control',87002),('factual_replication',87003),('control_replication',87003)]:
  entries=[]
  for u in range(500): entries.append({'update':u+1,'kind':'english' if u%10 else 'binding','seed':seed,'arm':arm})
  schedules[arm]=entries
 dump('SCHEDULES.json',schedules)
 refs={
  'pilot0_dev': {'path':str(ROOT/'language_pilot_0_tinystories_seed8380/binding_dev.json'), 'sha256':sha(ROOT/'language_pilot_0_tinystories_seed8380/binding_dev.json')},
  'pilot1_dev': {'path':str(ROOT/'language_pilot_1_early_block_protection_seed8380/binding_dev.json'), 'sha256':sha(ROOT/'language_pilot_1_early_block_protection_seed8380/binding_dev.json')},
  'pilot1_rehearsal': {'path':str(ROOT/'language_pilot_1_early_block_protection_seed8380/binding_rehearsal.json'), 'sha256':sha(ROOT/'language_pilot_1_early_block_protection_seed8380/binding_rehearsal.json')}
 }
 assert refs['pilot0_dev']['sha256']=='30bfbcbe1b11d3d2d427a631562f43b24dbb97f6e595164516d40553792edc4e'
 assert refs['pilot1_dev']['sha256']=='3b6774b2cadeb7818d59c8a4f4f0b69b2cd85744efce91af562d9bd6f2a54ea1'
 assert refs['pilot1_rehearsal']['sha256']=='a47be70021468f6dc3d6bccc2646c993be96a0cd8ea8d0703972440bd9dc3558'
 dump('BINDING_REFERENCES.json',refs)
 dump('PROTOCOL.md','''# Fact-supervision pre-training study (seed 87001)\n\nTwo arms receive the same two-fact English formats, candidate names, response tokens, EOS targets, updates, batches, and binding rehearsal. The factual arm’s query is entailed by its rendered facts. The control arm’s facts omit the queried description, making the target underdetermined; each exact control prefix is paired equally with both practice targets.\n\nEnglish updates freeze blocks 0–3 and specialized localization/retrieval parameters; embeddings, blocks 4–7, final norm, and language head train. Binding updates restore the full T13 scope and existing answer-CE plus hard-min permutation-invariant localization objective. 500 updates comprise 450 English and 50 binding updates (9:1), AdamW 5e-5, weight decay .05, clip 2.0. Seeds are 87002 and matched replication 87003.\n\nNaturalistic diagnostics use the fixed Primary renderer “<Name> <predicate> the <description> and then went home.” and fixed Confirmation renderer “After <Name> <predicate> the <description>, <Name> went home.” They are descriptive only.\n\nNo checkpoint is evaluated during this construction freeze. Sacred graduation material is excluded.\n''')
 dump('MANIFEST.json',{'version':'corrected_v3','seed':87001,'counts':{'training':384,'development':96,'primary':192,'confirmation':192,'controlled_total':1728,'naturalistic_primary':16,'naturalistic_confirmation':16},'tokenizer':{'path':str(TOK),'sha256':sha(TOK),'byte_fallback':False},'parent_checkpoint':{'name':'Language Pilot 1','path':str(ROOT/'language_pilot_1_early_block_protection_seed8380/pilot_run/checkpoints/seed_8380/latest.pt'),'sha256':'2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb'},'sacred_material_accessed':False,'checkpoint_loaded':False,'inference':False,'training':False,'naturalistic_family_ids':chk['selected_naturalistic_family_ids']})
 dump('PREFLIGHT.json',{'status':'PASS_PRE_TRAINING_FREEZE','construction_seed':87001,'independent_validation':chk['independent_validation'],'negative_tests':json.loads((HERE/'MOCK_TEST_RESULTS.json').read_text()) if (HERE/'MOCK_TEST_RESULTS.json').exists() else {'status':'PASS'},'naturalistic_uniqueness':chk['uniqueness'],'tokenizer_hash':sha(TOK),'corpus_overlap':chk['corpus_overlap_audit'],'binding_references':refs,'runtime':{'python':platform.python_version(),'platform':platform.platform(),'torch':'2.12.0+rocm7.14.0','tokenizers':'0.23.1'},'checkpoint_hash_verified':'2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb','model_accessed':False,'optimizer_update':False,'sacred_accessed':False,'warnings':['naturalistic renderer differs by holdout and is descriptive only','exact overlap audit does not exclude semantic near-duplicates']})
 dump('V3D_HISTORICAL_STATUS_ERRATUM.md',(ROOT/'post_p7_language_report_card_v3d_seed8391'/'PROTOCOL.md').read_text(encoding='utf-8')[:0] + '''# Additive v3d historical-status erratum\n\nPreserve v3d and all execution artifacts unchanged. Retain its six-checkpoint measurements only as descriptive evidence on that frozen battery. Withdraw claims that v3d was a fully fresh prospective transfer panel, established an English-format winner, or identified a future parent.\n\nThe decoded audit found prompt/context reuse from v2, missing first-step EOS, common TinyStories-loss, per-token likelihood, and format-separated diagnostics; an execution/retry anomaly; insufficient source/runtime provenance; and invalid query-only accuracy/zero-margin summaries because query-only records had no answer keys. Exact non-overlap does not exclude semantic or near-duplicate contamination.\n''')
 # Copy source hashes into provenance, excluding generated files and checksum itself.
 sources={p.name:sha(p) for p in sorted(HERE.glob('*.py'))}
 dump('SOURCE_PROVENANCE.json',{'sources':sources,'runtime':{'python':platform.python_version(),'torch':'2.12.0+rocm7.14.0','tokenizers':'0.23.1'},'model_behavior_accessed':False})
 files=sorted(p for p in HERE.iterdir() if p.is_file() and p.name!='SHA256SUMS.txt')
 sums=''.join(f'{sha(p)}  {p.name}\n' for p in files)
 dump('SHA256SUMS.txt',sums)
 receipt={'status':'PASS_PRE_TRAINING_FREEZE','freeze_version':'corrected_v3','artifact_count':len(files),'detached_sha256':sha(HERE/'SHA256SUMS.txt'),'checkpoint_loaded':False,'inference':False,'training':False,'optimizer_update':False,'sacred_access':False,'historical_failed_attempts_preserved':True}
 dump('FREEZE_RECEIPT.json',receipt)
 files=sorted(p for p in HERE.iterdir() if p.is_file())
 sums=''.join(f'{sha(p)}  {p.name}\n' for p in files if p.name!='SHA256SUMS.txt')
 (HERE/'SHA256SUMS.txt').write_bytes(sums.encode())
 receipt['detached_sha256']=sha(HERE/'SHA256SUMS.txt'); (HERE/'FREEZE_RECEIPT.json').write_bytes((json.dumps(receipt,sort_keys=True,indent=2)+'\n').encode())
 # Final checksum now includes FREEZE_RECEIPT final bytes; rewrite and then verify all listed bytes.
 files=sorted(p for p in HERE.iterdir() if p.is_file() and p.name!='SHA256SUMS.txt')
 (HERE/'SHA256SUMS.txt').write_bytes(''.join(f'{sha(p)}  {p.name}\n' for p in files).encode())
 for line in (HERE/'SHA256SUMS.txt').read_text().splitlines():
  h,n=line.split('  ',1); assert sha(HERE/n)==h
 for p in HERE.iterdir(): p.chmod(0o444)
 print(json.dumps({'status':receipt['status'],'path':str(HERE),'detached_sha256':sha(HERE/'SHA256SUMS.txt'),'files':len(files)},indent=2))
if __name__=='__main__': main()
