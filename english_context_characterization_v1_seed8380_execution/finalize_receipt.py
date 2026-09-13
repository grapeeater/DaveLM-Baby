from pathlib import Path
import hashlib, json

out = Path(__file__).resolve().parent
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return {'bytes':p.stat().st_size,'sha256':h.hexdigest()}

files = ['RAW_SCORES.jsonl','FAMILY_RESULTS.csv','BINDING_REFERENCE_RESULTS.json',
         'REPORT.md','SUMMARY.json','VERIFICATION.json','POST_VERIFICATION.json',
         'RUNTIME_AND_LOAD_RECEIPT.json','EXECUTION_STATUS.json']
checks = {n:digest(out/n) for n in files}
receipt = {
  'status':'FROZEN_ENGLISH_CONTEXT_CHARACTERIZATION_COMPLETE',
  'protocol_id':'english_context_characterization_v1_seed8380',
  'frozen_detached_sha256sumstxt_sha256':'26cf767af6f85738c52e506f6dcfe2a34fbcce40a9b2829101408038cfa81e1b',
  'checkpoint_sha256_after': {
    'graduate':'fed298748e62def6f1751ef1bb7df0cdec51d53fac6e4c86e8c00a95dccdd430',
    'pilot0':'769bd01efd28e7888064c0d5fe1dd0d85a4344e2039aef04bcbf7d9e906f47f5',
    'pilot1':'2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb'},
  'english_comparisons_per_checkpoint':1040,
  'contextual_prompts_per_checkpoint':832,
  'logical_prior_comparisons_per_checkpoint':208,
  'binding_documents_per_checkpoint':160,
  'english_path':'model.base_model(input_ids)',
  'specialized_path_calls':0,
  'generation':False,'training':False,'optimizer_created':False,'backward':False,
  'sacred_exam_opened':False,'frozen_artifacts_modified':False,
  'mechanical_corrections':[
    'Runner accepted verifier status PASS_EXECUTION_INPUT_VERIFICATION (interface-label correction only).',
    'Report aggregator mapped sealed frame name frame_holdout (internal reserve shorthand mismatch only).'],
  'outputs':checks,
  'output_manifest_excludes_self_hash':True}
(out/'OUTPUT_MANIFEST.json').write_bytes((json.dumps(receipt,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode())
lines=[]
for n,v in checks.items(): lines.append(f"{v['sha256']}  {n}")
(out/'OUTPUT_SHA256SUMS.txt').write_bytes(('\n'.join(lines)+'\n').encode())
print(json.dumps(receipt,indent=2))
