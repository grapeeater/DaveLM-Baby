import ast,hashlib,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent; SRC=H.parent/'sf12_answer_vocab_retention_v1'
def sha(p):
 h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
def main():
 assert sha(SRC/'FREEZE_RECEIPT.json')==(SRC/'FREEZE_RECEIPT.sha256').read_text().split()[0]
 r=json.loads((SRC/'FREEZE_RECEIPT.json').read_text()); assert sha(SRC/'SHA256SUMS.txt')==r['manifest_sha256']
 for line in (SRC/'SHA256SUMS.txt').read_text().splitlines(): hh,n=line.split('  ',1); assert sha(SRC/n)==hh,n
 led=json.loads((SRC/'RUN_LEDGER.json').read_text()); assert len(led)==3 and all(x['state']=='CLASSIFIED' and x['completed']==100 for x in led)
 sys.path.insert(0,str(H)); import REPORTER as R
 rows=[R.summarize(s) for s in [87029,87030,87031]]
 assert all(set(x['binding'])=={'pilot0','pilot1'} for x in rows)
 assert all(v['answer_exact']==80 and v['both_distinct']==80 and v['slot_collapse']==0 for x in rows for v in x['binding'].values())
 assert R.classify(rows)=='ANSWER_VOCAB_RETENTION_WEAKENED'
 assert all(not x['gates']['d3'] and x['gates']['retention_acquisition'] and x['gates']['language'] and all(x['gates']['binding'].values()) for x in rows)
 (H/'PREFLIGHT.json').write_text(json.dumps({'status':'PASS_MECHANICAL_REPORTING_HOTFIX','source_integrity_verified':True,'runs':3,'classification':'ANSWER_VOCAB_RETENTION_WEAKENED','model_loaded':False,'inference_run':False,'training_replayed':False},indent=2)+'\n',encoding='utf-8',newline='\n')
 print('SF12_REPORTING_HOTFIX_PREFLIGHT_PASS')
if __name__=='__main__': main()
