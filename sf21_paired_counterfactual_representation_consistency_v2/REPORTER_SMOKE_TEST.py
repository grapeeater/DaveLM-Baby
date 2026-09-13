import json
from pathlib import Path
import REPORTER as R
def row(end=False,ret=True,status='ACQUISITION_FAIL',completed=200):
 return {'status':status,'completed':completed,'gates':{'endpoint_pass':end,'retention_acquisition':ret,'language':ret,'binding':{'p0':ret,'p1':ret}}}
def main():
 assert R.classify([row(True),row(True),row()])=='SF21_COEXISTENCE_FRONTIER_SUCCESS'
 assert R.classify([row(),row(),row()])=='SF21_COEXISTENCE_FRONTIER_FAIL_10M_SERIES_CLOSED'
 assert R.classify([row(ret=False),row(),row()])=='RETENTION_REGRESSION_10M_SERIES_CLOSED'
 assert R.classify([row(),row()])=='MECHANICAL_INCOMPLETE'
 out={'status':'PASS','cases':4,'outcomes_used':False,'series_stop_logic_frozen':True}
 (Path(__file__).resolve().parent/'REPORTER_SMOKE.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8',newline='\n')
 print('SF21_REPORTER_SMOKE_PASS')
if __name__=='__main__': main()
