import json
from pathlib import Path
import REPORTER as R
def row(end=False,d3=True,ret=True,s=8,o=5,status='ACQUISITION_FAIL'):
 return {'status':status,'gates':{'endpoint_pass':end,'d3':d3,'retention_acquisition':ret,'language':ret,'binding':{'a':ret,'b':ret}},'widening_movement':s>0 and o>0}
def main():
 assert R.classify([row(True),row(True),row(False,d3=False,status='STOP_REGRESSION')])=='FIRST_TOKEN_CE_INTERPOLATION_SUPPORTED'
 assert R.classify([row(d3=False,status='STOP_REGRESSION'),row(d3=False,status='STOP_REGRESSION'),row()])=='FIRST_TOKEN_CE_INTERPOLATION_INSUFFICIENT'
 assert R.classify([row(ret=False,status='STOP_REGRESSION'),row(),row()])=='RETENTION_REGRESSION'
 assert R.classify([row(),row()])=='MECHANICAL_INCOMPLETE'
 # SF14-like exact-generation failure with safe D3 and intact retention.
 z=[row(),row(),row()]
 for r in z[:2]: r['terminal']={'surface':{'exact':2},'order':{'exact':2}}
 z[2]['terminal']={'surface':{'exact':6},'order':{'exact':6}}
 assert R.classify(z)=='FIRST_TOKEN_CE_INTERPOLATION_INSUFFICIENT'
 assert R.classify([row(),row()])=='MECHANICAL_INCOMPLETE'
 (Path(__file__).resolve().parent/'REPORTER_SMOKE.json').write_text(json.dumps({'status':'PASS','cases':5,'outcomes_used':False},indent=2)+'\n',encoding='utf-8',newline='\n'); print('SF16_REPORTER_SMOKE_PASS')
if __name__=='__main__': main()
