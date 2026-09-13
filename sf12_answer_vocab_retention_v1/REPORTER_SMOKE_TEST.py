"""Outcome-blind synthetic tests for frozen SF12 classification logic."""
import json
from pathlib import Path
import REPORTER as R

def row(status='ACQUISITION_FAIL',u=200,d3=True,ret=True,lang=True,bind=True,end=False,s=12,o=8,gain=True):
    return {'status':status,'completed':u,'gates':{'retention_acquisition':ret,'language':lang,'binding':{'p0':bind,'p1':bind},'d3':d3,'continue':ret and lang and bind and d3,'endpoint_pass':end,'development_gain':gain},'dev_surface':{'exact':s},'dev_order':{'exact':o}}

def main():
    assert R.classify([row(status='ACQUISITION_SUCCESS',end=True)]*2+[row(d3=False,u=100)])=='ANSWER_VOCAB_RETENTION_SUPPORTED_FULL_ENDPOINT'
    assert R.classify([row(),row(),row(d3=False,u=100)])=='ANSWER_VOCAB_RETENTION_SUPPORTED_PARTIAL_WIDENING'
    assert R.classify([row(d3=False,u=100),row(d3=False,u=100),row()])=='ANSWER_VOCAB_RETENTION_WEAKENED'
    assert R.classify([row(s=10,o=6),row(s=10,o=6),row(d3=False,u=100)])=='WIDENING_STALLED_UNDER_RETENTION'
    assert R.classify([row(ret=False,u=100),row(),row()])=='RETENTION_REGRESSION'
    assert R.classify([row(),row()])=='INCONCLUSIVE_EARLY_STOP'
    (Path(__file__).resolve().parent/'REPORTER_SMOKE.json').write_text(
        json.dumps({'status':'PASS','synthetic_cases':6,'outcome_data_used':False},indent=2)+'\n',
        encoding='utf-8',newline='\n')
    print('SF12_REPORTER_SMOKE_PASS')

if __name__=='__main__': main()
