import sys,json
from pathlib import Path
import torch
ROOT=Path(r'C:\DaveLM-CADAVER');sys.path[:0]=[str(ROOT),r'C:\DaveLM-v0.9']
from treatment13_model import Treatment13Model
from fact_supervision_87001_corrected_v8.PINNED_PILOT1_BINDING_IMPLEMENTATION import OrthoLocalizer,binding_eval
refs=json.loads((ROOT/'fact_supervision_87001_corrected_v8/BINDING_REFERENCES.json').read_text())
def load(p,d):
 sd=dict(torch.load(p,map_location=d,weights_only=True)['model_state_dict']); vals=[sd.pop('localizer.'+k) for k in ('u','q','bs','ba')];m=Treatment13Model();m.load_state_dict(sd,strict=False);m.localizer=OrthoLocalizer(*vals);return m.to(d)
def main():
 d=torch.device('cuda' if torch.cuda.is_available() else 'cpu');cks={'P5':ROOT/'language_sentencebound_p5/latest.pt','P7':ROOT/'language_compositional_p7/latest.pt','Pilot1':ROOT/'language_pilot_1_early_block_protection_seed8380/pilot_run/checkpoints/seed_8380/latest.pt'};o={}
 for n,p in cks.items():
  m=load(p,d);o[n]={}
  for k in ('pilot0_dev','pilot1_dev'):
   qs=json.loads(Path(refs[k]['path']).read_text())['quartets'];docs=[x for q in qs for x in q['docs']];z=binding_eval(m,docs,d);o[n][k]=z['overall']
 print(json.dumps(o,indent=2));(ROOT/'human_readiness_binding_measurement.json').write_text(json.dumps(o,indent=2))
main()
