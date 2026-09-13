import json,hashlib
from pathlib import Path
d=Path('post_p7_v3d_stage1_execution_retry'); s=json.loads((d/'SUMMARY.json').read_text()); b=json.loads((d/'BINDING_REFERENCE_RESULTS.json').read_text())
L=['# Stage 1 v3d evaluation','\nOne authorized execution across six checkpoints and two separate nonsacred binding DEV pools.','\n## English controlled results','']
for k,v in s.items():
 L += [f'### {k}','|section|n|correct|ties|mean margin|','|---|---:|---:|---:|---:|']
 for sec,x in v.items(): L.append(f"|{sec}|{x['n']}|{x['correct']}|{x['ties']}|{x['mean_margin']:.6g}|")
 L.append('')
L += ['## Binding results','']
for k,refs in b.items():
 L.append(f'### {k}')
 for ref,x in refs.items():
  o=x['overall']; L.append(f"- {ref}: answer {o['answer_exact']}/80 ({o['answer_accuracy']:.3f}); BOTH_DISTINCT {o['BD']}/80; collapse {o['collapse']}/80; complete quartets {x['complete_quartets']}/20; reversal both-correct {x['reversal_both_correct']}/40; query-row|BD {x['queried_row_given_BD']['correct']}/{x['queried_row_given_BD']['n']}; answer|BD {x['answer_given_BD']['correct']}/{x['answer_given_BD']['n']}")
(d/'REPORT.md').write_text('\n'.join(L)+'\n',encoding='utf8')
files=sorted(d.iterdir()); (d/'OUTPUT_MANIFEST.json').write_text(json.dumps({'artifacts':{p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in files}},indent=2)+'\n')
files=sorted(d.iterdir()); (d/'SHA256SUMS.txt').write_text(''.join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in files),encoding='utf8')
print(hashlib.sha256((d/'SHA256SUMS.txt').read_bytes()).hexdigest())
