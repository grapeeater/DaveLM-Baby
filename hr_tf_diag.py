import json, math, sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(r'C:\\DaveLM-CADAVER')
sys.path[:0] = [str(ROOT), r'C:\\DaveLM-v0.9']
from treatment13_model import Treatment13Model
from fact_supervision_87001_eval_v1.PINNED_PILOT1_BINDING_IMPLEMENTATION import OrthoLocalizer

TOK = __import__('tokenizers').Tokenizer.from_file(r'C:\\DaveLM-v0.9\\tokenizer\\v0_7\\davelm_tokenizer.json')
CK = {
    'Pilot1': ROOT/'language_pilot_1_early_block_protection_seed8380/pilot_run/checkpoints/seed_8380/latest.pt',
    'HR1': ROOT/'human_readiness_hr1_seed87004_v8/run/checkpoint_500.pt',
    'HR2': ROOT/'human_readiness_hr2_seed87005_v7/run/checkpoint_500.pt',
}
DEV = ROOT/'human_readiness_hr1_seed87004_v8/ENGLISH_DEV.jsonl'
DEVICE = 'cuda'

def load(path):
    sd = dict(torch.load(path, map_location=DEVICE, weights_only=True)['model_state_dict'])
    v = [sd.pop('localizer.'+k) for k in ('u','q','bs','ba')]
    m = Treatment13Model()
    m.load_state_dict(sd, strict=False)
    m.localizer = OrthoLocalizer(*v)
    return m.to(DEVICE).eval()

def score_batch(logits, y):
    mask = y.ne(-100)
    ys, ls = y[mask], logits[mask]
    lp = ls.log_softmax(-1); probs = lp.exp()
    vals, ix = ls.max(-1); second = torch.topk(ls, 2, dim=-1).values[:,1]
    gold_lp = lp.gather(1, ys[:,None]).squeeze(1)
    entv = -(probs * lp.clamp_min(-30)).sum(-1)
    return {
        'n': int(ys.numel()), 'correct': int((ix == ys).sum()),
        'nll': float((-gold_lp).sum()), 'entropy': float(entv.sum()),
        'top1p': float(vals.softmax(0).sum()) if False else float(ls.log_softmax(-1).max(-1).values.exp().sum()),
        'margin': float((vals-second).sum()), 'goldp': float(gold_lp.exp().sum()),
    }

def main():
    rows = [json.loads(x) for x in DEV.read_text(encoding='utf-8').splitlines()[:128]]
    out = {'dataset': str(DEV), 'rows': len(rows), 'models': {}}
    for name, path in CK.items():
        m = load(path)
        # Report both the correctly aligned causal objective and the exact shifted
        # objective used by the historical HR1/HR2 DEV helper/training loop.
        totals = {'aligned': {'n':0,'correct':0,'nll':0.,'entropy':0.,'top1p':0.,'margin':0.,'goldp':0.},
                  'shifted_copy': {'n':0,'correct':0,'nll':0.,'entropy':0.,'top1p':0.,'margin':0.,'goldp':0.}}
        with torch.inference_mode():
            for j in range(0, len(rows), 32):
                rs = rows[j:j+32]; zlist = [[2] + r['token_ids'] + [3] for r in rs]
                L = max(len(z) for z in zlist)
                # Correct causal alignment: x=z[:-1], y=z[1:] at the same position.
                xa = torch.zeros((len(rs), L-1), dtype=torch.long, device=DEVICE)
                ya = torch.full((len(rs), L-1), -100, dtype=torch.long, device=DEVICE)
                # Historical alignment: x starts at 0 while y starts at 1; this
                # trains/predicts token identity at positions 1..n and EOS from pad.
                xs = torch.zeros((len(rs), L), dtype=torch.long, device=DEVICE)
                ys = torch.full((len(rs), L), -100, dtype=torch.long, device=DEVICE)
                for i,z in enumerate(zlist):
                    xa[i,:len(z)-1] = torch.tensor(z[:-1], device=DEVICE)
                    ya[i,:len(z)-1] = torch.tensor(z[1:], device=DEVICE)
                    xs[i,:len(z)-1] = torch.tensor(z[:-1], device=DEVICE)
                    ys[i,1:len(z)] = torch.tensor(z[1:], device=DEVICE)
                a = score_batch(m.base_model(xa), ya)
                s = score_batch(m.base_model(xs), ys)
                for key,val in a.items(): totals['aligned'][key] += val
                for key,val in s.items(): totals['shifted_copy'][key] += val
        model_out = {'checkpoint': str(path)}
        for mode,t in totals.items():
            n=t['n']; model_out[mode] = {
                'token_targets': n, 'mean_loss': t['nll']/n, 'ppl': math.exp(t['nll']/n),
                'teacher_forced_top1_accuracy': t['correct']/n,
                'mean_top1_probability': t['top1p']/n,
                'mean_top1_margin_logit': t['margin']/n,
                'mean_gold_probability': t['goldp']/n,
                'mean_entropy': t['entropy']/n,
            }
        out['models'][name] = model_out
    (ROOT/'generation_pathology_teacher_forced.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
