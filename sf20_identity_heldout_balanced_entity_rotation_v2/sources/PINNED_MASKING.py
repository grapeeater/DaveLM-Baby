"""Reusable scoring/masking/scope/journal helpers; no checkpoint loading or training entrypoint."""
import json, os
from pathlib import Path

def prepare_example(row):
    target = row['candidates'][row['correct_index']] if row['arm'] == 'factual' else row['practice_target']
    c = row['candidate_token_ids'][row['candidates'].index(target)]
    prefix = [2] + row['prompt_token_ids']
    seq = prefix + c + [3]
    return seq[:-1], [-100] * (len(prefix)-1) + c + [3]

def pad_batch(rows, shared_length):
    import torch
    x = torch.zeros((len(rows), shared_length), dtype=torch.long)
    y = torch.full_like(x, -100)
    for i, r in enumerate(rows):
        a, b = prepare_example(r)
        assert len(a) <= shared_length and len(a) == len(b)
        x[i, :len(a)] = torch.tensor(a)
        y[i, :len(b)] = torch.tensor(b)
    return x, y

def set_scope(model, binding):
    for name, p in model.named_parameters():
        p.grad = None
        active = binding or (name.startswith('base_model.') and not any(
            name.startswith('base_model.blocks.' + str(i) + '.') for i in range(4)))
        p.requires_grad_(active)

def candidate_scores(logits, prompt_length, candidate_ids):
    import torch
    # Input to the caller is BOS + P + C + EOS. Candidate 0 is predicted at index len(P).
    idx = torch.tensor(candidate_ids + [3])
    logp = logits[0, prompt_length:prompt_length+len(idx)].detach().to(device='cpu', dtype=torch.float64).log_softmax(-1)
    vals = logp[torch.arange(len(idx)), idx].tolist()
    return {'token_log_probabilities': vals, 'candidate_ll': sum(vals[:-1]),
            'word_only_ll': sum(vals[:-2]), 'eos_terminated_ll': sum(vals)}

def eos_diagnostic(logits):
    v = logits.detach().to(device='cpu', dtype=__import__('torch').float64)
    lp = v.log_softmax(-1)[3]
    return {'log_probability': float(lp), 'probability': float(lp.exp()),
            'rank': 1+int((v > v[3]).sum()), 'wins_argmax': int(v.argmax()) == 3}

class Journal:
    """Exclusive durable append with write-ahead records; uncertain work never silently repeats."""
    def __init__(self, path):
        self.path = Path(path)
        self.started, self.completed = set(), {}
        if self.path.exists():
            for line in self.path.read_text(encoding='utf-8').splitlines():
                rec = json.loads(line)
                if rec['event'] == 'STARTED':
                    assert rec['unit'] not in self.started
                    self.started.add(rec['unit'])
                else:
                    assert rec['event'] == 'COMPLETED' and rec['unit'] in self.started
                    assert rec['unit'] not in self.completed
                    self.completed[rec['unit']] = rec['result']
    def _append(self, rec):
        with self.path.open('ab') as f:
            f.write((json.dumps(rec, sort_keys=True, ensure_ascii=False, allow_nan=False)+'\n').encode('utf-8'))
            f.flush(); os.fsync(f.fileno())
    def start(self, unit):
        if self.started - self.completed.keys():
            raise RuntimeError('Uncertain in-flight unit: stop without replay')
        if unit in self.completed:
            return False
        self._append({'event': 'STARTED', 'unit': unit}); self.started.add(unit)
        return True
    def finish(self, unit, result):
        assert unit in self.started and unit not in self.completed
        self._append({'event': 'COMPLETED', 'unit': unit, 'result': result})
        self.completed[unit] = result
