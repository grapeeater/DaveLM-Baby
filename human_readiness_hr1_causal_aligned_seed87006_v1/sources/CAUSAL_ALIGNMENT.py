"""Frozen causal next-token batch construction for the corrected HR successor.

This module is deliberately independent of the model and optimizer.  Both the
training and DEV paths must use the same construction:

    z = [BOS] + content_tokens + [EOS]
    x = z[:-1]
    y = z[1:]

Only the y entries corresponding to z are supervised; batch padding is -100.
"""

BOS_ID = 2
EOS_ID = 3
IGNORE_INDEX = -100
PAD_INPUT_ID = 0


def sequence_pair(content_ids, pad_to):
    content = list(content_ids)
    z = [BOS_ID] + content + [EOS_ID]
    if pad_to < len(z) - 1:
        raise ValueError("pad_to is shorter than the input sequence")
    x = [PAD_INPUT_ID] * pad_to
    y = [IGNORE_INDEX] * pad_to
    x[: len(z) - 1] = z[:-1]
    y[: len(z) - 1] = z[1:]
    return x, y


def batch_pairs(records, pad_to=None):
    ids = [r["token_ids"] if isinstance(r, dict) else r for r in records]
    required = [len(v) + 2 for v in ids]
    width = max(required) if pad_to is None else int(pad_to)
    if width < max(required):
        raise ValueError("pad_to is shorter than a batch sequence")
    return [sequence_pair(v, width) for v in ids]


def supervised_targets(y):
    return [v for v in y if v != IGNORE_INDEX]


def validate_pair(content_ids, x, y):
    expected_x, expected_y = sequence_pair(content_ids, len(x))
    if list(x) != expected_x or list(y) != expected_y:
        raise AssertionError("causal pair mismatch")
    target = supervised_targets(y)
    if target != list(content_ids) + [EOS_ID]:
        raise AssertionError("targets do not contain next tokens plus final EOS")
