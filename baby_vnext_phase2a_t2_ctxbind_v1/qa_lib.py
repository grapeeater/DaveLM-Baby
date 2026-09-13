"""Phase 2A corpus generation library (deterministic).

Every episode is simulated: item ownership state is updated by transfers and the
correct answer is the simulated final owner. Generators emit exactly ORBIT
episodes per family (8 for curriculum levels, 4 for the audit slice) so panel
quotas are filled with whole families.
"""
from __future__ import annotations

import hashlib
import itertools
import random
from dataclasses import dataclass, field

from tokenizers import Tokenizer

BOS = 2
EOS = 3
PAD = 0

NAMES_F = ["Mia", "Lily", "Sara", "Ann", "Zoe", "Ivy"]
NAMES_M = ["Tom", "Ben", "Noah", "Sam", "Max", "Owen"]
ALL_NAMES = NAMES_F + NAMES_M
GENDER = {n: ("f" if n in NAMES_F else "m") for n in ALL_NAMES}

COLORS = ["red", "blue", "green", "yellow"]
OBJECTS = ["ball", "book", "box", "car", "kite", "hat"]
PHRASES = [f"{c} {o}" for c in COLORS for o in OBJECTS]

PROP_VERBS = ["has", "owns"]
TRANS_VERBS = ["gives", "hands", "passes"]
Q_VERBS = ["has", "has got"]

ORBIT = {"A": 8, "B": 8, "C": 8, "D": 8, "E": 8, "F": 8, "G": 8, "H": 8, "AUD": 4}
CANDIDATES = {"A": 2, "B": 2, "C": 4, "D": 2, "E": 4, "F": 2, "G": 4, "H": 3, "AUD": 4}
LEVELS = ["A", "B", "C", "D", "E", "F", "G", "H"]


@dataclass
class Episode:
    kind: str
    family_id: str
    partition: str
    prompt_text: str
    candidate_names: list
    correct_name: str
    correct_index: int
    phrase_queried: str
    style: dict
    is_audit: bool = False
    prompt_token_ids: list = field(default_factory=list)
    candidate_token_ids: list = field(default_factory=list)
    prompt_len: int = 0
    answer_len: int = 0


def _rng(family_id: str, salt: int) -> random.Random:
    h = hashlib.sha256(f"{family_id}:{salt}".encode("utf-8")).hexdigest()
    return random.Random(int(h[:14], 16))


def _pron(name: str) -> str:
    return "She" if GENDER[name] == "f" else "He"


def _candidate_order(names_seen: list) -> list:
    seen = []
    for n in names_seen:
        if n not in seen:
            seen.append(n)
    return seen


def _mk(tokenizer, kind, family_id, partition, names_seen, answer, query,
        sentences, qtext, style, is_audit=False) -> Episode:
    cand = _candidate_order(names_seen)
    ep = Episode(
        kind=kind, family_id=family_id, partition=partition,
        prompt_text=" ".join(sentences) + " " + qtext,
        candidate_names=cand, correct_name=answer,
        correct_index=cand.index(answer), phrase_queried=query,
        style=style, is_audit=is_audit,
    )
    ep.prompt_token_ids = tokenizer.encode(ep.prompt_text).ids
    ep.candidate_token_ids = [tokenizer.encode(f" {n}.").ids for n in cand]
    ep.prompt_len = len(ep.prompt_token_ids)
    ep.answer_len = len(ep.candidate_token_ids[ep.correct_index])
    assert ep.correct_index < len(cand)
    return ep


def _s_prop(name, phrase, verb="has"):
    return f"{name} {verb} a {phrase}."


def _s_trans(name, name2, phrase, verb="gives"):
    return f"{name} {verb} the {phrase} to {name2}."


# ---- Level generators (exactly ORBIT episodes) ---------------------------
def gen_A(fid, names, items, partition, tok):
    """Two owners, one distinct colored object each; no transfer."""
    a, b = names
    p1, p2 = items
    out = []
    for assign in (0, 1):
        o1 = p1 if assign == 0 else p2
        o2 = p2 if assign == 0 else p1
        for q in (0, 1):
            query = o1 if q == 0 else o2
            answer = (a if assign == 0 else b) if q == 0 else (b if assign == 0 else a)
            s1, s2 = _s_prop(a, o1), _s_prop(b, o2)
            for order in (0, 1):
                sents = [s1, s2] if order == 0 else [s2, s1]
                out.append(_mk(tok, "A", fid, partition, [a, b], answer, query,
                               sents, f"Who has the {query}?",
                               {"assign": assign, "q": q, "order": order}))
    return out


def gen_B(fid, names, items, partition, tok):
    """Ownership transfer: giver owns two objects, transfers one."""
    g0, r0 = names
    px, py = items
    out = []
    for assign in (0, 1):
        giver, recv = (g0, r0) if assign == 0 else (r0, g0)
        for q in (0, 1):
            if q == 0:
                query, answer = px, recv
            else:
                query, answer = py, giver
            s1 = _s_prop(giver, px)
            s2 = _s_prop(giver, py)
            s3 = _s_trans(giver, recv, px)
            for order in (0, 1):
                sents = [s1, s2, s3] if order == 0 else [s2, s1, s3]
                out.append(_mk(tok, "B", fid, partition, [giver, recv], answer, query,
                               sents, f"Who has the {query} now?",
                               {"assign": assign, "q": q, "order": order}))
    return out


def _c_sentences(perm, names, items, transfer, distractor):
    a, b, c, d = names
    i1, i2, i3 = items
    o1, o2, o3 = perm  # owners of i1,i2,i3
    s = [_s_prop(o1, i1), _s_prop(o2, i2), _s_prop(o3, i3)]
    if transfer:
        s.append(f"{d} walks to the park.")
        s.append(_s_trans(o1, d, i1))
        s.append(f"{o2} talks to {d}.")
    else:
        s.append(f"{d} plays outside.")
        s.append(f"{o2} sings a song.")
    return s, (o1, o2, o3)


def gen_C(fid, names, items, partition, tok, transfer=True, shuffle=False, kind="C"):
    """Three owners + one distractor/receiver; 4 candidates. Orbit 8 balanced a/b/c/d."""
    perms = list(itertools.permutations(names[:3]))
    special = {0, 5}  # (a,b,c),(c,b,a) get the i1 (-> d) query too
    out = []
    rng = _rng(fid, 3)
    for pi, perm in enumerate(perms):
        sents, owners = _c_sentences(perm, names, items, transfer, True)
        o1, o2, o3 = owners
        if pi in special:
            for q in (0, 1):
                if q == 0:
                    query, answer = items[0], names[3]
                else:
                    query, answer = items[1], o2
                ss = list(sents)
                if shuffle:
                    rng.shuffle(ss)
                out.append(_mk(tok, kind, fid, partition, list(names), answer, query, ss,
                               f"Who has the {query} now?", {"perm": pi, "q": q}))
        else:
            query, answer = items[1], o2
            ss = list(sents)
            if shuffle:
                rng.shuffle(ss)
            out.append(_mk(tok, kind, fid, partition, list(names), answer, query, ss,
                           f"Who has the {query} now?", {"perm": pi, "q": 1}))
    assert len(out) == 8, len(out)
    return out


def gen_D(fid, names, items, partition, tok):
    """Pronoun/reference: actor picks up object, gives it away or keeps it.
    mention=0 uses a pronoun for the actor; mention=1 repeats the name."""
    a, b = names
    obj = items[0]  # full colored phrase, e.g. 'red ball'
    out = []
    for assign in (0, 1):
        actor, other = (a, b) if assign == 0 else (b, a)
        for q in (0, 1):
            if q == 0:
                answer = other
            else:
                answer = actor
            for mention in (0, 1):
                subj = _pron(actor) if mention == 0 else actor
                if q == 0:
                    s2 = f"{subj} gave it to {other}."
                else:
                    s2 = f"{subj} kept it."
                s1 = f"{actor} picked up the {obj}."
                out.append(_mk(tok, "D", fid, partition, [actor, other], answer, obj,
                               [s1, s2], f"Who has the {obj} now?",
                               {"assign": assign, "q": q, "mention": mention}))
    assert len(out) == 8
    return out


def gen_E(fid, names, items, partition, tok):
    """Order/position robustness: C-worlds under full sentence shuffles."""
    return gen_C(fid, names, items, partition, tok, transfer=True, shuffle=True, kind="E")


def gen_F(fid, names, items, partition, tok):
    """Paraphrase robustness (2 names): owns/has, gives/hands/passes, has/has got."""
    g0, r0 = names
    px, py = items
    out = []
    for variant in (0, 1):
        prop_v = PROP_VERBS[variant]
        tr_v = TRANS_VERBS[variant]
        q_v = Q_VERBS[variant]
        for assign in (0, 1):
            giver, recv = (g0, r0) if assign == 0 else (r0, g0)
            for q in (0, 1):
                if q == 0:
                    query, answer = px, recv
                    sents = [_s_prop(giver, px, prop_v), _s_prop(giver, py, prop_v),
                             _s_trans(giver, recv, px, tr_v)]
                    qtext = f"Who {q_v} the {query} now?"
                else:
                    query, answer = py, giver
                    sents = [_s_prop(giver, px, prop_v), _s_prop(giver, py, prop_v)]
                    qtext = f"Who {q_v} the {query}?"
                out.append(_mk(tok, "F", fid, partition, [giver, recv], answer, query,
                               sents, qtext, {"variant": variant, "assign": assign, "q": q}))
    assert len(out) == 8
    return out


def gen_G(fid, names, items, partition, tok):
    """Counterfactual identity swaps: cyclic ownership rotation over 4 owners/items."""
    n4 = names[:4]
    i4 = items[:4]
    out = []
    rng = _rng(fid, 7)
    for rot in range(4):
        owners = n4[rot:] + n4[:rot]
        for qi in (0, 1):
            query, answer = i4[qi], owners[qi]
            sents = [_s_prop(owners[k], i4[k]) for k in range(4)]
            if rng.random() < 0.5:
                sents = [sents[1], sents[0], sents[2], sents[3]]
            out.append(_mk(tok, "G", fid, partition, list(n4), answer, query, sents,
                           f"Who has the {query}?", {"rot": rot, "qi": qi}))
    assert len(out) == 8
    return out


def gen_H(fid, names, items, partition, tok):
    """Multi-step state tracking over 3 names: none / one / two transfer variants."""
    x, y, z = names[:3]
    ph = items[0]
    out = []
    plans = []
    for init in (x, y, z):
        plans.append(("none", init, None, None, init))
    two = [(x, y, z), (y, z, x), (z, x, y)]
    for p1, p2, p3 in two:
        plans.append(("two", p1, p2, p3, p3))
    plans.append(("one", y, x, None, x))
    plans.append(("one", z, y, None, y))
    assert len(plans) == 8
    for kind, p1, p2, p3, answer in plans:
        sents = [f"{p1} has the {ph}."]
        if kind == "none":
            pass
        elif kind == "one":
            sents.append(f"{p1} gives the {ph} to {p2}.")
        else:
            sents.append(f"{p1} gives the {ph} to {p2}.")
            sents.append(f"{p2} gives the {ph} to {p3}.")
        # ensure all three candidate names are present in the passage
        mentioned = set()
        for s in sents:
            for n in (x, y, z):
                if n in s:
                    mentioned.add(n)
        for n in (x, y, z):
            if n not in mentioned:
                sents.append(f"{n} came to the park.")
        out.append(_mk(tok, "H", fid, partition, list(names[:3]), answer, ph, sents,
                       f"Who has the {ph} now?", {"kind": kind}))
    return out


def gen_AUD(fid, names, items, partition, tok):
    """Shortcut-audit (4 names, 4 items, chance 0.25): each name is the correct
    answer in one of four episodes; the queried owner is never the last entity
    mentioned (recency trap), and a neutral by another entity ends the passage."""
    n4 = names[:4]
    i4 = items[:4]
    out = []
    rng = _rng(fid, 9)
    for qi in range(4):
        answer = n4[qi]
        query = i4[qi]
        sents = [_s_prop(n4[k], i4[k]) for k in range(4)]
        # one non-answer transfer to muddy state without touching the queried item
        other = i4[(qi + 1) % 4]
        mover = n4[(qi + 1) % 4]
        receiver = n4[(qi + 2) % 4]
        sents.append(_s_trans(mover, receiver, other))
        # neutral by a different name ends the passage (recency trap)
        neutral_names = [n for n in n4 if n != answer]
        rng.shuffle(neutral_names)
        sents.append(f"{neutral_names[0]} walks to the park.")
        out.append(_mk(tok, "AUD", fid, partition, list(n4), answer, query, sents,
                       f"Who has the {query} now?", {"qi": qi}, is_audit=True))
    assert len(out) == 4
    return out


GENERATORS = {
    "A": gen_A, "B": gen_B, "C": gen_C, "D": gen_D, "E": gen_E,
    "F": gen_F, "G": gen_G, "H": gen_H, "AUD": gen_AUD,
}


def family_id(kind: str, names, items, extra=None) -> str:
    base = f"{kind}|{','.join(names)}|{','.join(items)}"
    if extra is not None:
        base += f"|{extra}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()
