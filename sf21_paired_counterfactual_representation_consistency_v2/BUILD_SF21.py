"""Build the prospective SF21 bundle from sealed SF20 v2. No model load/training."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import hashlib, json, shutil

ROOT=Path(r"C:\DaveLM-CADAVER")
SRC=ROOT/"sf20_identity_heldout_balanced_entity_rotation_v2"
DST=ROOT/"sf21_paired_counterfactual_representation_consistency_v2"

def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def write(p,x): Path(p).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding="utf-8",newline="\n")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert not DST.exists()
    # Verify authoritative sealed SF20 scientific payload.
    assert sha(SRC/"FREEZE_RECEIPT.json") == (SRC/"FREEZE_RECEIPT.sha256").read_text().split()[0]
    receipt=read(SRC/"FREEZE_RECEIPT.json")
    assert receipt["status"]=="SF20_PROSPECTIVE_PREFLIGHT_PASS"
    assert sha(SRC/"SHA256SUMS.txt")==receipt["manifest_sha256"]
    for line in (SRC/"SHA256SUMS.txt").read_text().splitlines():
        h,n=line.split("  ",1); assert sha(SRC/n)==h,n
    DST.mkdir()
    names=["D3_SELECTION.json","DEV_ORDER.json","DEV_SURFACE.json","EXTERNAL_INPUTS.json",
      "FORBIDDEN_IDENTITIES.json","IDENTITY_ASSIGNMENT.json","KL_POOL_MANIFEST.json","KL_POOL.json",
      "NAME_CENSUS.json","SCHEDULE.json","SF13_KL_SCHEDULE.json","SF2_ENGINE.py","SF2_PROTOCOL.json",
      "TOKENIZATION_MATCH.json","TRAIN.json","TRAIN16_RETENTION.json"]
    for n in names: shutil.copy2(SRC/n,DST/n)
    shutil.copytree(SRC/"data",DST/"data"); shutil.copytree(SRC/"sources",DST/"sources")

    rows=read(DST/"TRAIN.json"); groups=defaultdict(list)
    for r in rows:
        if r["id"].startswith("SF20:"): groups[r["pair_id"]].append(r)
    assert len(groups)==16 and all(len(v)==2 for v in groups.values())
    pairs=[]
    for key in sorted(groups):
        a,b=sorted(groups[key],key=lambda x:x["correct_index"])
        assert a["correct_index"]==0 and b["correct_index"]==1
        assert a["candidates"]==b["candidates"] and a["subgroup"]==b["subgroup"]
        assert a["predicate"]==b["predicate"] and a["object"]==b["object"]
        ids_a=[x.strip().rstrip(".") for x in a["candidates"]]
        ids_b=[x.strip().rstrip(".") for x in b["candidates"]]
        assert ids_a==ids_b and a["actor"]!=b["actor"]
        pairs.append({"pair_id":key,"record_a":a["id"],"record_b":b["id"],
          "structural_group":{"subgroup":a["subgroup"],"predicate":a["predicate"],"object":a["object"]},
          "identities_a":ids_a,"identities_b":ids_b,
          "correct_answer_a":a["candidates"][a["correct_index"]],
          "correct_answer_b":b["candidates"][b["correct_index"]],
          "representation_position_a":len(a["prompt_token_ids"]),
          "representation_position_b":len(b["prompt_token_ids"]),
          "position_semantics":"zero-based BOS-prefixed final prompt/query token; its logits predict first answer token",
          "identity_assignment_differs":True})
    write(DST/"PAIR_MANIFEST.json",{"rule":"Pair the two assignment-reversal records sharing each frozen SF20 pair_id; every widening record occurs exactly once.","pairs":pairs})

    p=read(SRC/"PROTOCOL.json")
    p.update(study="SF21_PAIRED_COUNTERFACTUAL_REPRESENTATION_CONSISTENCY_V2",
      created_utc=datetime.now(timezone.utc).isoformat(),
      predecessor=str(SRC), predecessor_receipt_sha256=sha(SRC/"FREEZE_RECEIPT.json"),
      hypothesis="A 0.1 cosine-consistency penalty between final-normalized pre-answer query states of each frozen identity-assignment reversal pair may teach identity-independent relational solution geometry while retaining exact identity output.",
      sole_scientific_variable="Add 0.1 times mean (1-cosine) across the 16 frozen widening reversal pairs at the final-normalized pre-answer query state on English updates.",
      english_objective="Byte-equivalent SF20 CE + KL160 + 0.25 margin M=1, plus lambda_consistency=0.1 paired pre-answer query-state cosine consistency. No other auxiliary objective.",
      seeds=[87056,87057,87058], lambda_consistency=0.1,
      representation_target="base_model.final_norm output at BOS-prefixed index len(prompt_token_ids), the last context/query state whose language-head logits predict first answer token; answer-token states are not compared.",
      pair_rule="16 assignment-reversal pairs by shared frozen SF20 pair_id; all 32 widening records exactly once; no negatives.",
      classification={"success":"SF21_COEXISTENCE_FRONTIER_SUCCESS if >=2/3 endpoint_pass",
        "retention":"RETENTION_REGRESSION if any frozen TRAIN16/language/binding retention gate fails",
        "failure":"SF21_COEXISTENCE_FRONTIER_FAIL_10M_SERIES_CLOSED for any other scientifically valid cross-seed result",
        "mechanical":"MECHANICAL_INCOMPLETE if integrity/execution prevents classification"},
      prospective_series_stop="If SF21 does not achieve coexistence in >=2/3 branches, close the 10.6M factual-supervision series without SF22 or coefficient rescue.")
    newruns=[]
    for seed,old in zip(p["seeds"],[87053,87054,87055]):
        source=next(x for x in read(SRC/"PROTOCOL.json")["runs"] if x["seed"]==old)
        r=dict(source); r["seed"]=seed; r["sf20_lineage_seed"]=old; newruns.append(r)
    p["runs"]=newruns
    write(DST/"PROTOCOL.json",p)
    write(DST/"PROVENANCE.json",{"study":p["study"],"authoritative_predecessor":str(SRC),
      "predecessor_receipt_sha256":sha(SRC/"FREEZE_RECEIPT.json"),"predecessor_manifest_sha256":sha(SRC/"SHA256SUMS.txt"),
      "copied_payloads":{n:sha(DST/n) for n in names},"pair_manifest_sha256":sha(DST/"PAIR_MANIFEST.json"),
      "scientific_change":"paired counterfactual representation consistency only","final_accessed":False,"sacred_accessed":False})

    s=(SRC/"CONTROLLER.py").read_text(encoding="utf-8")
    s=s.replace("SF20", "SF21")
    s=s.replace("identity-heldout balanced entity rotation with SF13 objective", "paired counterfactual representation consistency on SF20")
    s=s.replace("ONE scientific variable vs SF13: widening identity set. Full CE, broad KL160, margin,\noptimizer, scope, schedule, gates, binding, evaluation and persistence remain unchanged.",
      "ONE scientific variable vs SF20: add 0.1 paired cosine consistency at the final-normalized\npre-answer query state. All data, CE, KL, margin, optimizer, scope, schedule and gates are unchanged.")
    s=s.replace("LAMBDA_MARGIN = 0.25\n", "LAMBDA_MARGIN = 0.25\nLAMBDA_CONSISTENCY = 0.1\n")
    anchor="def verify(sealed=True):\n"
    helper='''def paired_consistency(hidden, ids_in_batch, idx, pairs):\n    positions = {rid: k for k, rid in enumerate(ids_in_batch)}\n    losses, cosines = [], []\n    for pair in pairs:\n        a, b = pair['record_a'], pair['record_b']\n        if a not in positions or b not in positions:\n            continue\n        ia, ib = positions[a], positions[b]\n        pa, pb = len(idx[a]['prompt_token_ids']), len(idx[b]['prompt_token_ids'])\n        va = F.normalize(hidden[ia, pa].float(), dim=-1)\n        vb = F.normalize(hidden[ib, pb].float(), dim=-1)\n        cosine = (va * vb).sum()\n        cosines.append(cosine); losses.append(1.0 - cosine)\n    assert len(losses) == 16, len(losses)\n    return torch.stack(losses).mean(), torch.stack(cosines).mean()\n\n\n'''
    assert anchor in s; s=s.replace(anchor,helper+anchor,1)
    old="""            logits = m.base_model(x)\n            ce = F.cross_entropy(logits.reshape(-1, 1024), y.reshape(-1), ignore_index=-100)\n            loss = ce + E.LAMBDA_KL * kl_val\n            margin_loss = margin_hinge_term(logits, u['ids'], idx)\n            loss = loss + LAMBDA_MARGIN * margin_loss\n"""
    new="""            captured = {}\n            def capture_final_norm(_module, _inputs, output): captured['hidden'] = output\n            handle = m.base_model.final_norm.register_forward_hook(capture_final_norm)\n            try: logits = m.base_model(x)\n            finally: handle.remove()\n            hidden = captured.pop('hidden')\n            ce = F.cross_entropy(logits.reshape(-1, 1024), y.reshape(-1), ignore_index=-100)\n            loss = ce + E.LAMBDA_KL * kl_val\n            margin_loss = margin_hinge_term(logits, u['ids'], idx)\n            consistency_loss, pair_cosine = paired_consistency(hidden, u['ids'], idx, read(H/'PAIR_MANIFEST.json')['pairs'])\n            loss = loss + LAMBDA_MARGIN * margin_loss + LAMBDA_CONSISTENCY * consistency_loss\n"""
    assert s.count(old)==1; s=s.replace(old,new)
    oldrec="if margin_loss is not None: rec.update(margin_loss=float(margin_loss.detach()))"
    newrec="if margin_loss is not None: rec.update(margin_loss=float(margin_loss.detach()), consistency_loss=float(consistency_loss.detach()), weighted_consistency=float((LAMBDA_CONSISTENCY*consistency_loss).detach()), mean_pair_cosine=float(pair_cosine.detach()))"
    assert s.count(oldrec)==1; s=s.replace(oldrec,newrec)
    (DST/"CONTROLLER.py").write_text(s,encoding="utf-8",newline="\n")
    print(DST)

if __name__=="__main__": main()
