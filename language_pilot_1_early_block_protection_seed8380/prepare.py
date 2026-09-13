from __future__ import annotations
import hashlib, json, random, shutil, sys
from pathlib import Path
from itertools import combinations

ROOT = Path(r"C:\DaveLM-CADAVER")
PILOT0 = ROOT / "language_pilot_0_tinystories_seed8380"
OUT = ROOT / "language_pilot_1_early_block_protection_seed8380"
TOK = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")
T13 = ROOT / "treatment13_learned_mapping_row_localization_seed8380"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, r"C:\DaveLM-v0.9")
from treatment10_common import build_member_doc, load_tokenizer
from treatment13_generate import load_key_value_pools
import experiments.two_mapping_contextual_binding.run as original_run

def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def collect_arrays(obj):
    out=[]
    if isinstance(obj, dict):
        for k,v in obj.items():
            if k in ("full_document_token_ids", "token_ids", "document_token_ids") and isinstance(v,list) and v and all(isinstance(x,int) for x in v): out.append(tuple(v))
            else: out.extend(collect_arrays(v))
    elif isinstance(obj, list):
        for v in obj: out.extend(collect_arrays(v))
    return out

def build_pool(tok, keys, values, filler_words, combos, per_combo, axis, seed_base, bos):
    qs=list(combinations(range(5),2)); vs=list(combinations(range(5),2)); quartets=[]; gid=0
    for ci,(prefix_len,between_len) in enumerate(combos):
        for local in range(per_combo):
            kp=qs[(ci+local)%len(qs)]; vp=vs[(ci*2+local)%len(vs)]; order=(ci+local)%2
            k0,k1=keys[kp[0]],keys[kp[1]]; v0,v1=values[vp[0]],values[vp[1]]
            rng=random.Random(seed_base+ci*1000+local*17)
            prefix=[filler_words[rng.randrange(len(filler_words))] for _ in range(prefix_len)]
            between=[filler_words[rng.randrange(len(filler_words))] for _ in range(between_len)]
            tail=[filler_words[rng.randrange(len(filler_words))] for _ in range(192)]
            line0,line1=(k0,k1) if order==0 else (k1,k0)
            def val(key_tok,orientation): return v0 if (orientation==1)==(key_tok==k0["token_id"]) else v1
            docs=[]
            for member in ("o1_k0","o1_k1","o2_k0","o2_k1"):
                orientation=1 if member.startswith("o1") else 2; query=k0 if member.endswith("k0") else k1
                lines=[]
                for lk in (line0,line1):
                    z=val(lk["token_id"],orientation); lines.append([lk["word"],z["word"]])
                ans=val(query["token_id"],orientation); raw=build_member_doc(tok,lines,query["word"],ans["word"],prefix,between,tail); ids=[bos]+raw
                if len(ids)!=193: raise RuntimeError("doc length")
                other=k1 if query["token_id"]==k0["token_id"] else k0; distract=val(other["token_id"],orientation)
                occ={name:[i for i,x in enumerate(ids) if x==tokv["token_id"]] for name,tokv in (("k0",k0),("k1",k1),("v0",v0),("v1",v1))}
                qname="k0" if query["token_id"]==k0["token_id"] else "k1"; oname="k1" if qname=="k0" else "k0"; tname="v0" if ans["token_id"]==v0["token_id"] else "v1"; dname="v1" if tname=="v0" else "v0"
                qdp=max(occ[qname]); ans_idx=max(occ[tname]); qcl=min(occ[qname]); tcl=min(occ[tname])
                docs.append({"doc_id":f"pilot1_{axis}:qt_{gid:06d}:{member}","quartet_id":f"pilot1_{axis}:qt_{gid:06d}","member":member,"axis":f"pilot1_{axis}","orientation":orientation,"query_key_word":query["word"],"query_key_token":query["token_id"],"target_value_word":ans["word"],"target_value_token":ans["token_id"],"distractor_value_token":distract["token_id"],"distractor_value_word":distract["word"],"candidate_pair_sorted":sorted([v0["token_id"],v1["token_id"]]),"query_slot":0 if line0["token_id"]==query["token_id"] else 1,"mapping_order":order,"qdp":qdp,"query_key_clause_pos":qcl,"target_clause_pos":tcl,"answer_token_index":ans_idx,"answer_causal_position":ans_idx-1,"distractor_value_pos":occ[dname][0],"other_key_pos":occ[oname][0],"full_document_token_ids":ids,"layout_combo":f"p{prefix_len}_b{between_len}"})
            quartets.append({"quartet_id":f"pilot1_{axis}:qt_{gid:06d}","axis":f"pilot1_{axis}","layout_combo":f"p{prefix_len}_b{between_len}","key_pair_tokens":sorted([k0["token_id"],k1["token_id"]]),"value_pair_tokens":sorted([v0["token_id"],v1["token_id"]]),"docs":docs}); gid+=1
    return quartets

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ("language_train.jsonl","language_dev.jsonl"):
        shutil.copyfile(PILOT0/name, OUT/name)
    tok=load_tokenizer(); groups=original_run._identity_pool(tok); filler_words=[str(x["value"]) for x in groups["filler"]]; keys,vals=load_key_value_pools(); bos=int(tok.token_to_id("<bos>"))
    rehearsal=build_pool(tok,keys,vals,filler_words,[(40,4),(42,5),(44,6),(46,7),(48,8),(50,9),(52,10),(54,11)],10,"rehearsal",1700000000,bos)
    dev=build_pool(tok,keys,vals,filler_words,[(56,12),(58,13)],10,"dev",2700000000,bos)
    (OUT/"binding_rehearsal.json").write_text(json.dumps({"artifact_type":"pilot1_binding_rehearsal","quartets":rehearsal},indent=1),encoding="utf-8")
    (OUT/"binding_dev.json").write_text(json.dumps({"artifact_type":"pilot1_binding_dev","quartets":dev},indent=1),encoding="utf-8")
    new_docs=[d for q in rehearsal+dev for d in q["docs"]]; new_arr={tuple(d["full_document_token_ids"]) for d in new_docs}
    if len(new_docs)!=400 or len(new_arr)!=400: raise RuntimeError("new pool duplicate/size")
    prior_files=[]
    for p in ROOT.rglob("*.json"):
        if p.is_relative_to(OUT): continue
        if "pool" in p.name.lower(): prior_files.append(p)
    prior_files += [PILOT0/"binding_rehearsal.json", PILOT0/"binding_dev.json"]
    prior={}; overlaps={}
    for p in sorted(set(prior_files)):
        try: arr={tuple(x) for x in collect_arrays(json.loads(p.read_text(encoding="utf-8")))}
        except Exception: continue
        if arr:
            prior[str(p)]=len(arr); ov=new_arr & arr
            if ov: overlaps[str(p)]=len(ov)
    if overlaps: raise RuntimeError(f"token-array overlap: {overlaps}")
    lang_arr=set()
    for name in ("language_train.jsonl","language_dev.jsonl"):
        for line in (OUT/name).read_text(encoding="utf-8").splitlines(): lang_arr.add(tuple(json.loads(line)["token_ids"]))
    manifest={"pilot":"Language Pilot 1","seed":8380,"source_manifest":str(PILOT0/"CORPUS_MANIFEST.json"),"tokenizer_path":str(TOK),"tokenizer_sha256":sha(TOK),"language_train_sha256":sha(OUT/"language_train.jsonl"),"language_dev_sha256":sha(OUT/"language_dev.jsonl"),"binding_rehearsal_sha256":sha(OUT/"binding_rehearsal.json"),"binding_dev_sha256":sha(OUT/"binding_dev.json"),"binding_namespace":"pilot1_*","generator":"existing T13 grammar/build_member_doc; new deterministic seed bases rehearsal=1700000000, dev=2700000000","rehearsal_quartets":80,"rehearsal_documents":320,"dev_quartets":20,"dev_documents":80,"layouts_rehearsal":["p40_b4","p42_b5","p44_b6","p46_b7","p48_b8","p50_b9","p52_b10","p54_b11"],"layouts_dev":["p56_b12","p58_b13"],"prior_pool_files_checked":sorted(prior),"prior_token_array_overlaps":overlaps,"language_exact_array_overlap":len(new_arr & lang_arr),"sacred_behavior_evaluated":False,"sacred_pool_used_only_for_token_array_disjointness":True}
    (OUT/"PILOT1_MATERIAL_MANIFEST.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    (OUT/"PREPARATION_RESULT.json").write_text(json.dumps({"status":"MATERIAL_PASS","manifest":manifest},indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2))
if __name__=="__main__": main()
