from __future__ import annotations
import hashlib,json,random,re,shutil
from pathlib import Path
from tokenizers import Tokenizer

ROOT=Path(r"C:\DaveLM-CADAVER")
OUT=ROOT/"post_p7_language_report_card_v1_seed8380"
TOK_PATH=Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")
P7=ROOT/"language_compositional_p7"
TRAIN=ROOT/"language_pilot_0_tinystories_seed8380"/"language_train.jsonl"
DEV=ROOT/"language_pilot_0_tinystories_seed8380"/"language_dev.jsonl"
SEED=8380

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def norm(s): return re.sub(r"\s+"," ",s.casefold()).strip()
def enc(tok,s): return tok.encode(s).ids
def family_rows(stage, n, pairs, r):
 allf=sorted(pairs); chosen=sorted(r.sample(allf,n)); return [(stage,f"{stage}:f{i:03d}",x) for i,x in enumerate(chosen)]

def main():
 OUT.mkdir(exist_ok=True)
 tok=Tokenizer.from_file(str(TOK_PATH))
 names=["Ben","Sam","Tim","Tom"]
 objects=["ball","book","box","car","fish","hat","kite","toy"]
 colors=["blue","green","red","yellow"]
 actions=["found","saw","carried","played with","looked for","opened","shared","saved"]
 # P7 training text is used only as a contamination reference; no checkpoint is loaded.
 p7_text=(P7/"TRAIN_TEXT.txt").read_text(encoding="utf-8")
 train_text=TRAIN.read_text(encoding="utf-8")
 dev_text=DEV.read_text(encoding="utf-8")
 p7_norm=norm(p7_text); corpus_norm=norm(train_text+"\n"+dev_text)
 pairs_near=sorted((a,b,o0,o1,c) for a in names for b in names if a<b for o0 in objects for o1 in objects if o0<o1 for c in colors)
 rng=random.Random(SEED)
 near=family_rows("near_distribution",12,pairs_near,rng)
 pairs_cf=sorted((a,b,o0,o1,c) for a in names for b in names if a<b for o0 in objects for o1 in objects if o0<o1 for c in colors)
 cf=family_rows("counterfactual",8,pairs_cf,rng)
 surf=family_rows("surface_form",8,pairs_cf,rng)
 dist=family_rows("distractor",8,pairs_cf,rng)
 families=[]; items=[]; natural=[]
 def add_family(stage,fid,tup,kind):
  a,b,o0,o1,col=tup; fam={"family_id":fid,"section":stage,"subject_pair":[a,b],"object_pair":[o0,o1],"color":col,"construction":kind}; families.append(fam)
  # assignment, query, order are all crossed; assignment 0 means a->object, 1 means b->object.
  for assignment in (0,1):
   subject=b if assignment else a
   other=a if assignment else b
   for query in (0,1):
    for order in (0,1):
     target=subject if query==0 else other
     primary_o=o1 if assignment else o0
     secondary_o=o0 if assignment else o1
     first=(subject if order==0 else other)
     second=(other if order==0 else subject)
     if kind=="near":
      facts=[f"{first} found the {col} {primary_o}.",f"{second} carried the {col} {secondary_o}."]
      q=f"Who found the {col} {primary_o}?" if query==0 else f"Who carried the {col} {secondary_o}?"
     elif kind=="counterfactual":
      facts=[f"{first} quickly picked up the {col} {primary_o}.",f"{second} left the {col} {secondary_o} on the table."]
      q=f"Who picked up the {col} {primary_o}?" if query==0 else f"Who left the {col} {secondary_o} on the table?"
     elif kind=="surface":
      facts=[f"The {col} {primary_o} was found by {first}.",f"The {col} {secondary_o} was carried by {second}."]
      q=f"The person who found the {col} {primary_o} was" if query==0 else f"The person who carried the {col} {secondary_o} was"
     else:
      facts=[f"{first} found the {col} {primary_o}.",f"{second} carried the {col} {secondary_o}.",f"{other} watched a small bird outside."]
      q=f"Who found the {col} {primary_o}?" if query==0 else f"Who carried the {col} {secondary_o}?"
     prompt=" ".join(facts[:2])+"\n"+q if kind!="distractor" else " ".join(facts)+"\n"+q
     cand=[f" {a}.",f" {b}."]
     correct=0 if target==a else 1
     items.append({"item_id":f"{fid}:a{assignment}q{query}o{order}","section":stage,"family_id":fid,"assignment":assignment,"query_index":query,"fact_order":order,"prompt":prompt,"candidates":cand,"candidate_token_ids":[enc(tok,x) for x in cand],"correct_index":correct,"expected_subject":target,"object_pair":[primary_o,secondary_o],"color":col,"role":"controlled"})
 for stage,rows,kind in [("near_distribution",near,"near"),("counterfactual",cf,"counterfactual"),("surface_form",surf,"surface"),("distractor",dist,"distractor")]:
  for _,fid,t in rows:add_family(stage,fid,t,kind)
 natural_prompts=[
  "One afternoon, Lily noticed a red ball near the tree.","Mia opened the old book and found a surprise.","Sam carried a blue kite across the yard.","Tom shared his lunch with Nora.","Alex heard a bird outside the window.","Zoe looked for her missing hat.","Owen built a small box for the toy.","Nora saw dark clouds above the park.","Lily dropped her green ball beside the bench.","Mia followed the sound of music down the hall.","Sam found a tiny fish in the shallow stream.","Tom helped Alex fix the broken car.","Zoe brought a yellow kite to the beach.","Owen placed the book beside his bed.","Nora waited quietly for her friend.","Alex noticed that the room was getting dark.","Lily heard a knock at the door.","Mia picked up the toy and smiled.","Sam walked home after the rain stopped.","Tom gave Nora a warm blanket.","Zoe watched the little bird fly away.","Owen found a new path through the garden.","Nora carried the box carefully upstairs.","Alex decided to share the red ball."]
 for i,p in enumerate(natural_prompts):natural.append({"item_id":f"naturalistic:n{i:03d}","section":"naturalistic","prompt":p,"role":"generation","max_new_tokens":32})
 # lexical checks and item integrity
 for word in names+objects+colors+actions:
  ids=enc(tok," "+word); assert ids and tok.decode(ids)==" "+word and all(i<1024 for i in ids)
 cand_lens={tuple(len(x) for x in it["candidate_token_ids"]) for it in items}; assert cand_lens=={(3,3)}
 assert len(items)==288 and len(families)==36 and len(natural)==24
 prompts=[it["prompt"] for it in items+natural]
 if len(prompts)!=len(set(prompts)):
  seen={}; dup=[]
  for it in items+natural:
   p=it["prompt"]
   if p in seen: dup.append((seen[p],it["item_id"],p))
   else: seen[p]=it["item_id"]
  raise AssertionError("duplicate prompts: "+repr(dup[:3]))
 # The new battery must not repeat P7 training examples or TinyStories exact strings.
 atomic=[]; overlap=[]
 for it in items+natural:
  s=norm(it["prompt"]); atomic.append(s)
  if s in p7_norm or s in corpus_norm: overlap.append({"item_id":it["item_id"],"source":"p7_or_tinystories"})
 assert not overlap
 # Each controlled family has eight members, balanced assignment/query/order and answer roles.
 balance={}
 for fam in families:
  rows=[x for x in items if x["family_id"]==fam["family_id"]]; assert len(rows)==8
  for key in ("assignment","query_index","fact_order","correct_index"): balance[f"{fam['family_id']}:{key}"]={str(k):sum(x[key]==k for x in rows) for k in (0,1)}; assert set(balance[f"{fam['family_id']}:{key}"].values())=={4}
 # Write all bytes explicitly as UTF-8/LF.
 protocol='''# Post-P7 language report card\n\nProtocol v1.0, construction seed 8380. This is a prospective behavioral battery; no Baby checkpoint was loaded or evaluated during construction.\n\n## Sections\n\n- Near-distribution compositional transfer: 12 families × 8 counterbalanced items = 96 items. Familiar names, objects, colors, and active clauses are recombined; the article and paired fact structure differ from P7 training text.\n- Counterfactual/reversal control: 8 families × 8 items = 64 items. Assignment, query, and fact order are crossed; both opposed assignments must be answered correctly.\n- Surface-form transfer: 8 families × 8 items = 64 items. Passive facts and a paraphrased query frame are evaluated separately.\n- Distractor/interference diagnostic: 8 families × 8 items = 64 items. An irrelevant third-entity sentence is inserted; this is diagnostic only.\n- Naturalistic transfer: 24 fixed story-start prompts, generation-only, reported separately.\n\n## Controlled scoring\n\nScore each candidate completion as the full conditional log-likelihood of the leading-space name plus period. The item is correct iff the correct-minus-incorrect margin is strictly positive; ties are incorrect. Report item accuracy, ties, family complete success (all eight members), reversal success (both assignments at fixed query/order), and raw margins. Primary controlled endpoints are complete-family proportion and mean within-family reversal success, reported separately by section. No prior subtraction or calibration is used.\n\nNaturalistic prompts use greedy generation with a maximum of 32 new tokens and stop at EOS or the first complete sentence. Human review is required for grammatical completeness, subject/reference consistency, relation appropriateness, repetition, contradiction, and truncation; reviewers must score every prompt using the frozen rubric and may not select examples after viewing outputs.\n\n## Controls and interpretation\n\nEvery candidate is correct and incorrect equally often; assignment, query, and fact order are balanced within every family. Full prompt duplication is rejected. Exact-string non-overlap is checked against P7 TRAIN_TEXT and TinyStories TRAIN/DEV under case-folded whitespace normalization; this does not exclude semantic or near-duplicate contamination. Controlled success establishes only the measured constructions. Naturalistic failure does not erase controlled success, and controlled success does not establish broad English competence. No sacred binding material is used.\n'''
 manifest={"protocol_id":"post_p7_language_report_card_v1_seed8380","seed":SEED,"tokenizer":{"path":str(TOK_PATH),"sha256":sha(TOK_PATH),"byte_fallback":False},"parent_milestone":{"path":str(P7/"latest.pt"),"sha256":sha(P7/"latest.pt")},"sections":{"near_distribution":{"families":12,"items":96},"counterfactual":{"families":8,"items":64},"surface_form":{"families":8,"items":64},"distractor":{"families":8,"items":64},"naturalistic":{"items":24}},"total_controlled_items":288,"total_items":312,"construction":{"python":"3.12","seed":SEED,"checkpoint_behavior_accessed":False,"model_evaluation_performed":False,"sacred_exam_accessed":False},"scoring":{"controlled":"full candidate sequence LL; margin>0; ties incorrect; family/reversal endpoints","naturalistic":"greedy max32; rubric review"}}
 lex={"names":names,"objects":objects,"colors":colors,"actions":actions,"tokenization":{w:{"leading_space_ids":enc(tok," "+w),"roundtrip":tok.decode(enc(tok," "+w))==" "+w} for w in sorted(set(names+objects+colors+actions))}}
 famout={"seed":SEED,"families":families,"balance":balance,"selection":"lexicographically sorted Cartesian tuples, Python Random(8380), sample without replacement per section"}
 preflight={"status":"PASS_PREEXECUTION_ONLY","tokenizer_sha256":sha(TOK_PATH),"checkpoint_behavior_accessed":False,"items":len(items),"controlled_items":len(items),"naturalistic_items":len(natural),"family_count":len(families),"eight_items_per_family":all(sum(x['family_id']==f['family_id'] for x in items)==8 for f in families),"unique_prompts":len(prompts)==len(set(prompts)),"candidate_lengths":sorted({len(x) for it in items for x in it['candidate_token_ids']}),"candidate_equal_lengths":cand_lens=={(3,3)},"corpus_overlap_count":len(overlap),"p7_train_text_sha256":sha(P7/'TRAIN_TEXT.txt'),"tinystories_train_sha256":sha(TRAIN),"tinystories_dev_sha256":sha(DEV),"no_model_loaded":True,"no_inference":True,"no_training":True,"notes":"Exact-string checks do not exclude semantic or near-duplicate contamination."}
 (OUT/"PROTOCOL.md").write_bytes(protocol.encode())
 for name,obj in [("MANIFEST.json",manifest),("LEXICON.json",lex),("FAMILIES.json",famout),("PREFLIGHT.json",preflight)]: (OUT/name).write_bytes((json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode())
 (OUT/"ITEMS.jsonl").write_bytes((''.join(json.dumps(x,ensure_ascii=False,separators=(",",":"))+"\n" for x in items+natural)).encode())
 receipt={"artifact_hashes":{p.name:{"bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(OUT.iterdir()) if p.is_file()},"status":"FROZEN_POST_P7_REPORT_CARD_READY","model_evaluation_performed":False,"checkpoint_behavior_accessed":False}
 (OUT/"RECEIPT.json").write_bytes((json.dumps(receipt,sort_keys=True,indent=2)+"\n").encode()); receipt["artifact_hashes"]["RECEIPT.json"]={"bytes":(OUT/"RECEIPT.json").stat().st_size,"sha256":sha(OUT/"RECEIPT.json")}
 (OUT/"SHA256SUMS.txt").write_bytes((''.join(f"{v['sha256']}  {k}\n" for k,v in sorted(receipt['artifact_hashes'].items()))).encode())
 print(json.dumps({"out":str(OUT),"status":preflight["status"],"items":len(items),"families":len(families),"tokenizer_sha256":sha(TOK_PATH),"checkpoint_sha256":manifest["parent_milestone"]["sha256"]},indent=2))
if __name__=='__main__':main()
