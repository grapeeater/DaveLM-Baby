# Frozen v4 trainer. This source consumes only the literal schedule; it is not executed during freeze.
import os,random,json,hashlib,tempfile
from pathlib import Path
import torch
import torch.nn.functional as F
from harness import prepare_example,pad_batch
from PINNED_PILOT1_BINDING_IMPLEMENTATION import rowpos,loc_loss,LAM
def scope(model,kind):
 for n,p in model.named_parameters():
  p.grad=None; p.requires_grad_(kind=="binding" or (n.startswith("base_model.") and not any(n.startswith(f"base_model.blocks.{i}.") for i in range(4))))
def english_loss(model,rows,pad):
 x,y=pad_batch(rows,pad); z=model.base_model(x); return F.cross_entropy(z.reshape(-1,z.size(-1)),y.reshape(-1),ignore_index=-100)
def atomic_state(path,state):
 tmp=path.with_suffix(path.suffix+".tmp"); torch.save(state,tmp)
 with open(tmp,"rb") as f: os.fsync(f.fileno())
 os.replace(tmp,path)
def step(model,opt,kind,batch):
 opt.zero_grad(set_to_none=True); scope(model,kind)
 if kind=="english": loss=english_loss(model,batch["rows"],batch["pad"])
 else:
  docs=batch["docs"]; x=torch.tensor([d["full_document_token_ids"] for d in docs],device=next(model.parameters()).device); q=torch.tensor([d["qdp"] for d in docs],device=x.device); a=torch.tensor([d["answer_causal_position"] for d in docs],device=x.device); y=torch.tensor([d["target_value_token"] for d in docs],device=x.device); out,extra=model(x,q,a); loss=F.cross_entropy(out[torch.arange(len(docs),device=x.device),a],y)+LAM*loc_loss(extra["localization_attention"],docs)
 loss.backward(); active=[p for p in model.parameters() if p.grad is not None]; torch.nn.utils.clip_grad_norm_(active,2.0); opt.step(); return float(loss)
