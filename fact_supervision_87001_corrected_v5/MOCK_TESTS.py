import json,collections
from pathlib import Path
d=Path(__file__).parent; s=json.loads((d/"MATERIALIZED_SCHEDULE_SEED87002.json").read_text()); u=s["updates"]; assert len(u)==500; assert [x["global_update"] for x in u if x["kind"]=="binding"]==list(range(10,501,10)); assert sum(x["kind"]=="english" for x in u)==450; assert sum(x["kind"]=="binding" for x in u)==50
for x in u:
 if x["kind"]=="english":
  assert x["shared_pad_to_length"]<=256
  for a in ("factual","control"): assert len(x["arms"][a]["item_ids"])==32
 else: assert len(x["quartet_ids"])==8 and len(x["document_ids"])==32
print("PASS schedule/masking-source/static persistence contract")
