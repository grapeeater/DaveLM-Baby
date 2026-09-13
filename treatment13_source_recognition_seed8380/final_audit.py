import sys
from pathlib import Path
sys.path[:0]=[r'C:\DaveLM-CADAVER',r'C:\DaveLM-v0.9',r'C:\DaveLM-CADAVER\treatment13_distinct_localization_supervision_seed8380']
import treatment13_distinct_final_audit as audit
ROOT=Path(r'C:\DaveLM-CADAVER'); OUT=ROOT/'treatment13_source_recognition_seed8380'; SRC=ROOT/'treatment13_learned_mapping_row_localization_seed8380'
audit.OUT=OUT; audit.CK=OUT/'checkpoints/source_recognition/seed_8380/latest.pt'; audit.RET=SRC/'treatment13_retention_quartet_pool.json'; audit.OUTJSON=OUT/'FINAL_FROZEN_RETENTION_RESULTS.json'; audit.main()
