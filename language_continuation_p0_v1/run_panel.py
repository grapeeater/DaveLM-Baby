from pathlib import Path
import json,sys
sys.path.insert(0,r'C:\DaveLM-CADAVER')
from language_continuation_p0_v1 import load,gen
from tokenizers import Tokenizer
PROMPTS=[
 'Once upon a time, there was a little girl named Zoe.',
 'Once upon a time, there was a little boy named Alex.',
 'Once upon a time, there was a little girl named Nora.',
 'Once upon a time, there was a little boy named Owen.',
 'A cat found a blue ball.',
 'A dog found a red hat.',
 'Mia went to the park with her mother.',
 'Leo saw a bird in the yard.',
 'The little girl dropped her book.',
 'The little boy wanted to play outside.',
 'Sam was sad because his toy broke.',
 'Lily gave the ball to Tom.',
]
def main():
 out=Path(__file__).resolve().parent; tok=Tokenizer.from_file(r'C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json'); m=load().to('cuda:0'); rows=[{'prompt':p,'greedy_48':gen(m,tok,p,48)} for p in PROMPTS]; (out/'HELDOUT_PANEL.json').write_text(json.dumps({'prompts_frozen_before_scoring':PROMPTS,'rows':rows},indent=2,ensure_ascii=False),encoding='utf-8'); print(json.dumps(rows,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
