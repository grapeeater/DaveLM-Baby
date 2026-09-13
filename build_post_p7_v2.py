"""Prospective v2 builder; stdlib and tokenizer only. Never loads a checkpoint."""
import hashlib,itertools,json,random,re,stat
from pathlib import Path
from tokenizers import Tokenizer
from post_p7_v2_independent_validator import validate

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'post_p7_language_report_card_v2_seed8391'
SEED=8391
TOK=Path(r'C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm(s):return ' '.join(s.casefold().split())
def write(name,obj):
    data=obj if isinstance(obj,str) else json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2)+'\n'
    (OUT/name).write_bytes(data.encode('utf-8'))

def answer_from_rendered(prompt):
    # Builder: resolve a queried predicate by regex against final serialized sentences.
    body,q=prompt.split('\n')
    predicate=re.fullmatch(r'Who (.+)\?',q)
    predicate=predicate.group(1) if predicate else re.fullmatch(r'The person who (.+) was',q).group(1)
    verb,rest=predicate.split(' the ')
    patterns=[rf'(Sam|Tom) {re.escape(predicate)}\.',rf'The {re.escape(rest)} was {verb} by (Sam|Tom)\.']
    hits=[m for p in patterns for m in re.findall(p,body)]
    assert len(hits)==1
    return hits[0]

def strings(obj):
    if isinstance(obj,str):yield obj
    elif isinstance(obj,list):
        for x in obj:yield from strings(x)
    elif isinstance(obj,dict):
        for k,v in obj.items():
            if k not in ('token_ids','candidate_token_ids','prompt_token_ids'):yield from strings(v)

def main():
    assert not OUT.exists(),'Never overwrite a previous construction'
    tok=Tokenizer.from_file(str(TOK)); assert sha(TOK)=='e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b'
    objs=['ball','book','box','car','fish','hat','kite','toy']; cols=['blue','green','red','yellow']
    # Different colors on the two objects; no padding-only renaming of v1 prompts.
    universe=list(itertools.product(list(itertools.combinations(objs,2)),list(itertools.permutations(cols,2))))
    chosen=random.Random(SEED).sample(universe,20)
    families=[];items=[]
    for section,indices in [('near_distribution',range(12)),('counterfactual',range(12,20)),('surface_form',range(8)),('distractor',range(8))]:
        for idx in indices:
            (x,y),(c0,c1)=chosen[idx];fid=f'{section}:lex{idx:02d}'
            families.append({'family_id':fid,'section':section,'lexical_id':idx,'objects':[x,y],'colors':[c0,c1]})
            for a,q,o in itertools.product(range(2),repeat=3):
                names=['Sam','Tom'] if a==0 else ['Tom','Sam']
                facts=[f'{names[0]} found the {c0} {x}.',f'{names[1]} carried the {c1} {y}.']
                if section=='surface_form':facts=[f'The {c0} {x} was found by {names[0]}.',f'The {c1} {y} was carried by {names[1]}.']
                if o:facts.reverse()
                predicate=f'found the {c0} {x}' if q==0 else f'carried the {c1} {y}'
                query=f'The person who {predicate} was' if section=='surface_form' else f'Who {predicate}?'
                for d in (range(2) if section=='distractor' else [-1]):
                    rendered=facts[:]
                    if d!=-1:rendered.insert(0 if d==0 else len(rendered),'Mia watched a bird by the window.')
                    prompt=' '.join(rendered)+'\n'+query
                    answer=answer_from_rendered(prompt);candidates=[' Sam.',' Tom.']
                    items.append({'item_id':f'{fid}:a{a}q{q}o{o}d{d}','section':section,'family_id':fid,'assignment':a,'query':q,'fact_order':o,'distractor_position':d,'prompt':prompt,'candidates':candidates,'correct_index':candidates.index(' '+answer+'.'),'prompt_token_ids':tok.encode(prompt).ids,'candidate_token_ids':[tok.encode(c).ids for c in candidates]})
    natural=[
    'Sam carried a green book to the porch, then sat beside the door.',
    'Tom found a yellow hat under the chair and brushed off the dust.',
    'Lily saw a red kite caught in a low branch and reached toward it.',
    'Mia put a blue toy on the shelf so her little brother could see it.',
    'Sam looked for his red ball behind the sofa but it was not there.',
    'Tom opened a small box and saw his missing green car inside.',
    'Lily carried a yellow book into the kitchen to show her father.',
    'Mia found a blue hat on the steps and called to its owner.',
    'Rain began while Sam was outside, so he hurried toward the house.',
    'Tom was cold after his walk and reached for a dry coat.',
    'Lily could not reach the toy on the high shelf and asked for help.',
    'Mia heard her friend calling from the gate and turned around.',
    'Sam spilled water on the table and went to get a cloth.',
    'Tom noticed that his friend had no lunch and moved his plate closer.',
    'Lily saw that the plant was dry and filled a small cup with water.',
    'Mia was tired after playing outside and lay down on her bed.',
    'Sam left the green ball indoors before going out to play.',
    'Tom held the blue book while Lily carried the red box.',
    'Lily put her yellow hat beside the door before washing her hands.',
    'Mia gave the red toy to Sam and kept the blue kite.',
    'Sam closed the window because the rain was blowing inside.',
    'Tom found the path blocked by a fallen branch and stopped walking.',
    'Lily heard a soft sound from the box and carefully lifted the lid.',
    'Mia waited for the paint to dry before touching her new toy.'
    ]
    for i,p in enumerate(natural):items.append({'item_id':f'naturalistic:{i:02d}','section':'naturalistic','prompt':p,'prompt_token_ids':tok.encode(p).ids,'max_new_tokens':32})
    assert len(items)==376 and len({norm(i['prompt']) for i in items})==376
    independent=validate(items)
    lengths=[]
    for r in items:
        assert tok.decode(tok.encode(r['prompt']).ids)==r['prompt']
        assert all(0<=v<1024 for v in r['prompt_token_ids'])
        for c,ids in zip(r.get('candidates',[]),r.get('candidate_token_ids',[])):
            assert len(ids)==3 and tok.decode(ids)==c
            assert tok.encode(r['prompt']+c).ids==r['prompt_token_ids']+ids
            lengths.append(1+len(r['prompt_token_ids'])+len(ids))
        assert 1+len(r['prompt_token_ids'])+(32 if r['section']=='naturalistic' else 3)<=256
    # Relevant local corpora and language evaluations only; never enumerate binding exams.
    sources=[]
    for directory in sorted(ROOT.glob('language_*')):
        if directory.is_dir():
            for p in sorted(directory.iterdir()):
                if p.name in ('TRAIN_TEXT.txt','RESULTS.json','FIXED_PANEL_RESULTS.json','NATURAL_TRANSFER_RESULTS.json','HELDOUT_PANEL.json','SAMPLED_PANEL.json','language_train.jsonl','language_dev.jsonl'):sources.append(p)
    for dirname in ('post_p7_language_report_card_v1_seed8380','english_context_characterization_v1_seed8380'):
        sources.append(ROOT/dirname/'ITEMS.jsonl')
    sources.append(ROOT/'post_p7_language_report_card_v1_seed8380_execution_p7'/'RAW_SCORES.jsonl')
    audit=[];fail=[];lexfreq={}
    for p in sources:
        raw=p.read_text(encoding='utf-8-sig')
        if p.suffix=='.jsonl':texts=[s for l in raw.splitlines() for s in strings(json.loads(l))]
        elif p.suffix=='.json':texts=list(strings(json.loads(raw)))
        else:texts=[raw]
        normalized=[norm(s) for s in texts if s]
        joined='\x00'.join(normalized)
        overlap=[r['item_id'] for r in items if norm(r['prompt']) in joined]
        completed=[r['item_id'] for r in items for c in r.get('candidates',[]) if norm(r['prompt']+c) in joined]
        atomic=sum(any(norm(s) in joined for s in r['prompt'].split('\n')[0].split('. ') if s) for r in items)
        audit.append({'path':str(p),'sha256':sha(p),'prompt_overlap':overlap,'completed_overlap':completed,'items_with_any_fact_substring_overlap_diagnostic':atomic})
        fail+=overlap+completed
        if p.name in ('language_train.jsonl','language_dev.jsonl','TRAIN_TEXT.txt'):
            lexfreq[str(p)]={w:len(re.findall(r'\b'+re.escape(w.casefold())+r'\b',joined)) for w in ['Sam','Tom','Mia','found','carried']+objs+cols}
    assert not fail,fail
    OUT.mkdir()
    write('ITEMS.jsonl',''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in items))
    write('FAMILIES.json',families)
    write('LEXICON.json',{'subjects':['Sam','Tom'],'distractor_entity':'Mia','objects':objs,'colors':cols,'relations':['found','carried'],'candidate_ids':{' Sam.':tok.encode(' Sam.').ids,' Tom.':tok.encode(' Tom.').ids},'source_frequency':lexfreq})
    write('OVERLAP_AUDIT.json',{'normalization':'Unicode casefold then collapse whitespace; substring within decoded text fields; NUL separates records','sources':audit,'full_prompt_overlaps':0,'completed_sequence_overlaps':0,'limitation':'Sentence/phrase familiarity is allowed and reported. Exact nonoverlap does not prove semantic novelty. Corpus scope is listed local artifacts.'})
    write('PREFLIGHT.json',{'status':'PASS_NO_MODEL_ACCESS','independent':independent,'unique_prompts':376,'max_controlled_completed_length':max(lengths),'encode_decode_roundtrips':'PASS','prefix_completion_token_boundary':'PASS','equal_completion_tokens':3,'context_limit':256,'naturalistic_review':'24 prompts manually inspected before behavior; no internally conflicting facts or unresolved pronoun-dependent questions; multiple valid continuations intentionally allowed','no_inference':True,'no_training':True,'sacred_material_accessed':False})
    write('MANIFEST.json',{'version':2,'seed':SEED,'counts':{'near_distribution':96,'counterfactual':64,'surface_form':64,'distractor':128,'naturalistic':24},'controlled_items':352,'total_items':376,'families':36,'tokenizer':{'path':str(TOK),'sha256':sha(TOK),'byte_fallback':False},'p7_archive':{'path':str(ROOT/'archive/DAVELM_P7_MILESTONE_seed8380/davelm_p7_latest.pt'),'sha256':sha(ROOT/'archive/DAVELM_P7_MILESTONE_seed8380/davelm_p7_latest.pt')},'model_behavior_accessed':False})
    write('PROTOCOL.md',PROTOCOL)
    for script in (Path(__file__),ROOT/'post_p7_v2_independent_validator.py'):(OUT/script.name).write_bytes(script.read_bytes())
    # No circular checksum: manifest excludes itself; detached sums include manifest.
    files=sorted(OUT.iterdir());write('FREEZE_RECEIPT.json',{'status':'FROZEN_PRE_INFERENCE','artifacts':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in files}})
    files=sorted(OUT.iterdir());write('SHA256SUMS.txt',''.join(sha(p)+'  '+p.name+'\n' for p in files))
    for p in OUT.iterdir():p.chmod(p.stat().st_mode & ~stat.S_IWRITE)
    print(json.dumps({'path':str(OUT),'detached_sha256':sha(OUT/'SHA256SUMS.txt'),'preflight':independent,'max_length':max(lengths)},indent=2))

PROTOCOL='''# Prospective post-P7 language report card v2

Seed 8391. This new version supersedes v1 for future measurement; it does not repair
or reuse exposed v1 items. V1 controlled-transfer conclusions are withdrawn because
144 answer keys contradicted their prompts. Historical files remain preserved.

## Construction and scientific scope

Sam and Tom, both present in P7 training, are the matched candidate subjects.
Found and carried are the two factual relations. Objects and colors come from P7.
Sample 20 distinct object-pair/color-pair tuples from the sorted-by-construction
Cartesian universe using Python Random(8391). Full details and selected identities
are materialized in scripts, FAMILIES.json and ITEMS.jsonl. Sampling is performed
once before any model behavior. Distinct colors make these fresh factual contexts,
not v1 contexts with a new label or an irrelevant prefix.

Near-distribution: lexical families 0-11, 8 members each (96). This is familiar
vocabulary recombination with a two-fact QA demand, NOT a direct replication of
P7 single-prompt sentence generation. Counterfactual: families 12-19 (64), same
active frame, an independent lexical panel focused on reversal. Every controlled
section is fully counterfactual; there is no claim that these two sections isolate
different mechanisms. Surface: families 0-7 in passive wording and cloze queries
(64); both fact and query wording change, so causes cannot be separated.
Distractor: families 0-7 with a third named entity and irrelevant sentence placed
before OR after the two facts independently of all other factors (128).
Compare these diagnostics to matched near-distribution families 0-7 only.
Naturalistic: 24 separately fixed story starts, generation only, no candidate key.

Each base family crosses assignment x queried relation x intact fact order:
2 x 2 x 2 = 8. Distractor families also cross placement (16 members). Assignment
swaps agents only; object descriptions, predicates and query stay unchanged.
Query change selects the other fact. Order reversal changes only intact sentence
order, never fact meaning. Both candidates occur once in the relevant facts;
Mia is the distractor, never an answer. Correct agent, queried object, queried
action, relevant first/second mention, recency and assignment are balanced within
each family. Colors/object sampling is enumerated, not claimed population-balanced.
The builder resolves answers from rendered text. The independent validator uses
token-position grammar parsing, does not import builder logic, checks every key
and transformation, and rejects every deliberately inverted key.

## Frozen scoring and runtime for a later authorized execution

Use ordinary base causal LM only, eval/inference mode, BOS=2, EOS=3, float32 model,
no autocast or TF32, one sequence at a time without padding/truncation. Save device,
torch/tokenizers versions and model source hashes. Use float64 log_softmax on CPU
over float32 logits. With P=prompt tokens and C=candidate tokens, score
L(C|P)=sum_j log_softmax(logits[BOS,P,C] at position len(P)+j)[C_j], j=0..2.
Thus the first candidate is predicted at the final prompt position. Candidates
are exactly leading-space Sam/Tom plus period; each has three tokens. Do not score
BOS or append/score EOS. Do not length-normalize or subtract priors. Margin is
L(correct)-L(other); >0 correct, ==0 tie and incorrect. Preserve token log scores,
candidate log scores and margins. No generation for controlled items.

Headlines separately per section: complete-family counts/proportions (all 8 or
16 correct), distribution of reversal successes per family, mean within-family
reversal fraction, and margin min/median/mean/max. Reversal pairs fix query, order
and placement and vary assignment: four per ordinary family, eight per distractor
family; success requires both items correct. Diagnostics: item counts/accuracy,
ties, family mean/minimum margin, paired surface/distractor versus matched active
family outcomes. No inferential pass/fail gate or broad-English confidence interval.
Representative controlled examples: every member of lexically first family ID
within each section, chosen before results. Do not cherry-pick.

Naturalistic: greedy argmax, BOS plus exact prompt, maximum 32 new tokens; stop
after EOS or the first generated period/question-mark/exclamation-mark token.
This is a punctuation boundary, not a judgment of grammatical completeness.
Save all generated token IDs including stop token and verbatim decoded continuation
with special tokens omitted; never strip whitespace or repair text. Save stop
reason. Human-review fields, separately for every prompt: grammatical completeness
(complete sentence/fragment/unclear); subject consistency (consistent/conflicting/
not assessable); object consistency (consistent/conflicting/not assessable);
relation/action appropriateness (plausible/conflicting/unclear); repetition failure
(yes/no/unclear, repeated content without progress); contextual contradiction
(yes/no/unclear with quoted evidence); truncation (32-token cap before punctuation
or EOS, mechanical flag). Empty output is recorded explicitly; it is not success.
Unspecified new facts are not automatically contradictions. Semantic judgments
await the user's review; do not invent automatic success labels. Multiple coherent
continuations are possible, intentionally. No grammar-scoring model is used.

## Audits and interpretation

Require zero normalized duplicate or previously exposed exact prompts, zero full
prompt/completion overlap against listed sources, valid token roundtrips/boundaries,
and length within 256 including BOS/answers or generation budget. Decode corpus
JSON text fields before normalization; do not search serialized escapes as prose.
Report atomic fact overlap diagnostically; familiar facts do not demonstrate novel
semantic knowledge. No exact overlap does not establish absence of paraphrase or
semantic contamination. No model output influenced selection. Neither controlled
success nor naturalistic plausibility establishes broad comprehension, conversation,
reasoning, capacity limits or synthetic-to-English transfer. Controlled failures
confound QA format with factual selection and cannot erase P7's original 12/12.
No inference, training, parent selection or sacred binding access during freeze.
'''
if __name__=='__main__':main()
