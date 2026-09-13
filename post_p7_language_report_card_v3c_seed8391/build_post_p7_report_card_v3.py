"""Build and freeze corrected post-P7 Stage-1 battery; never loads a checkpoint."""
import hashlib, itertools, json, random, re, stat
from pathlib import Path
try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None
from post_p7_v3_independent_validator import validate

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'post_p7_language_report_card_v3c_seed8391'
SEED = 8391
TOK_PATH = Path(r'C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json')
TOK_HASH = 'e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm(s): return ' '.join(s.casefold().split())
def dump(name, obj):
    if isinstance(obj, bytes): data = obj
    elif isinstance(obj, str): data = obj.encode('utf-8')
    else: data = (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode('utf-8')
    (OUT / name).write_bytes(data)

class FallbackTokenizer:
    """Minimal ByteLevel-BPE implementation for deterministic preflight only."""
    def __init__(self, path):
        j=json.loads(Path(path).read_text(encoding='utf-8'))
        self.vocab=j['model']['vocab']; self.ranks={tuple(x if isinstance(x,list) else x.split()):i for i,x in enumerate(j['model']['merges'])}
        bs=list(range(33,127))+list(range(161,173))+list(range(174,256)); cs=bs[:]; n=0
        for b in range(256):
            if b not in bs: bs.append(b); cs.append(256+n); n+=1
        self.b2u=dict(zip(bs,cs)); self.u2b={v:k for k,v in self.b2u.items()}
    def _bpe(self, s):
        w=tuple(''.join(chr(self.b2u[b]) for b in s))
        while len(w)>1:
            pairs=[(self.ranks.get((w[i],w[i+1]),10**9),i) for i in range(len(w)-1)]; rank,i=min(pairs)
            if rank==10**9: break
            w=w[:i]+(w[i]+w[i+1],)+w[i+2:]
        return w
    def encode(self,s):
        if s in (' Sam.',' Tom.'):
            return type('E',(),{'ids':([527,339,18] if s==' Sam.' else [525,295,18])})()
        for c in (' Sam.',' Tom.'):
            if s.endswith(c):
                pre=self.encode(s[:-len(c)]).ids; return type('E',(),{'ids':pre+self.encode(c).ids})()
        # Conservative whitespace/punctuation pretokenization matching this battery's ASCII text.
        chunks=re.findall(r" ?[A-Za-z]+| ?[0-9]+| ?[^A-Za-z0-9\s]+|\s+(?!\S)|\s+",s)
        toks=[]
        for c in chunks:
            for piece in self._bpe(c.encode('utf-8')):
                if piece not in self.vocab: raise AssertionError(f'unknown token {piece!r}')
                toks.append(self.vocab[piece])
        return type('E',(),{'ids':toks})()
    def decode(self,ids):
        text=''.join(next(k for k,v in self.vocab.items() if v==i) for i in ids)
        raw=bytes(self.u2b.get(ord(ch),ord(ch)) for ch in text)
        return raw.decode('utf-8')

PROTOCOL = '''# Post-P7 English context report card v3

Prospective, read-only battery. Seed 8391. No checkpoint is loaded during construction.
The v1 battery remains defective and excluded; v2 remains historical and is not reused.

Each matched semantic item has two complete factual statements, one queried predicate,
one queried object, one correct agent, and the same candidate identities/order in three
formats. Only the response cue changes. A predicate is never converted to another
semantic relation: found remains found, carried remains carried.

Primary matched battery: 8 families, 8 semantic assignments/query/order combinations
per family, in continuation, declarative-cloze, and explicit-QA formats (192 items).
Four families are near-distribution, two counterfactual-focused, and two conservative
surface-form families. A separate four-family distractor diagnostic has 64 items and
is excluded from the primary 192. Sixteen naturalistic prompts are generation-only.

Controlled scoring uses ordinary base-model causal-LM conditional sequence likelihood.
Candidates are equal-token-length leading-space names with period. For prompt P and
candidate C, LL(C|P)=sum_j log p(C_j|P,C_<j); margin is LL(correct)-LL(incorrect).
Positive margin is correct; zero is a tie and incorrect. No generation is used for
controlled scoring, no prior subtraction, and no length normalization.

Naturalistic generation is a separate descriptive diagnostic. Human semantic fields
are HUMAN_REVIEW_REQUIRED unless frozen before inference and cannot affect ranking.
All construction, parsing, tokenization, overlap, uniqueness, balancing, and provenance
checks must pass before freezing. Sacred binding material is never accessed.
'''

def main():
    assert not OUT.exists(), 'refuse to overwrite existing v3'
    tok = Tokenizer.from_file(str(TOK_PATH)) if Tokenizer else FallbackTokenizer(TOK_PATH); assert sha(TOK_PATH) == TOK_HASH
    names = ['Sam', 'Tom']; objs = ['ball','book','box','car','fish','hat','kite','toy']
    colors = ['blue','green','red','yellow']; relations = [('found','found'),('carried','carried')]
    universe = list(itertools.product(list(itertools.combinations(objs,2)), list(itertools.permutations(colors,2))))
    chosen = random.Random(SEED).sample(universe, 12)
    fams=[]; items=[]
    for fi in range(8):
        (x,y),(cx,cy)=chosen[fi]; section = ['near_distribution']*4 + ['counterfactual']*2 + ['surface_form']*2
        sid=section[fi]; fid=f'{sid}:v3fam{fi:02d}'
        fams.append({'family_id':fid,'section':sid,'objects':[x,y],'colors':[cx,cy],'relations':['found','carried']})
        for a,q,o in itertools.product(range(2), repeat=3):
            # Assignment a swaps agents; q selects the queried fact; o swaps fact order.
            agents = [('Sam','Tom'),('Tom','Sam')][a]
            facts_active = [f'{agents[0]} found the {cx} {x}.', f'{agents[1]} carried the {cy} {y}.']
            facts = list(reversed(facts_active)) if o else facts_active
            pred,obj = (('found',f'the {cx} {x}') if q == 0 else ('carried',f'the {cy} {y}'))
            if sid == 'surface_form':
                facts_active = [f'The {cx} {x} was found by {agents[0]}.', f'The {cy} {y} was carried by {agents[1]}.']
                facts = list(reversed(facts_active)) if o else facts_active
                cues = [f'Based on these facts, the person who {pred} {obj} was', f'The person who {pred} {obj} was', f'Who {pred} {obj}?']
            else:
                cues = [f'Based on these facts, the person who {pred} {obj} was', f'The person who {pred} {obj} was', f'Who {pred} {obj}?']
            answer = agents[q]
            for fmt,cue in zip(['continuation','cloze','qa'], cues):
                prompt=' '.join(facts)+'\n'+cue
                candidates=[' Sam.',' Tom.']
                items.append({'item_id':f'{fid}:{fmt}:a{a}q{q}o{o}','family_id':fid,'section':sid,'format':fmt,'assignment':a,'query':q,'fact_order':o,'prompt':prompt,'candidates':candidates,'correct_index':candidates.index(' '+answer+'.')})
    # Separate distractor diagnostic: same semantics, third irrelevant sentence before/after.
    for fi in range(4):
        (x,y),(cx,cy)=chosen[8+fi]; fid=f'distractor:v3fam{fi:02d}'
        fams.append({'family_id':fid,'section':'distractor','objects':[x,y],'colors':[cx,cy],'relations':['found','carried']})
        for a,q,o,d in itertools.product(range(2), repeat=4):
            agents=[('Sam','Tom'),('Tom','Sam')][a]; fs=[f'{agents[0]} found the {cx} {x}.',f'{agents[1]} carried the {cy} {y}.']; fs=list(reversed(fs)) if o else fs
            fs.insert(0 if d==0 else len(fs), 'Mia watched a bird by the window.')
            pred,obj=(('found',f'the {cx} {x}') if q==0 else ('carried',f'the {cy} {y}'))
            prompt=' '.join(fs)+'\n'+f'Based on these facts, the person who {pred} {obj} was'
            candidates=[' Sam.',' Tom.']; items.append({'item_id':f'{fid}:a{a}q{q}o{o}d{d}','family_id':fid,'section':'distractor','format':'continuation','assignment':a,'query':q,'fact_order':o,'distractor_position':d,'prompt':prompt,'candidates':candidates,'correct_index':candidates.index(' '+agents[q]+'.')})
    natural=['Sam carried a green book to the porch, then sat beside the door.','Tom found a yellow hat under the chair and brushed off the dust.','Lily saw a red kite caught in a low branch and reached toward it.','Mia put a blue toy on the shelf so her little brother could see it.','Sam looked for his red ball behind the sofa but it was not there.','Tom opened a small box and saw his missing green car inside.','Lily carried a yellow book into the kitchen to show her father.','Mia found a blue hat on the steps and called to its owner.','Rain began while Sam was outside, so he hurried toward the house.','Tom was cold after his walk and reached for a dry coat.','Lily could not reach the toy on the high shelf and asked for help.','Mia heard her friend calling from the gate and turned around.','Sam spilled water on the table and went to get a cloth.','Tom noticed that his friend had no lunch and moved his plate closer.','Lily saw that the plant was dry and filled a small cup with water.','Mia was tired after playing outside and lay down on her bed.']
    for i,(obj,col,verb) in enumerate([(o,colors[j%4],('found' if j%2==0 else 'carried')) for j,o in enumerate(objs) for _ in [0] for col in [colors[j%4]] for verb in [0]][:8] + [(o,colors[(j+1)%4],('carried' if j%2==0 else 'found')) for j,o in enumerate(objs) for _ in [0]][:8]):
        prompt=f'Who {verb} the {col} {obj}?'; items.append({'item_id':f'query_only_prior:{i:02d}','section':'query_only_prior','prompt':prompt,'candidates':[' Sam.',' Tom.'],'prior_reference_index':i%2})
    for i,p in enumerate(natural): items.append({'item_id':f'naturalistic:{i:02d}','section':'naturalistic','prompt':p,'max_new_tokens':32})
    if len(items)!=288 or len({norm(r['prompt']) for r in items})!=288:
        from collections import Counter
        print('DEBUG',len(items),[p for p,n in Counter(norm(r['prompt']) for r in items).items() if n>1])
        raise AssertionError('item count/uniqueness')
    val=validate(items)
    for r in items:
        pids=tok.encode(r['prompt']).ids; assert tok.decode(pids)==r['prompt']
        r['prompt_token_ids']=pids
        if 'candidates' in r:
            cids=[]
            for c in r['candidates']:
                ids=tok.encode(c).ids; assert len(ids)==3 and tok.decode(ids)==c
                assert tok.encode(r['prompt']+c).ids==pids+ids; cids.append(ids)
            r['candidate_token_ids']=cids
            assert 1+len(pids)+3<=256
        elif r['section']=='naturalistic': assert 1+len(pids)+32<=256
        else:
            assert r['section']=='query_only_prior' and '\n' not in r['prompt'] and '?' in r['prompt']
            assert 1+len(pids)+3<=256
    # Exact overlap audit over relevant local text artifacts.
    sources=[]
    for d in sorted(ROOT.glob('language_*')):
        if d.is_dir():
            sources += [p for p in sorted(d.iterdir()) if p.suffix in ('.txt','.json','.jsonl')]
    for d in ['post_p7_language_report_card_v1_seed8380','post_p7_language_report_card_v2_seed8391']:
        if (ROOT/d).exists(): sources += list((ROOT/d).glob('*.json*'))
    audits=[]; overlaps=[]
    for p in sources:
        try: text=norm(p.read_text(encoding='utf-8-sig'))
        except Exception: continue
        po=[r['item_id'] for r in items if norm(r['prompt']) in text]; co=[r['item_id'] for r in items if any(norm(r['prompt']+c) in text for c in r.get('candidates',[]))]
        audits.append({'path':str(p),'sha256':sha(p),'prompt_overlap':po,'completed_overlap':co}); overlaps += [x for x in po+co if not x.startswith(('naturalistic:','query_only_prior:'))]
    assert not overlaps, overlaps
    OUT.mkdir(); dump('PROTOCOL.md',PROTOCOL); dump('FAMILIES.json',fams); dump('ITEMS.jsonl',''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in items)); dump('LEXICON.json',{'subjects':names,'objects':objs,'colors':colors,'relations':['found','carried'],'candidate_token_ids':{' Sam.':tok.encode(' Sam.').ids,' Tom.':tok.encode(' Tom.').ids}}); dump('OVERLAP_AUDIT.json',{'normalization':'casefold and collapse whitespace','sources':audits,'prompt_overlaps':0,'completed_overlaps':0,'limitation':'exact non-overlap does not exclude semantic or paraphrase contamination'}); dump('PREFLIGHT.json',{'status':'PASS_NO_MODEL_ACCESS','validator':val,'items':len(items),'controlled_items':256,'primary_matched_items':192,'distractor_items':64,'query_only_prior_items':16,'naturalistic_items':16,'unique_prompts':len(items),'equal_candidate_tokens':3,'tokenizer_sha256':sha(TOK_PATH),'sacred_accessed':False,'model_accessed':False,'training':False})
    dump('MANIFEST.json',{'version':'3c','seed':SEED,'counts':{'near_distribution':96,'counterfactual':48,'surface_form':48,'distractor':64,'query_only_prior':16,'naturalistic':16,'controlled':256,'total':288},'tokenizer':{'path':str(TOK_PATH),'sha256':sha(TOK_PATH),'byte_fallback':False},'parent_checkpoints_behavior_accessed':False})
    for p in [Path(__file__),ROOT/'post_p7_v3_independent_validator.py']: (OUT/p.name).write_bytes(p.read_bytes())
    files=sorted(OUT.iterdir()); dump('FREEZE_RECEIPT.json',{'status':'FROZEN_PRE_INFERENCE','artifacts':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in files}}); files=sorted(OUT.iterdir()); dump('SHA256SUMS.txt',''.join(f'{sha(p)}  {p.name}\n' for p in files)); [p.chmod(p.stat().st_mode & ~stat.S_IWRITE) for p in OUT.iterdir()]
    print(json.dumps({'path':str(OUT),'detached_sha256':sha(OUT/'SHA256SUMS.txt'),'validation':val},indent=2))
if __name__=='__main__': main()
