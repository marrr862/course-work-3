"""Evidence retrieval -> NLI -> extractive correction with source provenance."""
from functools import lru_cache
from pathlib import Path
import re
from threading import Lock
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from .sources import cached_sources, fetch_article, sentences, wikipedia_search

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = .92
MIN_SIMILARITY = .14
LABELS = ['contradiction', 'entailment', 'neutral']


class NLI:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        path = ROOT / 'models/nli_deberta'
        if not (path / 'model.safetensors').exists():
            raise FileNotFoundError('Run python -m fakenews.setup_verifier first')
        torch.set_num_threads(4)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True, use_safetensors=True)
        self.model.eval()
        self.lock = Lock()

    def score(self, pairs):
        if not pairs: return []
        rows = []
        with self.lock, self.torch.inference_mode():
            for start in range(0,len(pairs),8):
                chunk=pairs[start:start+8]
                encoded=self.tokenizer([p[0] for p in chunk],[p[1] for p in chunk],padding=True,truncation='only_first',max_length=512,return_tensors='pt')
                probabilities=self.torch.softmax(self.model(**encoded).logits,dim=1).cpu().numpy()
                rows.extend([{label:float(value) for label,value in zip(LABELS,row)} for row in probabilities])
        return rows


@lru_cache(maxsize=1)
def get_nli():
    return NLI()


def split_claims(text):
    text = re.sub(r'\s+', ' ', text).strip()
    # Keep common abbreviations intact during sentence segmentation.
    protected = text
    for abbreviation in ['U.S.', 'U.K.', 'Mr.', 'Mrs.', 'Dr.']:
        protected = protected.replace(abbreviation, abbreviation.replace('.', '<DOT>'))
    chunks = [p.replace('<DOT>', '.').strip() for p in re.split(r'(?<=[.!?])\s+(?=[A-Z])',protected) if p.strip()]
    return chunks


def retrieve(claim, documents, limit=16):
    passages=[]
    words = re.findall(r"[A-Za-z]+", claim.lower())
    meaningful = [w for w in words if w not in ENGLISH_STOP_WORDS and w not in {'says','said','claim','claims'}]
    subject = meaningful[0] if meaningful else ''
    for document in documents:
        # Require evidence about the claim's leading subject before invoking NLI.
        # This conservative gate prevents an unrelated sentence from being treated as contradiction.
        title_words = set(re.findall(r'[a-z]+', document['title'].lower()))

        for sentence in sentences(document['text']):
            sentence_words = re.findall(r'[a-z]+', sentence.lower())
            prefix = [w for w in sentence_words[:8] if w not in ENGLISH_STOP_WORDS]
            if subject not in title_words and subject not in prefix[:2]:
                continue
            claim_terms = {w.rstrip('s') for w in meaningful}
            evidence_terms = {w.rstrip('s') for w in sentence_words} | title_words
            if subject not in title_words and len(claim_terms & evidence_terms) < min(2, len(claim_terms)):
                continue
            passages.append({'text':sentence,'source':{k:v for k,v in document.items() if k!='text'}})
    if not passages: return []
    tfidf=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,stop_words='english')
    matrix=tfidf.fit_transform([p['text'] for p in passages])
    scores=(matrix @ tfidf.transform([claim]).T).toarray().ravel()
    found=[];seen=set()
    for idx in np.argsort(scores)[::-1]:
        if scores[idx] < MIN_SIMILARITY or len(found)>=limit:break
        text=passages[idx]['text']
        if text in seen:continue
        seen.add(text)
        found.append({**passages[idx],'similarity':float(scores[idx])})
    return found


def verify_claim(claim, documents, nli=None):
    candidates=retrieve(claim,documents)
    if not candidates:
        return {'claim':claim,'status':'insufficient_evidence','correction':None,'evidence':[]}
    scorer=nli or get_nli()
    scores=scorer.score([(p['source']['title']+'. '+p['text'],claim) for p in candidates])
    for p,score in zip(candidates,scores):p['nli_scores']=score
    support=[p for p in candidates if p['nli_scores']['entailment']>=THRESHOLD]
    contradict=[p for p in candidates if p['nli_scores']['contradiction']>=THRESHOLD]
    if support and contradict:
        return {'claim':claim,'status':'conflicting_evidence','correction':None,'evidence':candidates}
    if contradict:
        best=max(contradict,key=lambda p:p['nli_scores']['contradiction'])
        return {'claim':claim,'status':'refuted_by_sources','correction':best['text'],
                'correction_method':'verbatim_source_sentence', 'correction_source':best['source'],
                'evidence':[best]+[p for p in candidates if p is not best]}
    if support:
        best=max(support,key=lambda p:p['nli_scores']['entailment'])
        return {'claim':claim,'status':'supported_by_sources','correction':None,
                'evidence':[best]+[p for p in candidates if p is not best]}
    return {'claim':claim,'status':'insufficient_evidence','correction':None,'evidence':candidates}


def verify_text(text, live=False, source_urls=None, *, verifier=None, initial_documents=None):
    started=time.perf_counter()
    claims=split_claims(text)
    if len(claims)>5 or any(len(c.split())>100 or re.search(r'\b(and|but|whereas|although)\b|;', c, re.I) for c in claims):
        return {'status':'needs_shorter_claims','claims':[],'corrected_text':None,'message':'Use at most five short factual sentences, each up to 100 words.'}
    documents=cached_sources() if initial_documents is None else list(initial_documents)
    errors=[]
    for url in source_urls or []:
        try:documents.append(fetch_article(url,refresh=live))
        except Exception as exc:
            # No URLs with credentials or server internals are echoed from exceptions.
            errors.append({'provider':'article_fetch','status':type(exc).__name__})
    if live:
        for claim in claims:
            found,failures=wikipedia_search(claim)
            documents.extend(found)
            errors.extend({'provider':'wikipedia','status':failure} for failure in failures)
    documents=list({d['url']:d for d in documents}.values())
    try:
        results=[(verifier or verify_claim)(claim,documents) for claim in claims]
    except (FileNotFoundError, OSError):
        return {'status':'model_unavailable','claims':[],'corrected_text':None,'message':'Install the verifier model with python -m fakenews.setup_verifier','provider_errors':errors}
    decisive={'refuted_by_sources','supported_by_sources'}
    complete=bool(results) and all(r['status'] in decisive for r in results)
    replacements=[('According to '+r['correction_source']['title']+': '+r['correction']+' ['+r['correction_source']['url']+']') if r['correction'] else r['claim'] for r in results]
    corrected=' '.join(replacements) if complete and any(r['correction'] for r in results) else None
    if complete:
        status='correction_proposed' if corrected else 'supported_by_sources'
    else:status='needs_review'
    return {'status':status,'claims':results,'corrected_text':corrected,'provider_errors':errors,
            'source_count':len(documents),'elapsed_seconds':round(time.perf_counter()-started,3),
            'mode':'live_and_cached' if live else 'cached_sources',
            'method':'Retrieved evidence scored by a pretrained NLI model; corrections quote source sentences and retain attribution.',
            'caveat':'Automated evidence-relative assessment, not guaranteed truth. Source dates, context, source reliability and model errors require consideration. NLI scores are not calibrated probabilities of truth.'}
