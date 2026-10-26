"""Stage-2 stacked classifiers.

p(x) = alpha * p_stage1(text) + (1 - alpha) * p_LR(features)

The stage-1 model (fine-tuned MiniLM + TF-IDF ensemble) is loaded lazily from its own
artifact, so the stage-2 artifact stays small and the neural weights are never duplicated.
Features: word + char TF-IDF on the normalized text key, plus the news source domain
(FakeNewsNet) or speaker metadata and a train-only speaker history (LIAR).
"""
import re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from .data import LABELS, text_key

ROOT = Path(__file__).resolve().parents[1]
LIAR_META = ['speaker', 'party', 'state', 'job', 'context']
LIAR_FIELDS = LIAR_META + ['subject']


def domain(url):
    """Host of a news URL without scheme and leading www; 'NONE' when missing."""
    if url is None:
        return 'NONE'
    u = str(url).strip().lower()
    if u in ('', 'nan', 'none'):
        return 'NONE'
    u = re.sub(r'^[a-z][a-z0-9+.-]*://', '', u)
    u = re.sub(r'^www\d?\.', '', u)
    host = re.split(r'[/?#]', u, maxsplit=1)[0]
    host = host.rsplit('@', 1)[-1].split(':', 1)[0]
    return host or 'NONE'


def url_words(url):
    """Domain plus the words of the URL path, e.g. 'hollywoodlife.com paris jackson cara'."""
    u = str(url).lower()
    if u in ('nan', '', 'none'):
        return 'no link'
    path = re.sub(r'^[a-z]+://', '', u).split('/', 1)
    words = ' '.join(re.findall(r'[a-z]{2,}', path[1] if len(path) > 1 else ''))
    return (domain(url) + ' ' + words).strip()


def normalize_meta(field, value):
    """Map user-typed LIAR metadata to the dataset's conventions (speaker slugs, lowercase party)."""
    value = '' if value is None else str(value).strip()
    if not value or value.lower() == 'nan':
        return 'NA'
    if field == 'speaker':
        return re.sub(r'[^a-z0-9]+', '-', value.lower()).strip('-') or 'NA'
    if field == 'subject':
        return ','.join(t for t in (re.sub(r'[^a-z0-9]+', '-', x.lower()).strip('-') for x in value.split(',')) if t) or 'NA'
    if field == 'party':
        return value.lower()
    return value


def _lexical():
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=.98, sublinear_tf=True, max_features=60000)
    char = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=100000)
    return word, char


class FakeNewsFeatures:
    """Word + char TF-IDF of the text key, optionally one-hot source domain."""

    def __init__(self, use_domain):
        self.use_domain = use_domain
        self.word, self.char = _lexical()
        self.domains = OneHotEncoder(handle_unknown='ignore', min_frequency=3)

    def fit(self, frame):
        self.word.fit(frame.key)
        self.char.fit(frame.key)
        if self.use_domain:
            self.domains.fit(frame[['domain']])
        return self

    def transform(self, frame, loo=False):
        parts = [self.word.transform(frame.key), self.char.transform(frame.key)]
        if self.use_domain:
            parts.append(self.domains.transform(frame[['domain']]))
        return sp.hstack(parts).tocsr()

    def frame(self, texts, urls=None, metadata=None):
        urls = urls if urls is not None else [None] * len(texts)
        return pd.DataFrame({'key': [text_key(t) for t in texts], 'domain': [domain(u) for u in urls]})


class LiarFeatures:
    """Word + char TF-IDF; optionally speaker metadata, subjects and train-only speaker history."""

    def __init__(self, use_metadata):
        self.use_metadata = use_metadata
        _, self.char = _lexical()
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.meta = OneHotEncoder(handle_unknown='ignore', min_frequency=2)
        self.subjects = TfidfVectorizer(token_pattern=r'[^,]+')
        self.history = None

    def fit(self, frame):
        self.word.fit(frame.key)
        self.char.fit(frame.key)
        if self.use_metadata:
            self.meta.fit(frame[LIAR_META])
            self.subjects.fit(frame.subject)
            self.history = pd.crosstab(frame.speaker, frame.label).reindex(columns=LABELS, fill_value=0)
        return self

    def credit(self, frame, loo):
        counts = self.history.reindex(frame.speaker).fillna(0).values.copy()
        if loo:  # training rows must not see their own label
            counts[np.arange(len(frame)), [LABELS.index(label) for label in frame.label]] -= 1
        counts = np.clip(counts, 0, None)
        total = counts.sum(1, keepdims=True)
        return np.hstack([(counts + 1) / (total + len(LABELS)), np.log1p(total) / 5])

    def transform(self, frame, loo=False):
        parts = [self.word.transform(frame.key), self.char.transform(frame.key)]
        if self.use_metadata:
            parts += [self.meta.transform(frame[LIAR_META]), self.subjects.transform(frame.subject), sp.csr_matrix(self.credit(frame, loo))]
        return sp.hstack(parts).tocsr()

    def frame(self, texts, urls=None, metadata=None):
        metadata = metadata if metadata is not None else [{}] * len(texts)
        rows = []
        for text, meta in zip(texts, metadata):
            meta = meta or {}
            rows.append({'key': text_key(text), **{f: normalize_meta(f, meta.get(f)) for f in LIAR_FIELDS}})
        return pd.DataFrame(rows)


class StackedClassifier:
    def __init__(self, base_path, features, lr, alpha, classes, threshold=None, positive=None, name=''):
        self.base_path = str(base_path)
        self.features = features
        self.lr = lr
        self.alpha = float(alpha)
        self.classes_ = np.asarray(classes)
        self.threshold = threshold
        self.positive = positive
        self.name = name
        self._base = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_base'] = None  # never pickle the neural model into this artifact
        return state

    @property
    def base(self):
        if self._base is None:
            self._base = joblib.load(ROOT / self.base_path)['model']
        return self._base

    def set_base(self, model):
        self._base = model
        return self

    def _align(self, probabilities, classes):
        order = [list(classes).index(c) for c in self.classes_]
        return np.asarray(probabilities)[:, order]

    def lr_proba(self, frame):
        return self._align(self.lr.predict_proba(self.features.transform(frame)), self.lr.classes_)

    def combine(self, base_proba, frame):
        """Combine already-computed stage-1 probabilities (columns in classes_ order)."""
        return self.alpha * np.asarray(base_proba) + (1 - self.alpha) * self.lr_proba(frame)

    def decide(self, probabilities):
        if self.threshold is not None and self.positive is not None:
            k = list(self.classes_).index(self.positive)
            other = [c for c in self.classes_ if c != self.positive][0]
            return np.where(probabilities[:, k] >= self.threshold, self.positive, other)
        return self.classes_[np.asarray(probabilities).argmax(1)]

    def predict_proba(self, texts, urls=None, metadata=None):
        texts = list(texts)
        base = self._align(self.base.predict_proba(texts), self.base.classes_) if self.alpha > 0 else np.zeros((len(texts), len(self.classes_)))
        return self.combine(base, self.features.frame(texts, urls, metadata))

    def predict(self, texts, urls=None, metadata=None):
        return self.decide(self.predict_proba(texts, urls, metadata))


class Stage3Classifier:
    """p = w_large * p_large(text) + w_stage1 * p_stage1(text) + (1 - w_large - w_stage1) * p_LR(features).

    Wraps a stage-2 StackedClassifier (its features, LR, threshold logic and lazily loaded
    stage-1 model) and adds a larger fine-tuned transformer stored under models/.
    """

    def __init__(self, stage2, large_path, w_large, w_stage1, threshold, max_length=64):
        from .advanced_models import TransformerClassifier
        self.stage2 = stage2
        self.large = TransformerClassifier(large_path, stage2.classes_)
        self.large.max_length = max_length
        self.w_large, self.w_stage1 = float(w_large), float(w_stage1)
        self.threshold = threshold
        self.positive = stage2.positive
        self.classes_ = stage2.classes_

    def set_base(self, model):
        self.stage2.set_base(model)
        return self

    def combine(self, large_proba, stage1_proba, frame):
        lr = self.stage2.lr_proba(frame)
        return self.w_large * np.asarray(large_proba) + self.w_stage1 * np.asarray(stage1_proba) + (1 - self.w_large - self.w_stage1) * lr

    def decide(self, probabilities):
        k = list(self.classes_).index(self.positive)
        other = [c for c in self.classes_ if c != self.positive][0]
        return np.where(np.asarray(probabilities)[:, k] >= self.threshold, self.positive, other)

    def predict_proba(self, texts, urls=None, metadata=None):
        texts = list(texts)
        s2 = self.stage2
        stage1 = s2._align(s2.base.predict_proba(texts), s2.base.classes_)
        large = s2._align(self.large.predict_proba(texts), self.large.classes_)
        return self.combine(large, stage1, s2.features.frame(texts, urls, metadata))

    def predict(self, texts, urls=None, metadata=None):
        return self.decide(self.predict_proba(texts, urls, metadata))


class UrlLR:
    """Logistic regression on headline TF-IDF, domain one-hot and TF-IDF of URL path words."""

    def __init__(self, features, path_vectorizer, lr):
        self.features, self.path_vectorizer, self.lr = features, path_vectorizer, lr

    def fake_proba(self, texts, urls):
        frame = self.features.frame(texts, urls)
        X = sp.hstack([self.features.transform(frame), self.path_vectorizer.transform([url_words(u) for u in urls])]).tocsr()
        return self.lr.predict_proba(X)[:, list(self.lr.classes_).index('fake')]


class PairTransformer:
    """Fine-tuned transformer whose input is (headline, domain + URL path words)."""

    def __init__(self, path, classes, max_length=96):
        self.path, self.classes_, self.max_length = path, np.asarray(classes), max_length

    def predict_proba(self, texts, urls):
        import torch
        from .advanced_models import encoder
        tok, model, lock = encoder(self.path, classification=True)
        second = [url_words(u) for u in urls]
        with lock, torch.inference_mode():
            batch = tok(list(texts), second, padding=True, truncation='longest_first', max_length=self.max_length, return_tensors='pt')
            return model(**batch).logits.float().softmax(-1).numpy()


# component name -> (model directory, max token length) for single-input transformers
STAGE4_TRANSFORMERS = {'roberta': ('models/fakenewsnet_large', 64), 'electra': ('models/fakenewsnet_electra', 96)}


class Stage4Classifier:
    """P(fake) = sigmoid(w . logit(p_component) + b), threshold chosen by cross-validation on validation.

    Components: stage1 (MiniLM + TF-IDF ensemble), fine-tuned RoBERTa / ELECTRA on the headline,
    RoBERTa on headline + URL words (rb_url), and the stage-2 logistic models (lr_text, lr_domain)
    plus a URL-path logistic model (lr_url). All loaded lazily; nothing heavy is pickled.
    """

    def __init__(self, components, meta, threshold, url_lr=None, v2_path='models/fakenewsnet_v2.joblib'):
        self.components, self.meta, self.threshold, self.url_lr = list(components), meta, float(threshold), url_lr
        self.v2_path = v2_path
        self.classes_ = np.array(['fake', 'real'])
        self.positive = 'fake'
        self._v2 = None
        self._base = None

    def __getstate__(self):
        state = self.__dict__.copy(); state['_v2'] = None; state['_base'] = None
        return state

    @property
    def v2(self):
        if self._v2 is None:
            self._v2 = joblib.load(ROOT / self.v2_path)['models']
        return self._v2

    def set_base(self, model):
        self._base = model
        return self

    @property
    def uses_url(self):
        return any(c in self.components for c in ('rb_url', 'lr_domain', 'lr_url'))

    def component_proba(self, texts, urls):
        """P(fake) of every component, columns in self.components order."""
        from .advanced_models import TransformerClassifier
        texts = list(texts); urls = list(urls) if urls is not None else [None] * len(texts)
        cols = []
        for name in self.components:
            if name == 'stage1':
                base = self._base if self._base is not None else self.v2['text'].base
                p = np.asarray(base.predict_proba(texts))[:, list(base.classes_).index('fake')]
            elif name in STAGE4_TRANSFORMERS:
                path, length = STAGE4_TRANSFORMERS[name]
                model = TransformerClassifier(path, ['fake', 'real']); model.max_length = length
                p = model.predict_proba(texts)[:, 0]
            elif name == 'rb_url':
                p = PairTransformer('models/fakenewsnet_rb_url', ['fake', 'real']).predict_proba(texts, urls)[:, 0]
            elif name in ('lr_text', 'lr_domain'):
                st = self.v2[name[3:]]
                p = st.lr_proba(st.features.frame(texts, urls))[:, 0]
            elif name == 'lr_url':
                p = self.url_lr.fake_proba(texts, urls)
            else:
                raise ValueError(f'unknown component {name}')
            cols.append(p)
        return np.column_stack(cols)

    def meta_proba(self, X):
        """X: logits of component P(fake). Returns [[P(fake), P(real)], ...]."""
        p = self.meta.predict_proba(X)[:, list(self.meta.classes_).index('fake')]
        return np.column_stack([p, 1 - p])

    def decide(self, probabilities):
        return np.where(np.asarray(probabilities)[:, 0] >= self.threshold, 'fake', 'real')

    def predict_proba(self, texts, urls=None, metadata=None):
        P = np.clip(self.component_proba(texts, urls), 1e-5, 1 - 1e-5)
        return self.meta_proba(np.log(P / (1 - P)))

    def predict(self, texts, urls=None, metadata=None):
        return self.decide(self.predict_proba(texts, urls))
