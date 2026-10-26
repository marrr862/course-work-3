"""Armenian and Russian input: detect the language and translate it to English before analysis.

The classifiers were trained on English data, so non-English text is first translated offline with
Meta's NLLB-200 (distilled, 600M parameters) stored under models/mt_nllb. NLLB was chosen over the
smaller opus-mt-hy-en model, whose Armenian translations were often wrong (it hallucinated names
and places on simple headlines).
Download once:  python -m fakenews.translate --download
Translation errors propagate to the prediction, so the API returns the English text it analyzed.
"""
import argparse
import re
from functools import lru_cache
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = 'facebook/nllb-200-distilled-600M'
MODEL_DIR = ROOT / 'models' / 'mt_nllb'
SUPPORTED = {'hy': 'hye_Armn', 'ru': 'rus_Cyrl'}
ARMENIAN = re.compile(r'[Ա-և]')
CYRILLIC = re.compile(r'[Ѐ-ӿ]')
LATIN = re.compile(r'[A-Za-z]')
OTHER_SCRIPT = re.compile(r'[֐-ۿऀ-෿฀-໿ᄀ-ᇿ぀-ヿ㐀-鿿가-힯]')


def detect_language(text):
    """'hy', 'ru', 'en' by dominant script; 'other' for scripts we cannot translate."""
    text = text or ''
    counts = {'hy': len(ARMENIAN.findall(text)), 'ru': len(CYRILLIC.findall(text)), 'en': len(LATIN.findall(text))}
    if len(OTHER_SCRIPT.findall(text)) > max(counts.values()):
        return 'other'
    lang = max(counts, key=counts.get)
    return lang if counts[lang] else 'other'


def available(lang):
    return lang == 'en' or (lang in SUPPORTED and (MODEL_DIR / 'config.json').exists())


@lru_cache(maxsize=1)
def _load():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_DIR), AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).eval(), Lock()


def generate(texts, src, tgt, batch_size=16):
    """Batched NLLB translation between NLLB language codes (e.g. 'hye_Armn' -> 'eng_Latn')."""
    import torch
    tok, model, lock = _load()
    out = []
    with lock, torch.inference_mode():
        tok.src_lang = src
        for i in range(0, len(texts), batch_size):
            batch = tok(list(texts[i:i + batch_size]), return_tensors='pt', padding=True, truncation=True, max_length=256)
            ids = model.generate(**batch, forced_bos_token_id=tok.convert_tokens_to_ids(tgt), num_beams=4, max_new_tokens=256)
            out += [t.strip() for t in tok.batch_decode(ids, skip_special_tokens=True)]
    return out


def translate(text, lang):
    """Translate hy/ru text to English, sentence by sentence so long inputs are not cut."""
    parts = [p for p in re.split(r'(?<=[.!?։:])\s+|\n+', text.strip()) if p.strip()] or [text]
    return ' '.join(generate(parts, SUPPORTED[lang], 'eng_Latn')).strip()


def to_english(text):
    """Returns (english_text, detected_language). Raises ValueError for unsupported input."""
    lang = detect_language(text)
    if lang == 'en':
        return text, 'en'
    if lang not in SUPPORTED:
        raise ValueError('Supported languages: English, Armenian, Russian.')
    if not available(lang):
        raise ValueError(f'Translation model for {lang} is not installed: run python -m fakenews.translate --download')
    return translate(text, lang), lang


def download():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    AutoTokenizer.from_pretrained(MODEL_NAME).save_pretrained(MODEL_DIR)
    AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).save_pretrained(MODEL_DIR)
    print('saved', MODEL_DIR)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--download', action='store_true')
    ap.add_argument('text', nargs='?')
    a = ap.parse_args()
    if a.download:
        download()
    if a.text:
        print(to_english(a.text))
