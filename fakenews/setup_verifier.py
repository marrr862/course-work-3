"""Fetch a pinned, local NLI model and reproducible source snapshots."""
import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from .sources import fetch_article

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'cross-encoder/nli-deberta-v3-small'
REVISION = 'fa2804872c3b4bd748f38c0185cc85775361e735'
SOURCES = [f'https://science.nasa.gov/{body}/facts/' for body in ['earth','sun','moon','mars','jupiter','venus','mercury','saturn','uranus','neptune']]


def main():
    target = ROOT / 'models/nli_deberta'
    snapshot_download(repo_id=MODEL, revision=REVISION, local_dir=target,
                      allow_patterns=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','special_tokens_map.json','spm.model','added_tokens.json'])
    records=[]
    for url in SOURCES:
        try:
            item=fetch_article(url)
            records.append({k:item[k] for k in ['url','retrieved_at','sha256','title']})
            print('SOURCE', item['title'], len(item['text']), flush=True)
        except Exception as exc:
            print('SOURCE FAILED',url,type(exc).__name__,flush=True)
    metadata={'model_id':MODEL,'revision':REVISION,'training':'Pretrained by model authors on SNLI and MultiNLI; not trained on user LIAR or FakeNewsNet by this project.',
              'model_sha256':hashlib.sha256((target/'model.safetensors').read_bytes()).hexdigest(),'source_snapshots':records}
    (ROOT/'reports/verifier_manifest.json').write_text(json.dumps(metadata,indent=2))
    print('Setup complete',flush=True)

if __name__=='__main__':main()
