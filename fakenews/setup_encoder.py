"""Download the pinned sentence encoder used in optimization."""
import json
from huggingface_hub import snapshot_download
from .train import ROOT
MODEL='sentence-transformers/all-MiniLM-L6-v2'
REVISION='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
if __name__=='__main__':
    snapshot_download(MODEL,revision=REVISION,local_dir=ROOT/'models/sentence_minilm',allow_patterns=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt'],max_workers=2)
    folder=ROOT/'reports/optimization';folder.mkdir(parents=True,exist_ok=True)
    (folder/'encoder_manifest.json').write_text(json.dumps({'model':MODEL,'revision':REVISION,'pooling':'attention-mask mean, L2 normalization','max_length':128},indent=2))
