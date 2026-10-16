"""Download and verify the exact public SciFact snapshot used in this study."""
import hashlib,io,json,tarfile
from datetime import datetime,timezone
import httpx
from .train import ROOT,write_json
URL='https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz'
SHA256='11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be'
NAMES={'corpus.jsonl','claims_train.jsonl','claims_dev.jsonl','claims_test.jsonl'}
if __name__=='__main__':
    root=ROOT/'data/scifact';root.mkdir(exist_ok=True,parents=True)
    archive=root/'data.tar.gz'
    content=archive.read_bytes() if archive.exists() else httpx.get(URL,follow_redirects=True,timeout=60).raise_for_status().content
    if hashlib.sha256(content).hexdigest()!=SHA256:raise RuntimeError('Dataset checksum changed; refusing silent benchmark replacement.')
    archive.write_bytes(content)
    with tarfile.open(fileobj=io.BytesIO(content),mode='r:gz') as tar:
        for member in tar.getmembers():
            name=member.name.rsplit('/',1)[-1]
            if member.isfile() and name in NAMES:(root/name).write_bytes(tar.extractfile(member).read())
    if any(not(root/name).exists() for name in NAMES):raise RuntimeError('Dataset archive incomplete.')
    write_json(root/'manifest.json',{'source':URL,'archive_sha256':SHA256,'retrieved_at':datetime.now(timezone.utc).isoformat(),'files':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sorted(NAMES)}})
    print('SciFact snapshot ready; original datasets unchanged.')
