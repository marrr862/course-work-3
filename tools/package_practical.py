"""Create a portable practical-only archive; preserve original files and omit environments/docs."""
from pathlib import Path
import tarfile,hashlib,json
root=Path(__file__).resolve().parents[1]
output=root/'deliverables/Final_Practical_Code.tar.gz'
output.parent.mkdir(parents=True,exist_ok=True)
temporary=output.with_suffix('.gz.partial')
skip={'.git','.venv','.pytest_cache','__pycache__','.ipynb_checkpoints','deliverables'}
# models not used by the working system: stopped RoBERTa-large run, superseded MarianMT and NLI weights
unused_models={'fakenewsnet_rl_title','mt_hy_en','mt_ru_en','nli'}
files=[]
for path in root.rglob('*'):
    rel=path.relative_to(root)
    if not path.is_file() or any(p in skip or p.startswith('docx_') for p in rel.parts):continue
    if rel.parts[0]=='models' and len(rel.parts)>2 and rel.parts[1] in unused_models:continue
    if path.name.startswith('.env') or path.name=='.DS_Store':continue  # never ship secrets (Telegram token)
    if path.suffix in {'.docx','.pdf','.pyc'} or path.name in {'PRACTICAL_REPORT_HY.md','build_report.py','jupyter_server.log'}:continue
    files.append(path)
with tarfile.open(temporary,'w:gz',compresslevel=1) as archive:
    for path in sorted(files):archive.add(path,arcname='FakeNews_Practical/'+str(path.relative_to(root)),recursive=False)
with tarfile.open(temporary,'r:gz') as archive:
    names=archive.getnames()
    assert not any(Path(n).name.startswith('.env') for n in names)
    for required in ['Final_Practical.ipynb','fakenews/social.py','models/fakenewsnet_v4.joblib','models/mt_nllb/config.json','data/raw/liar/train.tsv','models/active_models.json','reports/optimization/fakenewsnet/selected.joblib','START_HERE_HY.md']:
        assert 'FakeNews_Practical/'+required in names
temporary.replace(output)
manifest={'archive':output.name,'files':len(files),'bytes':output.stat().st_size,'sha256':hashlib.file_digest(output.open('rb'),'sha256').hexdigest(),'excluded':'Python environment, .env secrets, unused model weights ('+', '.join(sorted(unused_models))+'), Word documents, PDF files, document renders and caches'}
(root/'deliverables/Final_Practical_Code.manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest))
