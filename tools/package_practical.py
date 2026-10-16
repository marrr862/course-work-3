"""Create a portable practical-only archive; preserve original files and omit environments/docs."""
from pathlib import Path
import tarfile,hashlib,json
root=Path(__file__).resolve().parents[1]
output=root/'deliverables/Final_Practical_Code.tar.gz'
output.parent.mkdir(parents=True,exist_ok=True)
temporary=output.with_suffix('.gz.partial')
skip={'.venv','.pytest_cache','__pycache__','.ipynb_checkpoints','deliverables'}
files=[]
for path in root.rglob('*'):
    rel=path.relative_to(root)
    if not path.is_file() or any(p in skip or p.startswith('docx_') for p in rel.parts):continue
    if path.suffix in {'.docx','.pdf','.pyc'} or path.name in {'PRACTICAL_REPORT_HY.md','build_report.py','jupyter_server.log'}:continue
    files.append(path)
with tarfile.open(temporary,'w:gz',compresslevel=1) as archive:
    for path in sorted(files):archive.add(path,arcname='FakeNews_Practical/'+str(path.relative_to(root)),recursive=False)
with tarfile.open(temporary,'r:gz') as archive:
    names=archive.getnames()
    for required in ['Final_Practical.ipynb','data/raw/liar/train.tsv','models/active_models.json','reports/optimization/fakenewsnet/selected.joblib','START_HERE_HY.md']:
        assert 'FakeNews_Practical/'+required in names
temporary.replace(output)
manifest={'archive':output.name,'files':len(files),'bytes':output.stat().st_size,'sha256':hashlib.file_digest(output.open('rb'),'sha256').hexdigest(),'excluded':'Python environment, Word documents, PDF files, document renders and caches'}
(root/'deliverables/Final_Practical_Code.manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest))
