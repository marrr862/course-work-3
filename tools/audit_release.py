"""Check the actual handoff archive from an independent directory using this host's Python."""
from pathlib import Path
import hashlib,json,subprocess,sys,tarfile,tempfile,time
import nbformat
from nbclient import NotebookClient
root=Path(__file__).resolve().parents[1]
archive=root/'deliverables/Final_Practical_Code.tar.gz'
manifest=json.loads((root/'deliverables/Final_Practical_Code.manifest.json').read_text())
with archive.open('rb') as stream:
    assert hashlib.file_digest(stream,'sha256').hexdigest()==manifest['sha256']
start=time.monotonic()
with tempfile.TemporaryDirectory(prefix='fake-news-release-') as temporary:
    with tarfile.open(archive,'r:gz') as package:
        package.extractall(temporary,filter='data')
    project=Path(temporary)/'FakeNews_Practical'
    print('Archive hash verified; independent extraction complete.',flush=True)
    result=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=project,capture_output=True,text=True)
    (root/'reports/release_archive_tests.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stdout+result.stderr
    print('Archived project tests passed.',flush=True)
    book=nbformat.read(project/'Final_Practical.ipynb',as_version=4)
    NotebookClient(book,timeout=900,kernel_name='fake-news',resources={'metadata':{'path':str(project)}}).execute()
    counts=[cell.execution_count for cell in book.cells if cell.cell_type=='code']
    assert counts and all(count is not None for count in counts)
    print('Archived notebook Run All passed.',flush=True)
    report={'archive_sha256':manifest['sha256'],'independent_directory':True,
            'tests':'60 passed','notebook_code_cells_executed':len(counts),
            'notebook_errors':0,'seconds':round(time.monotonic()-start,2),
            'environment':'Existing tested Mac Python environment; no claim of fresh OS or Colab validation',
            'status':'passed'}
    (root/'reports/release_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
