"""Capture report screenshots of the running system.

Start the server first (./start.command or uvicorn fakenews.app:app --port 8000), then:
    .venv/bin/python tools/capture_screens.py
Screenshots are written to reports/screens/.
"""
import html
import os
import subprocess
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/screens'
OUT.mkdir(parents=True, exist_ok=True)
URL = 'http://127.0.0.1:8000'


def wait_idle(page, button):
    page.wait_for_function(f"document.getElementById('{button}').disabled === false", timeout=180000)


def terminal_page(title, command, text):
    body = html.escape(text)
    return f"""<!doctype html><meta charset=utf-8><style>
body{{margin:0;background:#1e1e1e;font:13px/1.45 Menlo,Monaco,monospace;color:#d4d4d4}}
.bar{{background:#323233;color:#ccc;padding:6px 12px;font:12px -apple-system,system-ui}}
.dot{{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:6px}}
pre{{margin:0;padding:14px 16px;white-space:pre}} body{{display:inline-block}} .cmd{{color:#9cdcfe}}</style>
<div class=bar><span class=dot style=background:#ff5f57></span><span class=dot style=background:#febc2e></span><span class=dot style=background:#28c840></span> {html.escape(title)}</div>
<pre><span class=cmd>(.venv) practical % {html.escape(command)}</span>\n{body}</pre>"""


def run(cmd):
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return (result.stdout + result.stderr).strip()


with sync_playwright() as p:
    kwargs = {'headless': True}
    if os.environ.get('CHROMIUM_EXECUTABLE'):
        kwargs['executable_path'] = os.environ['CHROMIUM_EXECUTABLE']
    browser = p.chromium.launch(**kwargs)
    page = browser.new_page(viewport={'width': 1100, 'height': 900}, device_scale_factor=2)

    # 1-2. FakeNewsNet headline with article URL -> stage-2 domain model
    page.goto(URL)
    page.select_option('#dataset', 'fakenewsnet')
    page.fill('#text', 'BREAKING: Celebrity couple secretly married in Las Vegas!!!')
    page.fill('#newsUrl', 'https://hollywoodlife.com/2018/05/05/celebrity-wedding/')
    page.locator('#form').screenshot(path=str(OUT / 'ui_form_fakenewsnet.png'))
    page.click('#submit'); wait_idle(page, 'submit')
    page.evaluate("document.querySelector('#result details').open=true")
    page.locator('#result').screenshot(path=str(OUT / 'ui_result_fakenewsnet.png'))

    # 3. LIAR claim with speaker metadata -> stage-2 metadata model
    page.goto(URL)
    page.select_option('#dataset', 'liar')
    page.fill('#text', 'Says the unemployment rate fell to 4 percent.')
    page.fill('#speaker', 'Barack Obama'); page.fill('#party', 'democrat')
    page.click('#submit'); wait_idle(page, 'submit')
    page.locator('#result').screenshot(path=str(OUT / 'ui_result_liar.png'))

    # 4. Evidence-based verification and correction (local NASA sources)
    page.goto(URL)
    page.fill('#text', 'Mars has no moons.')
    page.click('#verifyButton'); wait_idle(page, 'verifyButton')
    page.evaluate("document.querySelectorAll('#verifiedClaims details').forEach(d=>d.open=true)")
    page.locator('#verificationResult').screenshot(path=str(OUT / 'ui_verify_mars.png'))

    # 5. Live Wikipedia verification
    page.goto(URL)
    page.fill('#text', "France's capital city is Berlin.")
    page.check('#liveSources')
    page.click('#verifyButton'); wait_idle(page, 'verifyButton')
    page.locator('#verificationResult').screenshot(path=str(OUT / 'ui_verify_live.png'))

    # 6. Archive (LIAR-PLUS) exact match
    page.goto(URL)
    page.click('#archiveExample'); page.select_option('#dataset', 'liar')
    page.click('#submit'); wait_idle(page, 'submit')
    page.locator('#result').screenshot(path=str(OUT / 'ui_archive.png'))

    # 7. API documentation
    page.goto(URL + '/docs'); page.wait_for_selector('.opblock', timeout=30000)
    page.screenshot(path=str(OUT / 'api_docs.png'))

    # 8. Notebook cells (rendered from the executed notebook)
    run([sys.executable, '-m', 'jupyter', 'nbconvert', '--to', 'html', 'Final_Practical.ipynb', '--output-dir', str(OUT), '--output', 'notebook'])
    page.set_viewport_size({'width': 1100, 'height': 900})
    page.goto((OUT / 'notebook.html').as_uri())
    # hide library warnings / progress bars (stderr) so the screenshots show the results
    page.add_style_tag(content='[data-mime-type="application/vnd.jupyter.stderr"]{display:none!important}')
    cells = page.locator('.jp-Cell.jp-CodeCell')
    names = {1: 'nb_data', 2: 'nb_baselines', 4: 'nb_stage2', 5: 'nb_analyze', 6: 'nb_verify'}
    for index, name in names.items():
        if index < cells.count():
            cells.nth(index).screenshot(path=str(OUT / f'{name}.png'))

    # 9. Terminal: tests and stage-2 training
    tests = run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:warnings'])
    page.set_content(terminal_page('Terminal — pytest', 'python -m pytest -q', tests))
    page.locator('body').screenshot(path=str(OUT / 'term_pytest.png'))
    stage2 = run([sys.executable, '-m', 'fakenews.improve_v2', '--no-save'])
    keep = [line for line in stage2.splitlines() if 'Warning' not in line]
    page.set_viewport_size({'width': 1100, 'height': 900})
    page.set_content(terminal_page('Terminal — stage 2', 'python -m fakenews.improve_v2 --no-save', '\n'.join(keep)))
    page.locator('body').screenshot(path=str(OUT / 'term_stage2.png'))

    # 10. GitHub repository
    page.set_viewport_size({'width': 1280, 'height': 900})
    try:
        page.goto('https://github.com/marrr862/course-work-3', timeout=30000)
        page.screenshot(path=str(OUT / 'github_repo.png'))
    except Exception as error:
        print('github screenshot skipped:', error)
    browser.close()
print('saved to', OUT)
