"""Screenshots for the stage-4 / translation / explanation part of the report.

Start the server first (uvicorn fakenews.app:app --port 8000), then:
    .venv/bin/python tools/capture_v4.py
"""
import html
import os
import re
import subprocess
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/screens'
OUT.mkdir(parents=True, exist_ok=True)
URL = 'http://127.0.0.1:8000'
sys.path.insert(0, str(ROOT))


def terminal_page(title, command, text):
    return f"""<!doctype html><meta charset=utf-8><style>
body{{margin:0;background:#1e1e1e;font:13px/1.45 Menlo,Monaco,monospace;color:#d4d4d4;display:inline-block}}
.bar{{background:#323233;color:#ccc;padding:6px 12px;font:12px -apple-system,system-ui}}
.dot{{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:6px}}
pre{{margin:0;padding:14px 16px;white-space:pre-wrap;max-width:900px}} .cmd{{color:#9cdcfe}}</style>
<div class=bar><span class=dot style=background:#ff5f57></span><span class=dot style=background:#febc2e></span><span class=dot style=background:#28c840></span> {html.escape(title)}</div>
<pre><span class=cmd>(.venv) practical % {html.escape(command)}</span>\n{html.escape(text)}</pre>"""


def run(cmd):
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return (result.stdout + result.stderr).strip()


def ui(page, name, text, url=None):
    page.goto(URL)
    page.select_option('#dataset', 'fakenewsnet')
    page.fill('#text', text)
    if url:
        page.fill('#newsUrl', url)
    page.click('#submit')
    page.wait_for_function("document.getElementById('submit').disabled === false", timeout=180000)
    page.locator('#result').screenshot(path=str(OUT / name))


BOT = r'''
from fakenews.telegram_bot import Analyzer, reply_for
import re
a = Analyzer()
for msg in ["Հայտնի դերասանուհին գաղտնի ամուսնացել է Լաս Վեգասում", "BREAKING: Celebrity couple secretly married in Las Vegas!!! https://hollywoodlife.com/2018/05/05/celebrity-wedding/"]:
    print(">>> " + msg)
    print(re.sub(r"<[^>]+>", "", reply_for(msg, a)).replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&"))
    print()
'''

with sync_playwright() as p:
    kwargs = {'headless': True}
    if os.environ.get('CHROMIUM_EXECUTABLE'):
        kwargs['executable_path'] = os.environ['CHROMIUM_EXECUTABLE']
    browser = p.chromium.launch(**kwargs)
    page = browser.new_page(viewport={'width': 1100, 'height': 900}, device_scale_factor=2)
    ui(page, 'ui_v4_armenian.png', 'Հայտնի դերասանուհին գաղտնի ամուսնացել է Լաս Վեգասում')
    ui(page, 'ui_v4_url.png', 'BREAKING: Celebrity couple secretly married in Las Vegas!!!', 'https://hollywoodlife.com/2018/05/05/celebrity-wedding/')
    bot = run([sys.executable, '-c', BOT])
    keep = '\n'.join(l for l in bot.splitlines() if not re.search(r'warn|Loading weights|max_new_tokens|noqa', l, re.I))
    page.set_content(terminal_page('Terminal — Telegram bot reply', 'python -c "reply_for(...)"', keep))
    page.locator('body').screenshot(path=str(OUT / 'term_bot.png'))
    tests = run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:warnings'])
    page.set_content(terminal_page('Terminal — pytest', 'python -m pytest -q', tests))
    page.locator('body').screenshot(path=str(OUT / 'term_pytest.png'))
    tr = (ROOT / 'reports/translation/round_trip_metrics.json').read_text()
    page.set_content(terminal_page('Terminal — translation evaluation', 'python -m fakenews.eval_translation', tr))
    page.locator('body').screenshot(path=str(OUT / 'term_translation.png'))
    browser.close()
print('saved to', OUT)
