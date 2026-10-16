"""Run against start.command: .venv/bin/python tests/browser_smoke.py"""
import json
import os
from pathlib import Path
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[1]
with sync_playwright() as p:
    kwargs = {'headless': True}
    if os.environ.get('CHROMIUM_EXECUTABLE'):
        kwargs['executable_path'] = os.environ['CHROMIUM_EXECUTABLE']
    browser = p.chromium.launch(**kwargs)
    page = browser.new_page(viewport={'width': 1280, 'height': 950})
    errors = []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.goto('http://127.0.0.1:8000')
    page.locator('#text').fill('The Sun is a planet.')
    page.locator('#submit').click()
    page.locator('#result').wait_for(state='visible')
    assert 'star' in page.locator('#correction').inner_text()
    assert page.locator('#sources a').get_attribute('href').startswith('https://science.nasa.gov/')
    page.screenshot(path=str(root / 'reports' / 'desktop_demo.png'), full_page=True)
    page.locator('#text').fill('The Sun is not a planet.')
    page.locator('#submit').click()
    page.wait_for_function("document.getElementById('submit').disabled === false")
    assert 'Բավարար ապացույց չկա' in page.locator('#verification').inner_text()
    page.locator('#archiveExample').click()
    page.locator('#submit').click()
    page.wait_for_function("document.getElementById('submit').disabled === false")
    assert 'արխիվային' in page.locator('#verification').inner_text()
    assert 'արխիվային հիմնավորումը' in page.locator('#caveat').inner_text()
    page.screenshot(path=str(root / 'reports' / 'archive_demo.png'), full_page=True)
    page.locator('#example').click()
    page.locator('#verifyButton').click()
    page.locator('#verificationResult').wait_for(state='visible', timeout=60000)
    assert 'ճշգրտում' in page.locator('#verifyStatus').inner_text()
    assert 'Phobos' in page.locator('#correctedText').inner_text()
    assert page.locator('#result').is_hidden()
    page.screenshot(path=str(root / 'reports' / 'verification_demo.png'), full_page=True)
    page.locator('#decisionMode').select_option('high_recall')
    assert 'ավելի հաճախ' in page.locator('#modeHelp').inner_text()
    scientific=json.loads((root/'reports/reliability/scifact/holdout_predictions.json').read_text())
    sample=next(r for r in scientific if r['predicted']=='refuted_by_sources')
    page.locator('#text').fill(sample['claim'])
    page.locator('#scienceButton').click()
    page.wait_for_function("document.getElementById('scienceButton').disabled === false", timeout=60000)
    assert 'հակասող' in page.locator('#verifyStatus').inner_text()
    assert page.locator('#verificationResult a[href^="/science-source/"]').count()>0
    page.screenshot(path=str(root/'reports/reliability/science_demo.png'),full_page=True)
    page.set_viewport_size({'width': 390, 'height': 844})
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    page.screenshot(path=str(root / 'reports' / 'mobile_demo.png'), full_page=True)
    assert not errors, errors
    (root / 'reports' / 'browser_checks.json').write_text(json.dumps({'source_correction': 'passed', 'negation_abstention': 'passed', 'mobile_horizontal_overflow': False, 'archive_lookup': 'passed', 'nli_source_correction': 'passed', 'scientific_source_verification': 'passed', 'sensitivity_mode': 'passed', 'javascript_errors': errors}, indent=2))
    browser.close()
    print('Browser checks passed: correction, negation, mobile layout, JavaScript.')
