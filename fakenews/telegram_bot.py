"""Telegram bot: send a news headline (optionally with its link) and get the analysis back.

    1. Create a bot with @BotFather and put the token in practical/.env:
           TELEGRAM_BOT_TOKEN=123456789:AA...
    2. .venv/bin/python -m fakenews.telegram_bot

The bot uses long polling (no public server or webhook needed) and calls the same
FastAPI endpoints as the web interface, in-process. Messages are not stored.
"""
import html
import os
import re
import time
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parents[1]
URL_RE = re.compile(r'https?://\S+|www\.\S+', re.I)
NON_LATIN = re.compile(r'[Ѐ-ӿ԰-֏一-鿿]')
MAX_CHARS = 1000

LABELS = {'fake': 'Կեղծ', 'real': 'Իրական', 'pants-fire': 'Ակնհայտ կեղծ', 'false': 'Սխալ', 'barely-true': 'Հիմնականում սխալ',
          'half-true': 'Մասամբ ճիշտ', 'mostly-true': 'Հիմնականում ճիշտ', 'true': 'Ճիշտ'}
MODELS = {'stage1': 'միայն տեքստ (փուլ 1)', 'stage2_text': 'տեքստ (փուլ 2)', 'stage2_domain': 'տեքստ + կայքի տիրույթ (փուլ 2)',
          'stage2_metadata': 'տեքստ + խոսնակի տվյալներ (փուլ 2)', 'stage3_text': 'RoBERTa + տեքստ (փուլ 3)',
          'stage3_domain': 'RoBERTa + տեքստ + կայքի տիրույթ (փուլ 3)'}
VERIFY = {'correction_proposed': 'Աղբյուրը հակասում է պնդմանը, առաջարկվում է ուղղում',
          'supported_by_sources': 'Աղբյուրները հաստատում են պնդումը',
          'refuted_by_sources': 'Աղբյուրները հակասում են պնդմանը',
          'insufficient_evidence': 'Բավարար ապացույց չի գտնվել',
          'conflicting_evidence': 'Աղբյուրները միմյանց հակասում են',
          'needs_review': 'Պահանջվում է լրացուցիչ ստուգում',
          'needs_shorter_claims': 'Ուղարկիր մինչև հինգ կարճ, առանձին պնդում',
          'model_unavailable': 'Ստուգման մոդելը տեղադրված չէ'}

HELP = ('<b>Կեղծ լուրերի ստուգման բոտ</b>\n\n'
        'Ուղարկիր <b>անգլերեն</b> լուրի վերնագիր կամ կարճ պնդում։ Կարող ես ավելացնել նաև լուրի հղումը, '
        'այդ դեպքում հաշվի կառնվի կայքի տիրույթը (էջը չի բեռնվում)։\n\n'
        'Օրինակ՝\n<code>BREAKING: Celebrity couple secretly married in Las Vegas!!! https://hollywoodlife.com/...</code>\n'
        '<code>Mars has no moons.</code>\n\n'
        'Հրամաններ՝\n/liar <i>պնդում</i>՝ քաղաքական պնդման գնահատում LIAR-ի 6 աստիճաններով\n'
        '/help՝ այս հաղորդագրությունը\n\n'
        'Արդյունքը մոդելի գնահատական է, ոչ թե վերջնական փաստ։')


def parse_message(text):
    """Split a message into the headline text and an optional article URL."""
    text = (text or '').strip()
    match = URL_RE.search(text)
    url = match.group(0).rstrip('.,;)') if match else None
    headline = URL_RE.sub(' ', text).strip(' -–|:\n\t') if match else text
    return re.sub(r'\s+', ' ', headline).strip(), url


def check_input(headline):
    if not headline:
        return 'Ուղարկիր նաև լուրի վերնագիրը, ոչ միայն հղումը։'
    if NON_LATIN.search(headline):
        return 'Մոդելն աշխատում է միայն անգլերեն տեքստով։ Ուղարկիր վերնագիրը անգլերեն։'
    if len(re.findall(r'[A-Za-z]', headline)) < 3:
        return 'Տեքստը չափազանց կարճ է։ Ուղարկիր ամբողջական վերնագիր կամ պնդում։'
    if len(headline) > MAX_CHARS:
        return f'Տեքստը շատ երկար է։ Ուղարկիր մինչև {MAX_CHARS} նիշ։'
    return None


def format_classification(result):
    scores = result.get('model_scores') or {}
    pred = result.get('model_prediction')
    if not pred:
        return '<b>Դասակարգում</b>\nՄոդելի բառապաշարում բավարար տվյալ չկա այս տեքստի համար։'
    lines = [f"<b>Դասակարգում</b>\nԿանխատեսում՝ <b>{LABELS.get(pred, pred)}</b>"]
    if pred in scores:
        lines.append(f'Միավոր՝ {scores[pred] * 100:.0f}%')
    lines.append(f"Մոդել՝ {MODELS.get(result.get('model_version'), result.get('model_version'))}")
    if result.get('seen_in_training'):
        lines.append('Այս տեքստը եղել է ուսուցման տվյալներում։')
    return '\n'.join(lines)


def format_verification(result):
    lines = [f"<b>Աղբյուրներով ստուգում</b>\n{VERIFY.get(result.get('status'), result.get('status'))}"]
    for claim in result.get('claims', [])[:3]:
        if claim.get('correction'):
            lines.append(f"Ուղղում՝ <i>{html.escape(claim['correction'])}</i>")
        evidence = claim.get('evidence') or []
        if evidence:
            src = evidence[0].get('source', {})
            title, url = html.escape(src.get('title', 'աղբյուր')), src.get('url', '')
            if url.startswith('https://'):
                lines.append(f'Աղբյուր՝ <a href="{html.escape(url)}">{title}</a>')
    return '\n'.join(lines)


def archive_line(result):
    ev = result.get('evidence') or {}
    if ev.get('status') == 'archived_claim_match':
        return f"<b>Արխիվ (LIAR-PLUS)</b>\nԱյս պնդումը նախկինում ստուգվել է՝ {LABELS.get(ev.get('verdict'), ev.get('verdict'))}։"
    return ''


class Analyzer:
    """Calls the project API in-process (no running server needed)."""

    def __init__(self, client=None):
        if client is None:
            from fastapi.testclient import TestClient
            from .app import app
            client = TestClient(app)
        self.client = client

    def news(self, headline, url=None):
        body = {'text': headline, 'dataset': 'fakenewsnet'}
        if url:
            body['source_url'] = url
        r = self.client.post('/analyze', json=body)
        if r.status_code == 422 and url:  # unusable link: classify the text alone
            r = self.client.post('/analyze', json={'text': headline, 'dataset': 'fakenewsnet'})
        r.raise_for_status()
        cls = r.json()
        v = self.client.post('/verify', json={'text': headline, 'live': True})
        parts = [format_classification(cls)]
        if v.status_code == 200:
            parts.append(format_verification(v.json()))
        parts.append(archive_line(cls))
        parts.append('<i>Արդյունքը մոդելի գնահատական է, ոչ թե վերջնական փաստ։</i>')
        return '\n\n'.join(p for p in parts if p)

    def liar(self, claim):
        r = self.client.post('/analyze', json={'text': claim, 'dataset': 'liar'})
        r.raise_for_status()
        res = r.json()
        return '\n\n'.join(p for p in [format_classification(res), archive_line(res)] if p) + '\n\n<i>LIAR՝ 6 աստիճան, ճշգրտությունը մոտ 28% է․ օգտագործիր միայն որպես ցուցիչ։</i>'


def reply_for(text, analyzer):
    text = (text or '').strip()
    if text in ('/start', '/help') or text.startswith('/start ') or text.startswith('/help@'):
        return HELP
    if text.startswith('/liar'):
        claim = text.split(maxsplit=1)[1].strip() if len(text.split(maxsplit=1)) > 1 else ''
        problem = check_input(claim)
        return problem or analyzer.liar(claim)
    if text.startswith('/'):
        return 'Անհայտ հրաման։ Տե՛ս /help'
    headline, url = parse_message(text)
    problem = check_input(headline)
    return problem or analyzer.news(headline, url)


def load_token():
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    env = ROOT / '.env'
    if not token and env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith('TELEGRAM_BOT_TOKEN='):
                token = line.split('=', 1)[1].strip().strip('"\'')
    if not token:
        raise SystemExit('TELEGRAM_BOT_TOKEN is missing: add it to practical/.env')
    return token


def main():
    token = load_token()
    api = f'https://api.telegram.org/bot{token}'
    http = httpx.Client(timeout=40)
    me = http.get(f'{api}/getMe').json()
    if not me.get('ok'):
        raise SystemExit('Telegram rejected the token: ' + str(me.get('description')))
    http.post(f'{api}/setMyCommands', json={'commands': [{'command': 'help', 'description': 'Ինչպես օգտվել'},
                                                          {'command': 'liar', 'description': 'Քաղաքական պնդման գնահատում'}]})
    print('Bot @' + me['result']['username'] + ' is running. Loading models...', flush=True)
    analyzer = Analyzer()
    print('Ready. Press Ctrl+C to stop.', flush=True)
    offset = None
    while True:
        try:
            updates = http.get(f'{api}/getUpdates', params={'timeout': 30, 'offset': offset}).json().get('result', [])
        except httpx.HTTPError:
            time.sleep(3); continue
        for upd in updates:
            offset = upd['update_id'] + 1
            msg = upd.get('message') or upd.get('edited_message')
            if not msg or 'text' not in msg:
                continue
            chat = msg['chat']['id']
            http.post(f'{api}/sendChatAction', json={'chat_id': chat, 'action': 'typing'})
            try:
                answer = reply_for(msg['text'], analyzer)
            except Exception as error:  # keep the bot alive on unexpected input
                print('error:', type(error).__name__, error, flush=True)
                answer = 'Ներողություն, տեքստը մշակել չհաջողվեց։ Փորձիր կրկին։'
            http.post(f'{api}/sendMessage', json={'chat_id': chat, 'text': answer, 'parse_mode': 'HTML',
                                                   'disable_web_page_preview': True, 'reply_to_message_id': msg['message_id']})


if __name__ == '__main__':
    main()
