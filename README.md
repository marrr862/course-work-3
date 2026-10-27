# Կեղծ լուրերի հայտնաբերման և ճշգրտման համակարգ

> **Repository contents.** The code, tests, notebook (`Final_Practical.ipynb`), datasets, results (`reports/`) and the final course work (`deliverables/Կուրսային աշխատանք 3 (փուլ 4).docx` and `.pdf`) are included. The Python environment and the large neural models (fine-tuned MiniLM, RoBERTa, ELECTRA, DeBERTa NLI, NLLB-200, about 5 GB) are not, because of GitHub's file-size limits. Only small models (`models/*.joblib`, stages 1–4 without neural weights) are included. To rebuild the rest, follow the commands below (`fakenews.train`, `setup_encoder`, `optimize`, `setup_verifier`, `improve_v2`, `train_large`, `train_transformer`, `stack_v4 --build`, `translate --download`).


**Հիմնական գործարկում՝ START_HERE_HY.md → open_notebook.command → Final_Practical.ipynb։** Notebook-ը սկզբից մինչև վերջ գործարկված է։ Պատրաստի վեբ ցուցադրությունը՝ start.command։

Ամբողջական տեղային հետազոտական նախատիպ՝ տվյալների մաքրում, TF-IDF հատկանիշներ, երեք ուսուցվող մոդելի և majority baseline-ի համեմատություն, առանձին թեստային գնահատում, պահպանված մոդելներ, FastAPI, հայերեն վեբ միջերես և Telegram բոտ։ Մոդելները ուսուցված են անգլերենով, իսկ հայերեն և ռուսերեն մուտքը նախ թարգմանվում է անգլերեն (NLLB-200)։

## Արագ մեկնարկ

Տերմինալում՝

```bash
cd /Users/macbook/Desktop/FakeNewsNet-master/practical
./start.command
```

Բացել http://127.0.0.1:8000 ։ API նկարագրությունը՝ http://127.0.0.1:8000/docs ։ Կարելի է նաև Finder-ից գործարկել `start.command` ֆայլը։ Ծառայությունը դադարեցնելու համար՝ Ctrl+C։

Արխիվային ուղու օրինակներ՝ `The Sun is a planet.` (տեղային աղբյուրով ճշգրտում), `The Sun is not a planet.` (ճշգրիտ համընկնում չկա, ինքնաբերաբար չի ուղղվում)։ NASA-ի օրինակները ցուցադրում են ճշգրտման մեխանիզմը, ոչ քաղաքական լուրերի դասակարգման որակը։

## Վերարտադրում

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m fakenews.train --liar /Users/macbook/Desktop/liar_dataset --fakenews /Users/macbook/Desktop/FakeNewsNet-master/dataset
.venv/bin/python -m pytest -q
```

Փորձարկված միջավայրի ճշգրիտ փաթեթները՝ `requirements.txt`, Python տարբերակն ու տվյալների SHA-256-երը՝ `reports/reproducibility.json`։ Ուսուցումն աշխատում է CPU-ով։ `train.command`-ը կատարում է նույն ուսուցումը լռելյայն հարակից պանակներով։

## Կառուցվածք

- `fakenews/data.py` — Unicode/HTML/URL մշակում, կրկնություններ, հակասող պիտակներ և բաժանումներ։
- `fakenews/train.py` — TF-IDF, majority, ComplementNB, LogisticRegression, LinearSVC, validation ընտրություն և test գնահատում։
- `fakenews/evidence.py` և `evidence.json` — աղբյուրներով ճշգրտում և Google Fact Check որոնում։
- `fakenews/app.py`, `index.html` — ծառայություն և միջերես։
- `fakenews/verify.py`, `sources.py` — NLI, աղբյուրների ստացում և նոր պնդումների ուղղում։
- `fakenews/setup_verifier.py`, `evaluate_verifier.py` — մոդելի տեղադրում և առանձին գնահատում։
- `models/` — երկու ուսուցված մոդել։ Մի բեռնեք անվստահելի joblib ֆայլեր։
- `reports/` — տվյալների աուդիտ, բաժանման մանիֆեստ, համեմատություններ, կանխատեսումներ, սխալներ, գրաֆիկներ, չափումներ և վերարտադրման տվյալներ։
- `PRACTICAL_REPORT_HY.md` — գործնական արդյունքների հայերեն բացատրություն՝ հետագա Word փաստաթղթի հիմք։
- `experiments.ipynb` — պատրաստի արդյունքների ուսումնասիրման նոթատետր։
- `tests/test_system.py` — տվյալների արտահոսքի, գնահատման, ծառայության և ճշգրտման ստուգումներ։

## Գնահատման կանոններ

LIAR-ի պաշտոնական train/valid/test բաժանումները պահպանվում են՝ կրկնությունների մաքրումից հետո։ Հակասող տեքստերի բոլոր օրինակները հանվում են; այլ կրկնությունների դեպքում առաջնահերթությունը train→valid→test է։ FakeNewsNet-ը բաժանվում է 70/15/15՝ աղբյուրի և պիտակի համատեղ շերտավորմամբ, seed=42։ Այդ նախնական ընդհանուր մաքրումը հղումային բաժանման չափերը փոխում է, ուստի արդյունքները ուղղակի համադրելի չեն այլ հոդվածների հետ։

TF-IDF-ը սովորում է միայն train-ում։ Ընտրությունը կատարվում է validation macro-F1-ով։ Ընտրված մոդելը չի վերաուսուցվում valid-ի վրա և մեկ անգամ գնահատվում է test-ում։ Bootstrap միջակայքը վերաբերում է test նմուշների փոփոխականությանը, ոչ ուսուցման seed-երի կամ բոլոր հնարավոր աղբյուրների անորոշությանը։

FakeNewsNet-ում առկա են վերնագրեր, ոչ ամբողջ հոդվածները։ URL-ները, tweet ID-ները և LIAR-ի խոսնակի պատմությունը չեն օգտագործվում որպես հատկանիշներ։ Նորմալացված ճշգրիտ կրկնությունները բացառվում են; իմաստային վերաշարադրումների, նույն իրադարձությունների կամ ժամանակային արտահոսքի ամբողջական բացառում չի երաշխավորվում։

## Նոր պնդումների ստուգում և ուղղում

`POST /verify` իրականացնում է աղբյուրների որոնում → հատվածների ընտրություն → DeBERTa NLI → աղբյուրից վերցված ուղղման առաջարկ։ Մուտքը՝ մինչև հինգ կարճ անգլերեն պնդում։ Օրինակ՝

```json
{"text":"Mars has no moons.","live":false,"source_urls":[]}
```

Միջերեսում ընտրել «Ստուգել ապացույցներով և ճշգրտել»։ `live=true`-ը որոնում է Wikipedia-ում՝ առանց API բանալու։ Օրինակ՝ `France's capital city is Berlin.`։ `source_urls`-ը կարող է պարունակել մինչև երեք NASA կամ անգլերեն Wikipedia հոդվածի HTTPS հղում։ Մուտքի բառերն առցանց ռեժիմում փոխանցվում են MediaWiki-ին։

Ուղղումը վերցվում է աղբյուրի բառացի նախադասությունից՝ վերնագրով և հղումով։ Պատասխանը պարունակում է յուրաքանչյուր պնդման `status`, `evidence`, `correction`, ինչպես նաև ամբողջական `corrected_text`՝ երբ բոլոր մասերը վճռորոշ գնահատված են։ Հաստատումն ու հերքումը մոդելի գնահատականներ են տվյալ աղբյուրների համեմատ։ Հակասող կամ անբավարար ապացույցների դեպքում ծրագիրը ձեռնպահ է մնում։ NLI-ի score-ները ճշմարտության հավանականություններ չեն։

## Ապացույցների բաղադրիչների վերարտադրում

```bash
.venv/bin/python -m fakenews.build_evidence
.venv/bin/python -m fakenews.setup_verifier
.venv/bin/python -m fakenews.evaluate_evidence
.venv/bin/python -m fakenews.evaluate_verifier --split development
.venv/bin/python -m fakenews.evaluate_verifier --split test
```

NLI մոդելը մոտ 0.6 GB է և տեղադրվում է տեղային `models/nli_deberta` պանակում՝ ամրագրված Hugging Face վերանայումից։ Այն հեղինակների կողմից նախապես ուսուցված է SNLI/MultiNLI-ով։ Մեր LIAR/FakeNewsNet դասակարգիչները ուսուցվել են առանձին։ Setup-ը պահպանում է NASA-ի տասը էջի աղբյուրային պատճենները։

Արխիվային բազայում առկա է LIAR-PLUS-ի 10,090 գրառում՝ միայն մաքրված LIAR train-ին համապատասխանող պնդումներով։ Թեստային պնդումները ինդեքսում չեն։ `/analyze`-ը շարունակում է վերադարձնել վիճակագրական կանխատեսումը և արխիվային որոնումը, իսկ `/verify`-ը նոր պնդումների ապացույցներով գնահատումն է։

## Իրական արդյունքներ և սահմանափակումներ

- Նոր NLI շղթայի 24 օրինակով թեստ՝ 75% accuracy, 58.75% macro-F1՝ ներառելով conflict ելքը։
- Առաջարկվել է վեց ուղղում. վեցն էլ հերքվող պնդումների համար են և բառացի պահպանել են աղբյուրի հատվածը։ Սա փոքր նմուշ է, ոչ 100% ընդհանուր որակի ապացույց։
- Մեկ սխալ հաստատում կա՝ `Mercury has a moon.`։ Ամբողջ սխալների ցանկը՝ `reports/verification/errors.json`։
- 20 ձեռքով վերաշարադրված արխիվային հարցման որոնումը՝ Recall@1=80%, Recall@5=100%։
- Անցել են բոլոր 108 ավտոմատ թեստերը և բրաուզերի փորձարկումը։
- Առցանց ստուգումը իրականում փորձարկվել է Ֆրանսիայի մայրաքաղաքի ճիշտ և սխալ պնդումներով՝ `reports/live_verification.json`։

Աղբյուրները կարող են հնանալ, որոնումն ու NLI-ն կարող են սխալվել, իսկ կորպուսից դուրս կամ բարդ պնդումները կարող են մնալ չստուգված։ Մոդելները հայերենով չեն ուսուցանվել․ հայերեն և ռուսերեն տեքստը մեքենայական թարգմանվում է, ինչը կարող է փոխել արդյունքը։ Google Fact Check-ի ընտրովի միակցիչի համար անհրաժեշտ է `GOOGLE_FACT_CHECK_API_KEY`․ այն իրական բանալիով չի փորձարկվել և չի պահանջվում հիմնական շղթայի համար։

## Միջերեսի փորձարկում

```bash
.venv/bin/python -m playwright install chromium
.venv/bin/python tests/browser_smoke.py
```

Ծառայությունը նախ պետք է աշխատի։ Կարելի է օգտագործել արդեն տեղադրված Chromium-ը՝ `CHROMIUM_EXECUTABLE` միջավայրի փոփոխականով։ Փորձարկվում են ուղղումը, փոփոխված պնդման մերժումը, արխիվային որոնումը, հեռախոսի էկրանի լայնությունը և JavaScript սխալները։

## Աղբյուրներ

- https://huggingface.co/cross-encoder/nli-deberta-v3-small
- https://www.mediawiki.org/wiki/API:Search

- https://github.com/Tariq60/LIAR-PLUS — հիմնավորումների լրացուցիչ հետազոտական բազա։
- https://github.com/KaiDMML/FakeNewsNet — տվյալների կառուցվածք և սահմանափակումներ։
- LIAR-ի տեղային README — 14 սյունակ, 6 դաս, հետազոտական օգտագործման սահմանափակում։
- https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- https://developers.google.com/fact-check/tools/api/reference/rest/v1alpha1/claims/search
- https://science.nasa.gov/sun/facts/
- https://science.nasa.gov/earth/facts/

Սկզբնական տվյալները և նախկին նոթատետրը չեն փոփոխվել։ Գործնական մասի Word տարբերակը՝ `deliverables/Practical_Report_HY.docx`։


## Բարելավված դասակարգիչներ

Ակտիվ տարբերակը յուրաքանչյուր հավաքածուի համար validation-ով ընտրված 50/50 համադրություն է՝ տեղում լրացուցիչ ուսուցված MiniLM և տեքստային դասակարգիչ։ FakeNewsNet test accuracy՝ 85.71%, macro-F1՝ 79.56%։ LIAR test accuracy՝ 26.31%, macro-F1՝ 26.38%։ Սրանք կրկնակի գնահատումներ են նախկին թեստային բաժանման վրա։ «Fake» recall-ի նվազումը և մյուս սահմանափակումները մանրամասն ներկայացված են `reports/optimization/RESULTS_HY.md`-ում։ Գործող փաստաստուգման մոդելը չի փոխվել։

```bash
.venv/bin/python -m fakenews.setup_encoder
.venv/bin/python -m fakenews.improve
.venv/bin/python -m fakenews.optimize
.venv/bin/python -m fakenews.evaluate_optimized
.venv/bin/python -m fakenews.activate_optimized
```

Նախ պետք է գոյություն ունենան սկզբնական ուսուցման մոդելներն ու բաժանման մանիֆեստները։ Լեզվային մոդելի ուսուցումն օգտագործում է MPS, եթե հասանելի է, այլապես CPU։ Նոր գործարկումների թվերը կարող են փոքր-ինչ տարբերվել՝ սարքի և ոչ դետերմինիստական գործողությունների պատճառով։ Ծառայությունը վերագործարկելուց հետո `models/active_models.json`-ը որոշում է ակտիվ մոդելները։ Սկզբնական տարբերակին վերադառնալու համար այդ JSON-ում կարելի է նշել `models/liar.joblib` և `models/fakenewsnet.joblib` ուղիները և վերագործարկել ծառայությունը։

### Փուլ 2՝ stacking (կուրսային աշխատանքի վերջնական մոդելները)

Փուլ 1-ի համադրության հավանականությունները միավորվում են բառային և նիշային TF-IDF-ով լոգիստիկ ռեգրեսիայի հետ՝ `p = α·p_stage1 + (1−α)·p_LR`։ FakeNewsNet-ի համար կա երկու տարբերակ՝ միայն տեքստ (α=0.8, շեմ 0.50) և տեքստ + լուրի կայքի տիրույթ (α=0.5, շեմ 0.55)։ LIAR-ի համար LR-ը օգտագործում է խոսնակի մետատվյալները և միայն train-ից հաշվարկված խոսնակի պատմությունը (α=0.6)։ Բոլոր ընտրությունները կատարված են validation-ով։

| Մոդել | Test accuracy | Test macro-F1 |
|---|---:|---:|
| FakeNewsNet, միայն տեքստ | 85.80% | 79.82% |
| FakeNewsNet, տեքստ + տիրույթ | 87.43% | 81.77% |
| LIAR 6 դաս, տեքստ + մետատվյալներ | 27.97% | 28.33% |

```bash
.venv/bin/python -m fakenews.improve_v2      # ուսուցում, գնահատում, models/*_v2.joblib, reports/v2/
.venv/bin/python -m pytest -q tests/test_stage2.py
```

API-ն ինքնաշխատ ընտրում է մոդելը․ `/analyze` հարցման մեջ `source_url` (FakeNewsNet) կամ `metadata` (LIAR՝ speaker, party, state, job, context, subject) դաշտերի առկայության դեպքում օգտագործվում է փուլ 2-ի համապատասխան մոդելը։ Հղման էջը չի բեռնվում, օգտագործվում է միայն տիրույթը։ `balanced` և `high_recall` ռեժիմները, ինչպես նաև `"model": "stage1"` պարամետրը, պահպանում են փուլ 1-ի գնահատված մոդելը։ Պատասխանի `model_version` դաշտը ցույց է տալիս, թե որ մոդելն է օգտագործվել։ Փուլ 2-ի արտեֆակտները չեն պարունակում նեյրոնային կշիռները․ դրանք բեռնում են փուլ 1-ի մոդելը `models/active_models.json`-ով։


### Փուլ 3՝ RoBERTa-base

`roberta-base` մոդելը լրացուցիչ ուսուցվել է FakeNewsNet-ի վերնագրերով (MPS, 4 epoch, լավագույնը՝ 2-րդը) և համադրվել է փուլ 1-ի և փուլ 2-ի բաղադրիչների հետ։ Կշիռներն ու շեմը ընտրված են validation-ով։

| Մոդել | Test accuracy | Test macro-F1 |
|---|---:|---:|
| Փուլ 3, միայն տեքստ | 86.05% | 80.77% |
| Փուլ 3, տեքստ + տիրույթ | 87.90% | 82.65% |

```bash
.venv/bin/python -m fakenews.train_large --dataset fakenewsnet --model roberta-base   # ~25 րոպե M4-ի վրա
.venv/bin/python -m fakenews.train_large --dataset fakenewsnet --build-only           # միայն արտեֆակտը
```

### Փուլ 4՝ մետա-մոդել (աշխատող տարբերակը)

Լրացուցիչ ուսուցված են ELECTRA-base (վերնագիր), RoBERTa-base («վերնագիր + հղման բառեր» զույգ մուտք) և LIAR-ի համար RoBERTa-base («պնդում + խոսնակի տվյալներ»)։ FakeNewsNet-ում բոլոր բաղադրիչների «կեղծ» հավանականությունների logit-ները միավորվում են լոգիստիկ մետա-մոդելով, որի կշիռներն ու շեմը ընտրված են validation-ի վրա 5-ապատիկ cross-validation-ով։ LIAR-ում (խոսնակի տվյալների առկայության դեպքում) օգտագործվում է փուլ 1-ի, RoBERTa-ի և փուլ 2-ի LR-ի հավանականությունների միջինը։

| Մոդել | Test accuracy | Test macro-F1 |
|---|---:|---:|
| FakeNewsNet, միայն տեքստ (4 բաղադրիչ) | 86.08% | 80.90% |
| FakeNewsNet, տեքստ + հղում (7 բաղադրիչ) | 88.70% | 83.54% |
| LIAR 6 դաս, տեքստ + մետատվյալներ | 28.61% | 28.87% |

Տեքստ + հղում տարբերակի աճը փուլ 2-ի համեմատ վիճակագրորեն նշանակալի է (macro-F1-ի 95% bootstrap միջակայք՝ [+0.65; +2.99] տոկոսային կետ)։ RoBERTa-large-ը և kNN հատկանիշները փորձարկվել են, բայց աշխատող տարբերակում չեն ներառվել։

```bash
.venv/bin/python -m fakenews.train_transformer --dataset fakenewsnet --model google/electra-base-discriminator --tag electra
.venv/bin/python -m fakenews.train_transformer --dataset fakenewsnet --model roberta-base --input url_title --tag rb_url
.venv/bin/python -m fakenews.train_transformer --dataset liar --model roberta-base --input liar_meta --tag rb_meta --max-len 128
.venv/bin/python -m fakenews.stack_v4 --build            # models/fakenewsnet_v4.joblib, models/liar_v4.joblib
```

API-ն ինքնաշխատ նախընտրում է փուլ 4-ը (`model_version`՝ `stage4_text`, `stage4_domain` կամ `stage4_metadata`)։

### Հայերեն և ռուսերեն մուտք

Լեզուն որոշվում է այբուբենով, իսկ թարգմանությունը կատարվում է տեղում՝ `facebook/nllb-200-distilled-600M` մոդելով (`models/mt_nllb`)։ Պատասխանում վերադարձվում են `detected_language`, `translated_text` և `analyzed_text` դաշտերը։ Շրջադարձային գնահատում (300 թեստային վերնագիր, EN → HY/RU → EN)՝ macro-F1 82.50% (անգլերեն), 78.13% (հայերեն), 79.90% (ռուսերեն)։

```bash
.venv/bin/python -m fakenews.translate --download         # ~2.3 GB
.venv/bin/python -m fakenews.eval_translation             # reports/translation/
```

### Կանխատեսման բացատրություն

`/analyze` հարցման մեջ `"explain": true`-ն վերադարձնում է `explanation` դաշտը՝ ամենաազդեցիկ բառերը։ Յուրաքանչյուր բառի ազդեցությունը հաշվարկվում է այդ բառը հեռացնելով (`Δ = P(պատասխան | տեքստ) − P(պատասխան | տեքստ առանց բառի)`)՝ հենց աշխատող մոդելի վրա (`fakenews/explain.py`)։

### Telegram բոտ

1. @BotFather-ով ստեղծիր բոտ և token-ը գրիր `.env` ֆայլում՝ `TELEGRAM_BOT_TOKEN=...` (ֆայլը git-ում չի պահվում)։
2. Գործարկիր `./telegram_bot.command` կամ `.venv/bin/python -m fakenews.telegram_bot`։
3. Ինքնաշխատ գործարկման համար՝ `bash tools/install_bot_service.sh`։ Մուտք գործելիս բոտը բացվում է Terminal-ում (`tools/bot_forever.command`), կանգ առնելու դեպքում վերագործարկվում է և թույլ չի տալիս Mac-ին քնել։ Հեռացնել՝ `bash tools/install_bot_service.sh remove`։

Բոտը ընդունում է վերնագիր անգլերեն, հայերեն կամ ռուսերեն (ցանկության դեպքում՝ հղումով) և հայերենով վերադարձնում է թարգմանությունը, դասակարգումը, ամենաազդեցիկ բառերը, աղբյուրներով ստուգումը և արխիվային համընկնումը։ `/liar <պնդում>` հրամանը գնահատում է քաղաքական պնդումը։ Բոտն աշխատում է long polling-ով, հաղորդագրությունները չի պահում։

### Reliability modes and scientific evidence

The default FakeNewsNet threshold remains 0.50. `decision_mode` on `/analyze` also accepts `balanced` (0.47) or `high_recall` (0.30). Thresholds were selected on validation only. The high-recall mode catches more fake headlines but raises more false alarms; model scores are not calibrated truth probabilities. See `reports/reliability/decision_modes_test.json` for the full tradeoff. Coverage curves are descriptive and are not deployed as confidence guarantees.

`/verify-science` accepts one English scientific claim (up to 100 words), searches 5,183 fixed SciFact abstracts, and evaluates source support/contradiction. Source links open local corpus abstracts with dataset provenance. It does not generate replacement scientific claims. This is a fixed research corpus, not a current-news or clinical service. The scientific model is separate from the general NASA/Wikipedia verifier.

Reproduce after installing the original NLI and MiniLM assets:

```sh
.venv/bin/python -m fakenews.setup_science
.venv/bin/python -m fakenews.benchmark_science --stage tune
.venv/bin/python -m fakenews.train_science_nli
.venv/bin/python -m fakenews.benchmark_science --stage tune --model-path models/nli_science
.venv/bin/python -m fakenews.benchmark_science --stage evaluate
.venv/bin/python -m fakenews.reliability_modes
```

Use a fresh experiment output directory when changing weights or data: benchmark score files are cached. Official SciFact train is split into 659 training and 150 tuning claims. Official dev (300 claims) is reserved as this project's final holdout; it is public, and pretraining overlap is unknown. Reported claim-level scoring is not the official SciFact evidence-set leaderboard metric. Retrieval never receives gold evidence IDs. Assumed-neutral non-gold training passages may contain unannotated evidence.

Context-window and refreshed-source experiments did not improve the small general-verifier test. They remain experimental and are not the default `/verify` path. Original source snapshots and models are preserved.
