"""Build the self-contained teaching/demo entry point; existing experiments stay intact."""
from pathlib import Path
import nbformat as nb
root=Path(__file__).resolve().parents[1]
cells=[]
def md(s):cells.append(nb.v4.new_markdown_cell(s))
def code(s):cells.append(nb.v4.new_code_cell(s))
md('''# Կեղծ լուրերի հայտնաբերում և աղբյուրներով ճշգրտում
## Գործնականի վերջնական նոթատետր
Գործարկել **Run → Run All Cells**։ Այս նոթատետրը իրականում մաքրում է տվյալները և ուսուցում սկզբնական համեմատական մոդելները։ Վերջնական, ավելի երկար ուսուցված մոդելները բեռնվում են պահպանված կշիռներից։ Դրանք կրկին ուսուցելու կարիք չկա ցուցադրության համար։

Միջերեսն ու բացատրությունները հայերեն են։ **Տվյալները և վերլուծվող տեքստերը անգլերեն են․ թարգմանված ուսուցման հավաքածու չունենք։** Սովորական գործարկումը առցանց ծառայություն չի կանչում, չի վերագրում վերջնական մոդելները և վճարովի բանալի չի պահանջում։''')
code('''from pathlib import Path
import sys, json, time
import numpy as np
import pandas as pd
from IPython.display import display, Image
ROOT = Path.cwd()
if not (ROOT / 'fakenews').is_dir():
    raise RuntimeError('Բացիր notebook-ը practical պանակից՝ open_notebook.command-ով։')
sys.path.insert(0, str(ROOT))
print('Նախագիծ՝', ROOT)
print('Python՝', sys.version.split()[0])
for asset in ['data/raw/liar/train.tsv','data/raw/fakenewsnet/politifact_fake.csv',
              'models/active_models.json','models/nli_deberta/model.safetensors']:
    assert (ROOT / asset).exists(), f'Բացակայում է՝ {asset}'
print('Անհրաժեշտ հիմնական ֆայլերն առկա են։')''')
md('''## 1. Տվյալների բեռնում և նախնական մշակում
LIAR-ը ունի վեց դաս, FakeNewsNet-ը՝ fake/real վերնագրեր։ Հեռացվում են դատարկ, կրկնվող և հակասող պիտակներով տեքստերը։ LIAR-ի պաշտոնական բաժանումը պահպանվում է մաքրումից հետո, FakeNewsNet-ը բաժանվում է 70/15/15։ Հատկանիշների բառապաշարը սովորում է միայն train-ից։''')
code('''from fakenews.data import load_liar, load_fakenews, assert_disjoint, clean_text
frames, audits = {}, {}
frames['liar'], audits['liar'] = load_liar(ROOT/'data/raw/liar')
frames['fakenewsnet'], audits['fakenewsnet'] = load_fakenews(ROOT/'data/raw/fakenewsnet')
for frame in frames.values(): assert_disjoint(frame)
display(pd.DataFrame(audits).T)
display(pd.DataFrame({name: frame.groupby('split').size() for name,frame in frames.items()}))
print('Մաքրման օրինակ՝', clean_text('<p>Breaking NEWS!</p> https://example.org'))
display(frames['fakenewsnet'][['text','label','split']].head())''')
md('''## 2. Հատկանիշներ և իրական ուսուցում
Ստորև ամբողջ մաքրված train-ով ուսուցվում են TF-IDF + majority, Naive Bayes, Logistic Regression և SVM տարբերակները՝ երկու հավաքածուների համար։ Սա կարող է տևել մի քանի րոպե։ Ընտրությունը կատարվում է միայն validation macro-F1-ով։ Վերջնական ակտիվ մոդելները չեն փոխվում։''')
code('''from fakenews.train import candidates, pipeline, metrics
baseline_models, baseline_results = {}, []
for name, frame in frames.items():
    train = frame[frame.split=='train']
    valid = frame[frame.split=='valid']
    fitted, comparisons = {}, []
    for model_name, estimator in candidates().items():
        started = time.perf_counter()
        model = pipeline(estimator).fit(train.text, train.label)
        fitted[model_name] = model
        comparisons.append({'model':model_name, **metrics(valid.label, model.predict(valid.text))})
        print(name, model_name, round(time.perf_counter()-started,1), 'վայրկյան')
    comparison = pd.DataFrame(comparisons).sort_values('macro_f1', ascending=False)
    display(comparison[['model','accuracy','macro_f1']])
    winner = comparison.iloc[0]['model']
    baseline_models[name] = fitted[winner]
    test = frame[frame.split=='test']
    baseline_results.append({'dataset':name,'selected':winner,
        **metrics(test.label, fitted[winner].predict(test.text))})
display(pd.DataFrame(baseline_results)[['dataset','selected','accuracy','macro_f1']])
vectorizer = baseline_models['fakenewsnet'].named_steps['tfidf']
print('TF-IDF հատկանիշների քանակը՝', len(vectorizer.get_feature_names_out()))
print('Հատկանիշների օրինակներ՝', vectorizer.get_feature_names_out()[:20])''')
md('''## 3. Վերջնական բարելավված մոդելները
Վերջնական մոդելը MiniLM-ի լրացուցիչ ուսուցված դասակարգչի և ոչ նեյրոնային մոդելի 50/50 համադրություն է։ Յուրաքանչյուր հավաքածուի համար նախկինում համեմատվել է 30 տարբերակ՝ validation-ով։ Այստեղ դիտում ենք ուսուցման պատմությունը և վերահաշվում պահպանված թեստային կանխատեսումների չափումները։ Այս բջիջը **չի վերաուսուցում** լեզվային մոդելները։''')
code('''final_rows=[]
for name in ['liar','fakenewsnet']:
    folder=ROOT/'reports/optimization'/name
    saved=json.loads((folder/'test_metrics.json').read_text())
    predictions=pd.read_csv(folder/'test_predictions.csv')
    actual=metrics(predictions.label, predictions.prediction)
    assert np.isclose(actual['accuracy'], saved['accuracy'])
    final_rows.append({'dataset':name,'test_count':len(predictions),
                       'accuracy':actual['accuracy'],'macro_f1':actual['macro_f1']})
    print(name, 'ուսուցման պատմություն')
    display(pd.DataFrame(json.loads((folder/'training_history.json').read_text())))
display(pd.DataFrame(final_rows))
display(Image(filename=str(ROOT/'reports/optimization/fakenewsnet/confusion_matrix.png'),width=620))''')
md('''## 3ա. Փուլ 2՝ նիշային n-gram-ներ, աղբյուրի տիրույթ, LIAR-ի մետատվյալներ
Փուլ 1-ի նեյրոնային համադրության հավանականությունները համադրվում են նոր լոգիստիկ ռեգրեսիայի հետ՝ p = α·p_neural + (1−α)·p_LR։ α կշիռը և որոշման շեմը ընտրված են միայն validation-ով։ FakeNewsNet-ի «տիրույթ» մոդելը օգտագործվում է միայն այն դեպքում, երբ հայտնի է լուրի հղումը։ Վերաուսուցման համար՝ `python -m fakenews.improve_v2`։ Այս բջիջը վերաուսուցում չի կատարում․ այն վերահաշվում է պահպանված մոդելի թեստային կանխատեսումները։''')
code('''import joblib
v2_results=pd.read_csv(ROOT/'reports/v2/fakenewsnet_v2_results.csv')
display(v2_results[['model','threshold','val_macro_f1','accuracy','macro_f1','fake_precision','fake_recall']])
display(pd.read_csv(ROOT/'reports/v2/liar_v2_results.csv'))
v2=joblib.load(ROOT/'models/fakenewsnet_v2.joblib')
manifest=pd.read_csv(ROOT/'reports/fakenewsnet/split_manifest.csv'); test_rows=manifest[manifest.split=='test'].reset_index(drop=True)
urls=pd.concat([pd.read_csv(ROOT/f'data/raw/fakenewsnet/{s}_{l}.csv',usecols=['id','news_url']) for s in ['politifact','gossipcop'] for l in ['fake','real']]).drop_duplicates('id')
test_rows=test_rows.merge(urls,on='id',how='left')
stage1_probs=np.load(ROOT/'reports/optimization/fakenewsnet/test_probabilities.npy')
for kind in ['text','domain']:
    model=v2['models'][kind]
    pred=model.decide(model.combine(stage1_probs, model.features.frame(test_rows.text.tolist(), test_rows.news_url.tolist())))
    m=metrics(test_rows.label, pred)
    print(kind, 'α =', model.alpha, 'շեմ =', model.threshold, 'accuracy =', round(m['accuracy'],4), 'macro-F1 =', round(m['macro_f1'],4))
display(Image(filename=str(ROOT/'reports/v2/fig_fnn_v2.png'),width=620))''')
md('''## 4. Քո նոր տեքստի դասակարգումը
Փոխիր միայն `MY_TEXT`-ը և գործարկիր բջիջը։ `standard`, `balanced`, `high_recall` ռեժիմները փոխում են դասակարգման շեմը։ Զգայուն ռեժիմը բաց է թողնում ավելի քիչ կեղծ լուրեր, բայց ավելացնում է սխալ ահազանգերը։ Դասակարգչի կանխատեսումը փաստացի ճշմարտության ապացույց չէ։''')
code('''from fastapi.testclient import TestClient
from fakenews.app import app
client=TestClient(app)
MY_TEXT = 'Scientists announce a new discovery about Mars.'
MODE = 'standard'
SOURCE_URL = ''   # ոչ պարտադիր՝ օրինակ https://www.example.com/article (միայն տիրույթն է օգտագործվում)
body={'text':MY_TEXT,'dataset':'fakenewsnet','decision_mode':MODE}
if SOURCE_URL: body['source_url']=SOURCE_URL
response=client.post('/analyze',json=body)
response.raise_for_status()
result=response.json()
display({key:result[key] for key in ['model_prediction','model_version','decision_mode','decision_threshold','model_scores','seen_in_training']})
display(pd.DataFrame(json.loads((ROOT/'reports/reliability/decision_modes_test.json').read_text())).T)
print('Միավորները ճշմարտության հավանականություններ չեն։')''')
md('''## 5. Աղբյուրով ստուգում և ուղղման առաջարկ
Սա դասակարգումից առանձին գործողություն է։ Համակարգը որոնում է աղբյուրային հատված, գնահատում դրա կապը պնդման հետ և հնարավոր ուղղումը վերցնում աղբյուրից։ `live=False`՝ տեղային NASA աղբյուրներ։ Առցանց որոնումը ընտրովի է և ցուցադրության համար պետք չէ։''')
code('''CLAIM = 'Mars has no moons.'
response=client.post('/verify',json={'text':CLAIM,'live':False})
response.raise_for_status()
verification=response.json()
print('Ելք՝', verification['status'])
print('Ուղղման առաջարկ՝', verification.get('corrected_text'))
for claim in verification['claims']:
    for evidence in claim.get('evidence',[])[:2]:
        print(evidence.get('text',''))
        print(evidence.get('source',{}).get('url',''))''')
md('''## 6. Լրացուցիչ գիտական ռեժիմ՝ փորձնական
Գիտական տարբերակը առանձին ուսուցվել է SciFact-ով։ 659 ուսուցման պնդում, 150 կարգավորման պնդում և 300 առանձին գնահատման պնդում։ Ամբողջ շղթայի accuracy-ն 56.33% է․ այն **չի նշանակում ընդհանուր նորությունների ստուգման ճշտություն**։ Գիտական ռեժիմը ցույց է տալիս աղբյուրներ, բայց ինքնաշխատ նոր գիտական ձևակերպում չի ստեղծում։''')
code('''science_metrics=json.loads((ROOT/'reports/reliability/scifact/holdout_metrics.json').read_text())
display({k:science_metrics[k] for k in ['count','accuracy','macro_f1_three_gold_classes','coverage']})
# Փոխիր պնդումը քո գիտական օրինակով՝ անգլերեն։
SCIENCE_CLAIM='1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants.'
response=client.post('/verify-science',json={'text':SCIENCE_CLAIM})
response.raise_for_status()
scientific=response.json()
print('Գիտական գնահատական՝',scientific['status'])
for evidence in scientific['claims'][0]['evidence'][:1]:
    print(evidence['source']['title'])
    print(evidence['text'])''')
md('''## 7. Արդյունքի սահմանները և ցուցադրությունը
- Փուլ 1՝ FakeNewsNet 85.71% accuracy, LIAR 26.31%։ Փուլ 2՝ FakeNewsNet 87.43% (տեքստ + աղբյուրի տիրույթ, macro-F1 81.77%) կամ 85.80% միայն տեքստով, LIAR 27.97% մետատվյալներով։ Տարբեր խնդիրներ են, դրանց թվերը չենք միավորում։
- Ընդհանուր աղբյուրներով շղթայի 75%-ը վերաբերում է ընդամենը 24 ձեռքով կազմված պնդումների։
- 60 ծրագրային ստուգումների անցնելը ծրագրի աշխատանքի ստուգում է, ոչ 100% կանխատեսման ճշտություն։
- Տվյալները անգլերեն են։ Հայերեն մոդել այս տարբերակում չկա։
- Թեստային արդյունքները արդեն դիտված են, ուստի հաջորդ բարելավումների համար անհրաժեշտ է նոր անկախ գնահատում։

**Պատրաստի միջերեսը բացելու համար** գործարկիր `start.command`, հետո բացիր http://127.0.0.1:8000 ։ Notebook-ը կարող է աշխատել նաև առանց այդ կայքը միացնելու։

## 8. Կոդի քարտեզ և ամբողջական վերաուսուցում
Մաքրում՝ `fakenews/data.py`, սկզբնական ուսուցում՝ `train.py`, ընդլայնված համեմատություն՝ `optimize.py`, փուլ 2՝ `improve_v2.py` և `stacking.py`, մոդելների կառուցվածք՝ `advanced_models.py`, փաստաստուգում՝ `verify.py`, աղբյուրներ՝ `sources.py`, գիտական ուսուցում՝ `train_science_nli.py`, API՝ `app.py`։

Ամբողջական վերաուսուցումը երկար է և կարող է վերագրել արդյունքները։ Այն գործարկիր միայն նախագծի առանձին պատճենում՝ README.md-ի վերարտադրման հրամաններով։ Այս նոթատետրի սովորական Run All-ը բավարար է գործնական մեթոդը և աշխատող վերջնական մոդելները ցուցադրելու համար։''')
book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Fake News (project)','language':'python','name':'fake-news'},'language_info':{'name':'python'}})
nb.write(book,root/'Final_Practical.ipynb')
