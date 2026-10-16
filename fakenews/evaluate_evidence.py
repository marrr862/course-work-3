"""Evaluate retrieval separately from factual classification and rewriting."""
import json
from pathlib import Path
import pandas as pd
from .archive import Archive, exact_key
from .evidence import EvidenceLookup

ROOT = Path(__file__).resolve().parents[1]


def main():
    archive = Archive()
    engine = EvidenceLookup(archive=archive)
    queries = json.loads((ROOT / 'data/retrieval_queries.json').read_text())
    results = []
    for item in queries:
        found = archive.search(item['query'])
        ids = [r['id'] for r in found['candidates']]
        rank = ids.index(item['id']) + 1 if item['id'] in ids else None
        assert not found['match'], 'Evaluation query must not duplicate the indexed wording'
        results.append({**item, 'rank_at_5': rank, 'retrieved_ids': ids})
    frame = pd.read_csv(ROOT / 'reports/liar/split_manifest.csv', keep_default_na=False)
    heldout = frame[frame.split == 'test']
    exact_overlap = [text for text in heldout.text if exact_key(text) in archive.exact]
    # Explicitly test changed numerical claims and added negation. No generated verdict is allowed.
    import re
    mutations = []
    for record in archive.data['records']:
        if re.search(r'\d+', record['claim']):
            changed = re.sub(r'\d+', '999999', record['claim'], count=1)
            mutations.append(changed)
        if len(mutations) == 30:
            break
    mutations += ['It is not true that ' + r['claim'] for r in archive.data['records'][:30]]
    mutations = [q for q in mutations if exact_key(q) not in archive.exact]
    inappropriate = sum(engine.search(q)['correction'] is not None for q in mutations)
    sample = heldout.sample(n=min(50, len(heldout)), random_state=42)
    unknown_corrections = sum(engine.search(q)['correction'] is not None for q in sample.text)
    metrics = {'indexed_records': len(archive.data['records']), 'paraphrase_queries': len(queries),
               'recall_at_1': sum(r['rank_at_5'] == 1 for r in results)/len(results),
               'recall_at_5': sum(r['rank_at_5'] is not None for r in results)/len(results),
               'mrr_at_5': sum(1/r['rank_at_5'] if r['rank_at_5'] else 0 for r in results)/len(results),
               'heldout_test_claims_checked_for_index_overlap': len(heldout), 'heldout_exact_overlap': len(exact_overlap),
               'changed_number_or_negation_cases': len(mutations), 'inappropriate_automatic_corrections': inappropriate,
               'heldout_end_to_end_cases': len(sample), 'heldout_automatic_corrections': unknown_corrections,
               'limitations': '20 manually authored paraphrases of archived training claims, fixed before this evaluation; small development challenge, not an independent real-world fact-checking benchmark. Refusal checks test policy, not semantic truth. No automatic paraphrase correction.'}
    for filename, data in [('evidence_evaluation.json', metrics), ('retrieval_predictions.json', results)]:
        (ROOT / 'reports' / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
