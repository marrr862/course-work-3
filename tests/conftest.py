"""Skip tests that need the large neural models when they are not installed.

The repository ships only the small joblib models (see README). Tests that load
torch, the fine-tuned encoders, the NLI model or the LIAR-PLUS archive index are
marked `heavy` and are skipped automatically on a fresh clone instead of failing.
Run them after rebuilding the models with the commands in README.md.
"""
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]

# browser_smoke.py is a standalone script that needs a running server.
collect_ignore = ['browser_smoke.py']

REQUIRED_PATHS = [
    'models/sentence_minilm',
    'models/nli_deberta',
    'models/evidence_archive.joblib',
    'models/science_index.joblib',
]

HEAVY_TESTS = {
    'test_advanced_models.py': {
        'test_embeddings_invariant_to_padding_and_batch_neighbors',
        'test_active_model_metrics_match_saved_predictions',
    },
    'test_reliability.py': {
        'test_high_recall_mode_changes_only_threshold',
        'test_scientific_api_matches_evaluation_and_exposes_source',
    },
    'test_system.py': {
        'test_sourced_correction',
        'test_archive_is_train_only',
        'test_archived_explanation_has_provenance',
        'test_paraphrase_returns_candidate_not_verdict',
        'test_changed_context_never_inherits_correction',
    },
    'test_verification.py': {
        'test_live_model_supports_new_wording',
        'test_partial_evidence_does_not_rewrite_entire_article',
        'test_multi_claim_correction_preserves_supported_statement',
    },
}


def missing_heavy_requirements():
    missing = [name for name in ('torch', 'transformers') if importlib.util.find_spec(name) is None]
    missing += [path for path in REQUIRED_PATHS if not (ROOT / path).exists()]
    return missing


def pytest_configure(config):
    config.addinivalue_line('markers', 'heavy: needs torch and the large models that are not stored in git')


def pytest_collection_modifyitems(config, items):
    missing = missing_heavy_requirements()
    skip = pytest.mark.skip(reason='large models not installed: ' + ', '.join(missing)) if missing else None
    for item in items:
        if item.originalname in HEAVY_TESTS.get(item.path.name, ()):
            item.add_marker(pytest.mark.heavy)
            if skip:
                item.add_marker(skip)
