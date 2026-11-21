"""Stage 7 leakage guards: LIAR2 copies of held-out LIAR claims must never reach training."""
import pandas as pd
from fakenews.liar2_stage7 import drop_near, norm_key


def test_norm_key_ignores_apostrophes_and_quotes():
    # LIAR stored "Hes", LIAR2 stores "He's" with quotes: the same claim.
    assert norm_key("Hes the only candidate whos balanced budgets.") == norm_key('He\'s the only candidate "who\'s balanced budgets."')


def test_drop_near_removes_reworded_copies_only():
    pool = pd.DataFrame({'text': ['Tom Ganley has "two F\'s from the Better Business Bureau" and "over 160 complaints in just three years.',
                                  'The unemployment rate in Ohio fell below five percent last year.']})
    heldout = pd.DataFrame({'text': ['Tom Ganley has two Fs from the Better Business Bureau and over 160 complaints in just three years.']})
    kept, removed = drop_near(pool, heldout)
    assert removed == 1 and list(kept.text) == ['The unemployment rate in Ohio fell below five percent last year.']
