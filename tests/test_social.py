import numpy as np
import pandas as pd
from fakenews.social import EPOCH_MS, propagation_features, tweet_times, load_frame


def snowflake(ms):
    return (ms - EPOCH_MS) << 22


def test_snowflake_times_are_decoded_in_hours():
    t0 = 1_500_000_000_000
    ids = [snowflake(t0 + 2 * 3_600_000), snowflake(t0), snowflake(t0 + 30 * 3_600_000)]
    assert np.allclose(tweet_times(ids), [0, 2, 30])


def test_features_use_relative_time_only_and_respect_horizon():
    ids = ' '.join(str(snowflake(1_500_000_000_000 + h * 3_600_000)) for h in [0, 1, 5, 48])
    shifted = ' '.join(str(snowflake(1_600_000_000_000 + h * 3_600_000)) for h in [0, 1, 5, 48])
    full, moved = propagation_features(ids), propagation_features(shifted)
    assert full == moved  # absolute date does not change the features
    early = propagation_features(ids, horizon=24)
    assert np.isclose(early['log_tweets'], np.log1p(3)) and np.isclose(full['log_tweets'], np.log1p(4))
    assert np.isclose(full['share_first_24h'], .75)


def test_missing_and_pre_snowflake_ids():
    assert propagation_features(np.nan)['has_tweets'] == 0
    f = propagation_features('12345 67890')
    assert f['has_tweets'] == 1 and f['untimed_share'] == 1 and f['log_span_h'] == 0


def test_frame_keeps_the_existing_split():
    df = load_frame()
    man = pd.read_csv('reports/fakenewsnet/split_manifest.csv')
    assert (df.id.values == man.id.values).all() and (df.split.values == man.split.values).all()
