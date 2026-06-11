import numpy as np

from backend.apply_feature_filter import apply_feature_filter


def _make_sample(n_rows, n_extra_channels=18):
    # First column is a timestamp/index column that the filter drops via [:, 1:]
    return np.column_stack([
        np.arange(n_rows, dtype=np.float32),
        np.random.randn(n_rows, n_extra_channels).astype(np.float32),
    ])


def test_apply_feature_filter_labels_and_filters_length():
    g1 = {"p1": _make_sample(100), "p2": _make_sample(200)}
    g0 = {"c1": _make_sample(150)}

    X, y = apply_feature_filter(g1, g0)

    assert len(X) == 3
    assert list(y) == [1, 1, 0]
    # Each sequence keeps only 18 feature channels
    for seq in X:
        assert seq.shape[1] == 18


def test_apply_feature_filter_drops_sequences_longer_than_5000():
    g1 = {"short": _make_sample(100), "long": _make_sample(6000)}
    g0 = {"c1": _make_sample(50)}

    X, y = apply_feature_filter(g1, g0)

    # 6000-length sequence should be dropped
    assert len(X) == 2
    assert sorted(y.tolist()) == [0, 1]


def test_apply_feature_filter_scales_features():
    np.random.seed(0)
    g1 = {f"p{i}": _make_sample(80) for i in range(3)}
    g0 = {f"c{i}": _make_sample(80) for i in range(3)}

    X, _ = apply_feature_filter(g1, g0)

    stacked = np.vstack(X)
    # StandardScaler should yield ~zero mean, ~unit variance across the concatenated set
    assert abs(stacked.mean()) < 0.1
    assert abs(stacked.std() - 1.0) < 0.1
