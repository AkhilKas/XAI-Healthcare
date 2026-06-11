import numpy as np

from backend.helper_functions import (
    auc_ci_bootstrap,
    kde_vals,
    lowpass_filter,
    rnn_channel_importance_from_weights,
)


def test_lowpass_filter_removes_high_frequency_noise():
    fs = 60
    t = np.linspace(0, 2, 2 * fs, endpoint=False)
    clean = np.sin(2 * np.pi * 1.0 * t)
    noise = 0.5 * np.sin(2 * np.pi * 20.0 * t)
    signal = (clean + noise).reshape(-1, 1)

    filtered = lowpass_filter(signal, cutoff=6, fs=fs, order=2)

    assert filtered.shape == signal.shape
    # Filtered signal should be closer to clean than noisy original
    err_before = np.mean((signal[:, 0] - clean) ** 2)
    err_after = np.mean((filtered[:, 0] - clean) ** 2)
    assert err_after < err_before


def test_lowpass_filter_multichannel():
    data = np.random.randn(120, 18)
    filtered = lowpass_filter(data)
    assert filtered.shape == data.shape


def test_auc_ci_bootstrap_basic():
    # Larger sample reduces probability of a bootstrap fold containing only one class
    # (which would make roc_auc_score return NaN).
    rng = np.random.RandomState(0)
    n = 40
    y_true = np.concatenate([np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)])
    y_proba = np.concatenate([
        rng.uniform(0.0, 0.5, n // 2),
        rng.uniform(0.5, 1.0, n // 2),
    ])
    mean_auc, (lo, hi) = auc_ci_bootstrap(y_true, y_proba, n_boot=100, seed=0)
    assert 0.0 <= lo <= mean_auc <= hi <= 1.0
    assert mean_auc > 0.5  # signal is well-separated


def test_kde_vals_returns_density():
    samples = np.random.RandomState(0).normal(0.5, 0.1, size=50)
    x = np.linspace(0, 1, 30)
    y = kde_vals(samples, x)
    assert y is not None
    assert y.shape == x.shape
    assert np.all(y >= 0)


def test_kde_vals_small_sample_returns_none():
    samples = np.array([0.5])
    x = np.linspace(0, 1, 30)
    assert kde_vals(samples, x) is None


def test_rnn_channel_importance_normalized():
    import torch.nn as nn

    rnn = nn.GRU(input_size=18, hidden_size=8, num_layers=1, bidirectional=True, batch_first=True)
    imp = rnn_channel_importance_from_weights(rnn, kind="gru")
    assert imp.shape[0] == 18
    assert abs(imp.sum().item() - 1.0) < 1e-5
    assert (imp >= 0).all().item()
