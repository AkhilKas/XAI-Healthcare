import torch

from backend.rnn_train import RNNClassifier


def test_rnn_classifier_forward_shape_gru():
    model = RNNClassifier(input_dim=18, rnn_type="gru", hidden_size=32, num_layers=1)
    x = torch.randn(4, 50, 18)
    out = model(x)
    assert out.shape == (4, 2)


def test_rnn_classifier_forward_shape_lstm():
    model = RNNClassifier(input_dim=18, rnn_type="lstm", hidden_size=32, num_layers=1)
    x = torch.randn(2, 30, 18)
    out = model(x)
    assert out.shape == (2, 2)


def test_rnn_classifier_supports_mean_pooling():
    model = RNNClassifier(input_dim=18, hidden_size=16, num_layers=1, pooling="mean")
    out = model(torch.randn(3, 20, 18))
    assert out.shape == (3, 2)


def test_rnn_classifier_supports_max_pooling():
    model = RNNClassifier(input_dim=18, hidden_size=16, num_layers=1, pooling="max")
    out = model(torch.randn(3, 20, 18))
    assert out.shape == (3, 2)


def test_rnn_classifier_handles_packed_lengths():
    model = RNNClassifier(input_dim=18, hidden_size=16, num_layers=1, pooling="last")
    x = torch.randn(3, 40, 18)
    lengths = torch.tensor([40, 25, 10])
    out = model(x, lengths)
    assert out.shape == (3, 2)


def test_rnn_classifier_rejects_invalid_rnn_type():
    import pytest

    with pytest.raises(AssertionError):
        RNNClassifier(input_dim=18, rnn_type="transformer")
