# 2 X 2 X 2 X 1 X 1 X 1 X 1 X 1 X 1 X 1 X 2 = 16
rnn_param_grid = {
    # architecture
    "rnn_type":      ["gru", "lstm"],    # 2×
    "hidden_size":   [64, 128],          # 2×
    "num_layers":    [1, 2],             # 2×

    # regularisation
    "bidirectional": [True],             # keep fixed (➜ doubles params)
    "dropout_rnn":   [0.2],              # after each RNN layer
    "dropout_fc":    [0.3],              # before final FC

    # optimisation
    "lr":            [1e-3],             # Adam learning-rate
    "epochs":        [30],               # training epochs
    "batch_size":    [8],               # mini-batch size

    # sequence pooling
    "pooling":       ["mean", "max"],   # 2× last
}