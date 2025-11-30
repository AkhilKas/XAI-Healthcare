import os
import time
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut, ParameterGrid

from joblib import Parallel, delayed, parallel_backend
import matplotlib.pyplot as plt

from .variables import *
from .apply_feature_filter import *
from .helper_functions import *
from .param_grid import *


#######################################################################################################
class RNNClassifier(nn.Module):
    def __init__(self, input_dim, rnn_type="gru", hidden_size=128, num_layers=2,
                 bidirectional=True, dropout_rnn=0.2, dropout_fc=0.3,
                 num_classes=2, pooling="last"):
        super().__init__()
        assert rnn_type in {"gru", "lstm"}, "rnn_type must be 'gru' or 'lstm'"
        self.pooling = pooling.lower()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1

        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=input_dim,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True,
                           bidirectional=bidirectional,
                           dropout=dropout_rnn if num_layers > 1 else 0.0)
        effective_dim = hidden_size * self.num_directions
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_fc),
            nn.Linear(effective_dim, num_classes),
        )

    def forward(self, x, lengths=None):
        if lengths is not None:
            lengths, perm_idx = lengths.sort(descending=True)
            x = x[perm_idx]
            packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True)
            out_packed, _ = self.rnn(packed)
            out, _ = torch.nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
            _, unperm_idx = perm_idx.sort()
            out = out[unperm_idx]
            lengths = lengths[unperm_idx]
        else:
            out, _ = self.rnn(x)

        # Pooling
        if self.pooling == "last":
            if lengths is None:
                feat = out[:, -1]
            else:
                idx = (lengths - 1).clamp(min=0)
                feat = out[torch.arange(out.size(0)), idx]
        elif self.pooling == "mean":
            if lengths is None:
                feat = out.mean(dim=1)
            else:
                mask = torch.arange(out.size(1), device=out.device)[None, :] < lengths[:, None]
                feat = (out * mask.unsqueeze(-1)).sum(1) / lengths.unsqueeze(-1)
        else:  # max
            if lengths is not None:
                mask = torch.arange(out.size(1), device=out.device)[None, :] < lengths[:, None]
                out[~mask] = float("-inf")
            feat = out.max(dim=1).values

        return self.classifier(feat)


#######################################################################################################

def _rnn_loo_score(cfg, X_scaled, y, loo):
    start = time.time()
    pid = log_process_info("Starting_rnn_loo_score", cfg)
    y_true, y_pred, y_proba = [], [], []

    g = torch.Generator().manual_seed(42)

    training_loss_per_epoch = []
    training_acc_per_epoch = []

    for train_idx, test_idx in loo.split(range(len(X_scaled))):
        # Prepare train and test
        X_tr_list = [X_scaled[i] for i in train_idx]
        y_tr_list = [int(y[i]) for i in train_idx]

        X_test_list = [X_scaled[i] for i in test_idx]
        y_test_val = int(y[test_idx][0])

        # DataLoader for training
        train_dataset = list(zip(X_tr_list, y_tr_list))
        loader = DataLoader(train_dataset, batch_size=cfg["batch_size"],
                            shuffle=True, generator=g, collate_fn=collate_fn,
                            pin_memory=(DEVICE.type == "cuda:0"))

        # Model
        model = RNNClassifier(
            input_dim=DOFS,
            rnn_type=cfg["rnn_type"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            bidirectional=cfg["bidirectional"],
            dropout_rnn=cfg["dropout_rnn"],
            dropout_fc=cfg["dropout_fc"],
            pooling=cfg["pooling"],
        ).to(DEVICE)

        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        crit = nn.CrossEntropyLoss()
        model.train()

        epoch_losses = []
        epoch_accs = []
        for _ in range(cfg["epochs"]):
            batch_losses = []
            correct = 0
            total = 0

            for xb, lengths, yb in loader:
                xb, lengths, yb = xb.to(DEVICE), lengths.to(DEVICE), yb.to(DEVICE)
                logits = model(xb, lengths)
                loss = crit(logits, yb)
                
                opt.zero_grad(); loss.backward(); opt.step()
                batch_losses.append(loss.item())

                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += len(yb)

            if len(batch_losses) > 0:
                epoch_losses.append(np.mean(batch_losses))  # mean loss per epoch
                epoch_accs.append(correct / total)
            else:
                epoch_losses.append(float("nan"))
                epoch_accs.append(float("nan"))

        training_loss_per_epoch.append(epoch_losses)
        training_acc_per_epoch.append(epoch_accs)

        # Inference on single test sequence
        model.eval()
        with torch.no_grad():
            x_test = X_test_list[0].unsqueeze(0).to(DEVICE)
            length_test = torch.tensor([len(X_test_list[0])], device=DEVICE)
            logit = model(x_test, length_test)[0]
            prob = torch.softmax(logit, dim=0)[1].item()
            pred = int(logit.argmax().item())

        y_pred.append(pred)
        y_proba.append(prob)
        y_true.append(y_test_val)

    ba = balanced_accuracy_score(y_true, y_pred)
    metrics = print_metrics(f"RNN-- {cfg} combinations", y_true, y_pred, y_proba)
    end = time.time()
    print(f"[PID {pid}] Finished _rnn_loo_score with params {cfg} in {end - start:.2f} seconds")
    
    # Average training loss across LOO folds
    try:
        avg_training_loss = np.nanmean(np.array(training_loss_per_epoch, dtype=float), axis=0)
    except Exception:
        avg_training_loss = np.mean([np.array(e, dtype=float).mean() for e in training_loss_per_epoch])

    try:
        avg_training_acc = np.nanmean(np.array(training_acc_per_epoch, dtype=float), axis=0)
    except Exception:
        avg_training_acc = np.mean([np.array(a, dtype=float).mean() for a in training_acc_per_epoch])

    return ba, cfg, np.array(y_true), np.array(y_pred), np.array(y_proba), model, avg_training_loss, avg_training_acc


#######################################################################################################
def run_rnn(g1, g0):
    X_scaled, y = apply_feature_filter(g1, g0)
    X_scaled = [torch.tensor(seq, dtype=torch.float32) for seq in X_scaled]
    y = torch.tensor(y, dtype=torch.long)
    loo = LeaveOneOut()
    grid = list(ParameterGrid(rnn_param_grid))

    with parallel_backend("loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=1, verbose=10)(
            delayed(_rnn_loo_score)(params, X_scaled, y, loo) for params in grid
        )

    best_score, best_params, y_true, y_pred, y_proba, model, avg_training_loss, avg_training_acc = max(results, key=lambda t: t[0])

    os.makedirs("trained_models", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'params': best_params,
    }, MODEL_SAVE_PATH)
    print(f"Model saved at: {MODEL_SAVE_PATH}")

    # Plot training vs test accuracy curve
    plt.figure(figsize=(8,5))
    plt.plot(avg_training_acc, label="Training Accuracy")
    plt.hlines(best_score, 0, len(avg_training_acc)-1,
            linestyles='--', label="LOO Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs LOO Test Accuracy Curve")
    plt.legend()
    plt.show()

    importance = rnn_channel_importance_from_weights(model, kind=best_params['rnn_type']).detach().cpu().numpy()
    top6_idx = np.argsort(importance)[-1:-7:-1]
    feature_imp = {chan_name[idx]: importance[idx] for idx in top6_idx}
    print("Top-6 sensor channels by permutation importance:")
    for k, v in feature_imp.items():
        print(f"{k}: {v:.4f}")

    return metrics, {'best_params': best_params, 'feature_importance': feature_imp}


#######################################################################################################
if __name__ == "__main__":
    try:
        with open(f"pickled_datasets/patient_data_task{TASK}.pkl", "rb") as f:
            patient_data = pickle.load(f)
        with open(f"pickled_datasets/control_data_task{TASK}.pkl", "rb") as f:
            control_data = pickle.load(f)

        g1, g0 = (patient_data, control_data)
        start_total = time.time()
        metrics, best_params = run_rnn(g1, g0)
        end_total = time.time()
        print(f"Total runtime of run_rnn: {end_total - start_total:.2f} seconds")
        print("Best params & feature importance:", best_params)

    except Exception as e:
        print(f"Error during execution: {str(e)}")
        raise
