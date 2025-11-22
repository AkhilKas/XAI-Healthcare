"""
Select the classification paradigm from main function
Select if use padding or truncating from variables.py
Select one attempt for LU or LL or both from variables.py
"""

import os
from typing import Dict

import numpy as np
import psutil
from scipy.stats import gaussian_kde, norm
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.nn.utils.rnn import pad_sequence

from variables import *


def auc_ci_bootstrap(y_true, y_proba, n_boot=1000, seed=42, ci=0.95):
    rng = np.random.RandomState(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)          # sample with replacement
        aucs.append(roc_auc_score(y_true[idx], y_proba[idx]))
    lower = np.percentile(aucs, (1-ci)*50)
    upper = np.percentile(aucs, 100-(1-ci)*50)
    return np.mean(aucs), (lower, upper)


def kde_vals(samples, x, jitter_scale=1e-3):
    samples = np.asarray(samples, dtype=float).ravel()
    if samples.size < 2:
        return None  # not enough points
    s = samples.std()
    if s < 1e-8:
        # near-constant; draw a narrow normal around the mean as a fallback
        mu = samples.mean()
        # width proportional to range to make it visible; clamp inside [0,1]
        width = max(jitter_scale, 0.02)
        return norm.pdf(x, loc=mu, scale=width)
    try:
        kde = gaussian_kde(samples)
        return kde(x)
    except np.linalg.LinAlgError:
        # add tiny jitter to break singularity, then retry
        eps = max(jitter_scale, 1e-3 * (s if s > 0 else 1.0))
        kde = gaussian_kde(samples + np.random.normal(0, eps, size=samples.shape))
        return kde(x)

def log_process_info(prefix: str, params: Dict):
    """Log process ID, CPU affinity, and start message."""
    pid = os.getpid()
    proc = psutil.Process(pid)
    try:
        affinity = proc.cpu_affinity()
    except AttributeError:
        affinity = "Not supported"
    print(f"[PID {pid}] {prefix} with params {params}, allowed cores: {affinity}")
    return pid


def print_metrics(name, y_true, y_pred, y_proba, paradigm=None, best=False):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)
        
    # Binary classification (0 and 1)
    y_pred = np.clip(y_pred, 0, 1)
    
    # Calculate metrics with explicit labels
    labels = [0, 1]
    ba = balanced_accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, labels=labels, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, zero_division=0)
    precision = precision_score(y_true, y_pred, labels=labels, zero_division=0)

    print(f'\n{name}')
    print('---------------------------------------------')
    
    # Handle classification report with explicit labels
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    print(report)
    
    print('Balanced accuracy :', ba)
    print('F1 score          :', f1)
    print('Recall (sens)     :', recall)
    print('Precision         :', precision)

    # if best:
    #     probs_0 = y_proba[y_true == 0]
    #     probs_1 = y_proba[y_true == 1]
        
    #     fig, ax = plt.subplots(figsize=(6, 4))

    #     ax.hist(probs_0, bins=30, range=(0, 1), density=True, alpha=0.5, label="Class 0 - hist")
    #     ax.hist(probs_1, bins=30, range=(0, 1), density=True, alpha=0.5, label="Class 1 - hist")

    #     # KDE curves
    #     x = np.linspace(0, 1, 300)

    #     y0 = kde_vals(probs_0, x)
    #     y1 = kde_vals(probs_1, x)
    #     if y0 is not None: ax.plot(x, y0, linewidth=2, label="Class 0 - KDE")
    #     if y1 is not None: ax.plot(x, y1, linewidth=2, label="Class 1 - KDE")

    #     ax.set_xlim(0, 1)
    #     ax.set_xlabel("Predicted probability")
    #     ax.set_ylabel("Density")
    #     ax.set_title(f"Probability density by true label - Model: {name} - Task: {TASK} - Paradigm: {paradigm}")
    #     ax.legend()
    #     plt.tight_layout()

    #     fig.savefig(f"Figures/TASK{TASK}/prob_density_by_label_m{name}_p{paradigm}.png", dpi=300, bbox_inches="tight")
    
    mean_auc, (ci_low, ci_high) = auc_ci_bootstrap(y_true, y_proba)
    print(f"AUC = {mean_auc:.2f} [{ci_low:.2f} - {ci_high:.2f}]")
    print("---------------------------------------------")
    return {"ba": round(ba, 3), "recall": round(recall, 3), "auc": [round(mean_auc, 3), round(ci_low, 3), round(ci_high, 3)]}


def collate_fn(batch):
    """Pads sequences in batch and returns lengths."""
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels


def rnn_channel_importance_from_weights(obj, kind="lstm", layer=0):
    rnn = obj.rnn if hasattr(obj, "rnn") else obj
    W_f = getattr(rnn, f"weight_ih_l{layer}").detach().abs()
    W_list = [W_f]
    if getattr(rnn, "bidirectional", False):
        W_r = getattr(rnn, f"weight_ih_l{layer}_reverse").detach().abs()
        W_list.append(W_r)
    W = torch.cat(W_list, dim=0)
    H = rnn.hidden_size
    n_gates = 4 if kind.lower() == "lstm" else 3
    Wg = W.view(-1, H, W.shape[1])
    imp = Wg.sum(dim=(0, 1))
    return imp / (imp.sum() + 1e-12)
