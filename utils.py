import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, matthews_corrcoef, roc_auc_score


class AsymmetricLoss(nn.Module):
    """Asymmetric loss for extreme binary imbalance.

    gamma_neg suppresses easy negatives much more aggressively than positives,
    which is useful for tolC-style screening tasks with roughly 2.5% positives.
    """

    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, reduction="mean"):
        super().__init__()
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.clip = float(clip)
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()
        probs = torch.sigmoid(logits)
        pos_probs = probs
        neg_probs = 1.0 - probs

        if self.clip is not None and self.clip > 0:
            neg_probs = (neg_probs + self.clip).clamp(max=1.0)

        log_pos = torch.log(pos_probs.clamp(min=1e-8))
        log_neg = torch.log(neg_probs.clamp(min=1e-8))
        loss = targets * log_pos + (1.0 - targets) * log_neg

        gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
        probs_t = pos_probs * targets + neg_probs * (1.0 - targets)
        loss = -loss * (1.0 - probs_t).pow(gamma)

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def _nan_metric_dict(invalid_count):
    """统一的“空/全缺失”返回，保证字段齐全。"""
    return {
        "AUC": float("nan"),
        "AUPRC": float("nan"),
        "AUPRC_LIFT": float("nan"),
        "NORM_AUPRC": float("nan"),
        "BEDROC20": float("nan"),
        "BED-ROC20": float("nan"),
        "ACC": float("nan"),
        "F1": float("nan"),
        "MCC": float("nan"),
        "EF@0.5%": float("nan"),
        "EF@1%": float("nan"),
        "EF1": float("nan"),
        "EF5": float("nan"),
        "EF10": float("nan"),
        "Precision@50": float("nan"),
        "Precision@100": float("nan"),
        "Precision@500": float("nan"),
        "Precision@1000": float("nan"),
        "Recall@100": float("nan"),
        "Recall@500": float("nan"),
        "Recall@1000": float("nan"),
        "N": 0,
        "N_POS": 0,
        "N_NEG": 0,
        "POS_RATE": float("nan"),
        "N_INVALID": int(invalid_count),
    }


def _bedroc_from_ranked_labels(ranked_labels, alpha=20.0):
    num_samples = int(len(ranked_labels))
    num_actives = int(np.sum(ranked_labels))
    if num_samples <= 1 or num_actives <= 0 or num_actives >= num_samples:
        return float("nan")

    alpha = float(alpha)
    active_ranks = np.flatnonzero(ranked_labels > 0) + 1  # 1-indexed ranks
    active_fraction = num_actives / num_samples

    observed = np.exp(-alpha * active_ranks / num_samples).sum()
    random_sum = (
        active_fraction
        * (-np.expm1(-alpha))
        / np.expm1(alpha / num_samples)
    )
    if random_sum <= 0:
        return float("nan")

    rie = observed / random_sum
    common = (num_samples / num_actives) * (-np.expm1(-alpha * active_fraction)) / (-np.expm1(-alpha))
    rie_max = common
    rie_min = common * np.exp(-alpha * (1.0 - active_fraction))
    denom = rie_max - rie_min
    if denom <= 0:
        return float("nan")
    return float(np.clip((rie - rie_min) / denom, 0.0, 1.0))


def compute_metrics(logits, labels):
    """二分类评估指标。

    新增字段（第一阶段要求）：
      - N, N_POS, N_NEG, POS_RATE
      - AUPRC_LIFT  (= AUPRC / POS_RATE, 随机基线的 lift)
      - BEDROC20 / BED-ROC20 (alpha=20, early-recognition metric)
      - EF@1%, EF@0.5%
      - Precision@100, Precision@50
    兼容保留旧字段（EF1/EF5/EF10、Precision@500/1000、Recall@K 等）。
    """
    probs = torch.sigmoid(logits).cpu().numpy().flatten()
    labels = labels.cpu().numpy().flatten()
    finite_mask = np.isfinite(probs)
    invalid_count = int((~finite_mask).sum())

    if invalid_count > 0:
        probs = probs[finite_mask]
        labels = labels[finite_mask]

    labels = np.rint(labels).astype(int)

    if len(labels) == 0:
        return _nan_metric_dict(invalid_count)

    unique_labels = np.unique(labels)
    preds = (probs > 0.5).astype(int)
    ranking = np.argsort(-probs, kind="stable")
    ranked_labels = labels[ranking]
    num_samples = len(labels)
    num_positives = int(labels.sum())
    num_negatives = int(num_samples - num_positives)
    pos_rate = float(labels.mean()) if num_samples > 0 else float("nan")

    auc = float("nan")
    auprc = float("nan")
    if len(unique_labels) > 1:
        auc = roc_auc_score(labels, probs)
        auprc = average_precision_score(labels, probs)

    # AUPRC_LIFT = AUPRC / baseline_pos_rate，衡量相对于随机猜测的提升倍数
    if np.isfinite(auprc) and pos_rate and pos_rate > 0:
        auprc_lift = float(auprc) / float(pos_rate)
    else:
        auprc_lift = float("nan")

    # NORM_AUPRC = (AUPRC - baseline) / (1 - baseline)，归一化到 [0,1]
    if np.isfinite(auprc) and 0.0 < pos_rate < 1.0:
        norm_auprc = (float(auprc) - float(pos_rate)) / (1.0 - float(pos_rate))
    else:
        norm_auprc = float("nan")

    def enrichment_factor(top_fraction):
        if num_samples == 0 or num_positives == 0:
            return float("nan")
        top_n = int(np.ceil(num_samples * top_fraction))
        if top_n <= 0:
            return float("nan")
        hits = float(ranked_labels[:top_n].sum())
        expected_hits = num_positives * (top_n / num_samples)
        if expected_hits <= 0:
            return float("nan")
        return hits / expected_hits

    def precision_at_k(k):
        if num_samples == 0:
            return float("nan")
        top_n = min(int(k), num_samples)
        if top_n <= 0:
            return float("nan")
        hits = float(ranked_labels[:top_n].sum())
        return hits / top_n

    def recall_at_k(k):
        if num_positives == 0:
            return float("nan")
        top_n = min(int(k), num_samples)
        if top_n <= 0:
            return float("nan")
        hits = float(ranked_labels[:top_n].sum())
        return hits / num_positives

    ef_1 = float(enrichment_factor(0.01))
    ef_05 = float(enrichment_factor(0.005))
    bedroc20 = _bedroc_from_ranked_labels(ranked_labels, alpha=20.0)

    metrics = {
        "AUC": float(auc),
        "AUPRC": float(auprc),
        "AUPRC_LIFT": float(auprc_lift),
        "NORM_AUPRC": float(norm_auprc),
        "BEDROC20": float(bedroc20),
        "BED-ROC20": float(bedroc20),
        "ACC": float(accuracy_score(labels, preds)),
        "F1": float(f1_score(labels, preds, zero_division=0)),
        "MCC": float(matthews_corrcoef(labels, preds)),
        # 第一阶段显式要求的字段命名
        "EF@0.5%": ef_05,
        "EF@1%": ef_1,
        # 旧字段，保留向后兼容
        "EF1": ef_1,
        "EF5": float(enrichment_factor(0.05)),
        "EF10": float(enrichment_factor(0.10)),
        "Precision@50": float(precision_at_k(50)),
        "Precision@100": float(precision_at_k(100)),
        "Precision@500": float(precision_at_k(500)),
        "Precision@1000": float(precision_at_k(1000)),
        "Recall@100": float(recall_at_k(100)),
        "Recall@500": float(recall_at_k(500)),
        "Recall@1000": float(recall_at_k(1000)),
        "N": int(num_samples),
        "N_POS": int(num_positives),
        "N_NEG": int(num_negatives),
        "POS_RATE": float(pos_rate),
        "N_INVALID": int(invalid_count),
    }
    return metrics
