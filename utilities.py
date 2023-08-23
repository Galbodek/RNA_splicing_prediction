import gc
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix, precision_recall_curve, roc_curve


def compute_metrics(y, y_prob):
    _, y_pred = torch.max(y_prob, 1)
    acc = balanced_accuracy_score(y, y_pred)
    roc = roc_auc_score(y, y_prob)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    return {
        'accuracy': acc,
        'auroc': roc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()


def get_loss(model, x, y, device):
    logits = model(x).to(device)
    loss = F.cross_entropy(logits, y)

    y = y.view((y.shape[0] * y.shape[1], y.shape[2])).cpu().detach()
    logits = logits.view((logits.shape[0] * logits.shape[1], logits.shape[2])).cpu().detach()
    metrics = compute_metrics(y, logits)
    return loss, metrics