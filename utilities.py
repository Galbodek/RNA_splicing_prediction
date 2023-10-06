import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix, precision_recall_curve, roc_curve


def collate_batch(batch, tokenizer):
    seqs, labels = [], []
    for _seq, _label in batch:
        seqs.append(_seq)
        labels.append(_label)

    # now sort text before padding
    lengths = [len(s) for s in seqs]
    sorted_inds = list(np.argsort(lengths)[::-1])
    clear_cache()
    # sort in descending order for batching, padding to the left - reverse the list and create tensors, pad and flip
    seqs = pad_sequence([seqs[i].flip(dims=[0]) for i in sorted_inds], batch_first=True, padding_value=tokenizer.pad_token_id).flip(dims=[1])

    # sort in descending order for batching, padding to the left - reverse the list and create tensors, pad and flip
    # labels = pad_sequence([labels[i].flip(dims=[0]) for i in sorted_inds], batch_first=True).flip(dims=[1])
    labels = torch.LongTensor(labels)
    return seqs, labels


def compute_metrics(y, logits):
    y_prob = torch.softmax(logits, dim=2)
    _, y_pred = torch.max(y_prob, 1)
    if len(torch.unique(y)) == 1:
        roc, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
    else:
        roc = roc_auc_score(y, y_prob[:, 1])
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, zero_division=1.0)
        precision, recall, f1 = precision[1], recall[1], f1[1]

    acc = balanced_accuracy_score(y, y_pred)
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


def get_loss(model, x, y,  weights, device):
    clear_cache()
    x, y = x.to(device), y.to(device)
    logits = model(x).to(device)
    # y = y.view((y.shape[0] * y.shape[1]))
    # logits = logits.view((logits.shape[0] * logits.shape[1], logits.shape[2]))
    loss = F.cross_entropy(logits, y,  weight=weights)
    metrics = compute_metrics(y.cpu().detach(), logits.cpu().detach())
    return loss, metrics