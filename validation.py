from sklearn.metrics import f1_score
import numpy as np
import torch
from tqdm import tqdm


def eval_metrics(outputs, labels):
    return {
        'f1': f1_score(y_true=labels, y_pred=(outputs>0.5).astype(int), average='macro')
    }


def mean_metrics(metrics_list):
    keys = metrics_list[0].keys()
    metrics = {k: [] for k in keys}
    for k in keys:
        for m in metrics_list:
            metrics[k].append(m[k])
    for k in keys:
        metrics[k] = np.mean(metrics[k])
    return metrics


def validation(model, val_loader):
    model.eval()
    metrics = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(val_loader)):
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs).view(-1)
            metrics.append(eval_metrics(outputs.cpu().numpy(), labels.cpu().numpy()))
        metrics_mean = mean_metrics(metrics)
    return metrics_mean
