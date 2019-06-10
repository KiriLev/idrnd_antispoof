from sklearn.metrics import f1_score
import numpy as np


def eval_metrics(outputs, labels):

    return {
        'f1': f1_score(y_true=labels, y_pred=outputs)
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
    for i, (inputs, labels) in enumerate(val_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs).view(-1)
        metrics.append(eval_metrics(outputs.cpu(), labels.cpu()))
    metrics_mean = mean_metrics(metrics)
    return metrics_mean
