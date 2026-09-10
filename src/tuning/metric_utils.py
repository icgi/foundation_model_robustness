from pathlib import Path

from sklearn.metrics import roc_auc_score, confusion_matrix, balanced_accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np


def calculate_auc(y_score, y_true, **kwargs):
    return roc_auc_score(y_true, y_score, **kwargs)


def calculate_bac(preds, labels):
    # preds = (np.array(preds) > 0.5).astype(np.uint8)
    # preds = np.array(preds).argmax()
    return balanced_accuracy_score(labels, preds)


# NOTE: Both functions below assumes that 1 is the positive class
def calculate_sensitivity(preds, labels):
    preds = (np.array(preds) > 0.5).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return tp / (tp + fn)


def calculate_specificity(preds, labels):
    preds = (np.array(preds) > 0.5).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return tn / (tn + fp)


def save_confusion_matrix_image(pred_scores, labels, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.close()
    pred_labels = (np.array(pred_scores) > 0.5).astype(np.uint8)
    cm = confusion_matrix(labels, pred_labels, labels=[0, 1])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Positive", "Negative"]
    )
    _, ax = plt.subplots()
    disp.plot(ax=ax)
    plt.savefig(path)
    plt.close()
