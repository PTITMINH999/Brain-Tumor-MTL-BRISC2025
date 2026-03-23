import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def classification_metrics(preds, labels):

    preds = np.array(preds)
    labels = np.array(labels)

    acc = (preds == labels).mean()

    precision = precision_score(labels, preds, average='weighted', zero_division=0)

    recall = recall_score(labels, preds, average='weighted', zero_division=0)

    f1 = f1_score(labels, preds, average='weighted', zero_division=0)

    cm = confusion_matrix(labels, preds)

    return acc, precision, recall, f1, cm