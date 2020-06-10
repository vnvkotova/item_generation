from item_generation.utils import input_file_to_list, preprocess_generated_list
import numpy as np
from torch import Tensor


def db_match(list_generated, list_db):
    """
    a function to calculate how many newly generated strings match the ones from the training file
    """

    list_generated = preprocess_generated_list(list_generated)

    overlap_count = len(list(set(list_generated) & set(list_db)))

    temp = set(list_db)
    list_overlap = [value for value in list_generated if value in temp]

    overlap_ratio = overlap_count/len(list_generated)

    return overlap_ratio, list_overlap


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def accuracy_thresh(y_pred:Tensor, y_true:Tensor, thresh:float=0.5, sigmoid:bool=True):
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    if sigmoid: y_pred = y_pred.sigmoid()
#     return ((y_pred>thresh)==y_true.byte()).float().mean().item()
    return np.mean(((y_pred>thresh)==y_true.byte()).float().cpu().numpy(), axis=1).sum()


def fbeta(y_pred: Tensor, y_true: Tensor, thresh: float = 0.3, beta: float = 2, eps: float = 1e-9, sigmoid: bool = True):
    "Computes the f_beta between `preds` and `targets`"
    beta2 = beta ** 2
    if sigmoid:
        y_pred = y_pred.sigmoid()
    y_pred = (y_pred > thresh).float()
    print(y_pred.shape)
    y_true = y_true.float()
    print(y_true.shape)
    TP = (y_pred * y_true).sum(dim=1)
    prec = TP / (y_pred.sum(dim=1) + eps)
    print("Precision = " + str(prec))
    rec = TP / (y_true.sum(dim=1) + eps)
    print("Recall = " + str(rec))
    res = (prec * rec) / (prec * beta2 + rec + eps) * (1 + beta2)
    return res.mean().item()
