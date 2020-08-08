from item_generation.utils import preprocess_generated_list, preprocess_generated_items_tuples, F_score_item
import numpy as np
from torch import Tensor
import edlib
import re

from joblib import Parallel, delayed
import multiprocessing

global_list_training_items = []
global_train_data = None
global_list_library_items = []
global_library = None


def overfit_iteration_library(preprocessed_tuple):

    list_rubbish = ["\n", "#0", "#1", "#2", "#3", "#4", "#5", "#6", "#7", "#8", "#9", "#_", "##",
                    "0#", "1#", "2#", "3#", "4#", "5#", "6#", "7#", "8#", "9#"]

    global global_list_training_items
    global global_train_data
    global global_list_library_items
    global global_library

    num_overfit_sentences = 0.0
    num_overfit_items = 0.0
    num_classification_num_overfit_items = 0.0
    num_classification_library_F_score = 0.0
    num_classification_num_overfit_items_labels = 0.0
    num_classification_num_overfit_correct_labels = 0.0
    num_library_items = 0.0
    num_classification_num_overfit_F_score = 0.0

    bool_rubbish = False
    if "@" not in preprocessed_tuple[0][1]:
        bool_rubbish = True
    elif any(str_rubbish in preprocessed_tuple[0][1] for str_rubbish in list_rubbish):
        bool_rubbish = True

    if bool_rubbish:
        tuple_type = (0, preprocessed_tuple[0][1])
    else:
        # if the item is in training data
        if preprocessed_tuple[0][0] in global_list_training_items:

            database_item = global_train_data.find_one({"augmented_item": preprocessed_tuple[0][0]})

            tuple_type = (1, preprocessed_tuple[0][1])

            # list_overfit_items.append(1)
            num_overfit_items = 1.0
            # if the whole sentence is in training data
            sentence_wo_eos_bos = re.sub('\<\|startoftext\|\>', '', database_item["training_data"])
            sentence_wo_eos_bos = re.sub('\<\|endoftext\|\>', '', sentence_wo_eos_bos)
            if sentence_wo_eos_bos == preprocessed_tuple[0][1]:
                # list_overfit_sentence.append(1)
                num_overfit_sentences = 1.0
            # list_classification_num_overfit_items.append(1)
            num_classification_num_overfit_items = 1.0
            # if dict_generated_items["labels"] == dict_db["labels"]
            if set(preprocessed_tuple[1]) == set(database_item["label"]):
                # list_classification_num_overfit_items_labels.append(1)
                num_classification_num_overfit_items_labels = 1.0
            # else if list for the generated item is a subset
            elif set(preprocessed_tuple[1]) <= set(database_item["label"]):
                # list_classification_num_overfit_correct_labels.append(1)
                num_classification_num_overfit_correct_labels = 1.0
            # else
            else:
                # list_classification_num_overfit_F_score.append(F-score)
                num_classification_num_overfit_F_score = F_score_item(preprocessed_tuple[1],
                                                                      database_item["label"])
        # else
        else:
            # list_overfit_items.append(Levenshtein_distance)
            list_Levenshtein_metrics = []
            list_Levenshtein_metrics_sentences = []
            for valid_item in global_train_data.find():
                item_metrics = edlib.align(preprocessed_tuple[0][0], valid_item["augmented_item"])
                normalized_distance = 1 - (item_metrics['editDistance'] / max(len(preprocessed_tuple[0][0]),
                                                                              len(valid_item["initial_item"])))
                list_Levenshtein_metrics.append(normalized_distance)

                # list_overfit_sentence.append(Levenshtein_distance)
                item_metrics = edlib.align(preprocessed_tuple[0][1], valid_item["training_data"])
                normalized_distance = 1 - (item_metrics['editDistance'] / max(len(preprocessed_tuple[0][1]),
                                                                              len(valid_item["training_data"])))
                list_Levenshtein_metrics_sentences.append(normalized_distance)

            if preprocessed_tuple[0][0] in global_list_library_items:
                library_item = global_library.find_one({"augmented_item": preprocessed_tuple[0][0]})

                num_library_items = 1.0
                num_classification_library_F_score = F_score_item(preprocessed_tuple[1],
                                                                  library_item["label"])

                tuple_type = (2, preprocessed_tuple[0][1])
            else:
                tuple_type = (0, preprocessed_tuple[0][1])

            num_overfit_items = max(list_Levenshtein_metrics)
            num_overfit_sentences = max(list_Levenshtein_metrics_sentences)

    return tuple_type, num_overfit_sentences, num_overfit_items, num_classification_num_overfit_items, num_classification_library_F_score, num_classification_num_overfit_items_labels, num_classification_num_overfit_correct_labels, num_library_items, num_classification_num_overfit_F_score


def overfit_iteration(preprocessed_tuple):

    list_rubbish = ["\n", "#0", "#1", "#2", "#3", "#4", "#5", "#6", "#7", "#8", "#9", "#_", "##",
                    "0#", "1#", "2#", "3#", "4#", "5#", "6#", "7#", "8#", "9#"]

    global global_list_training_items
    global global_train_data

    num_overfit_sentences = 0.0
    num_overfit_items = 0.0
    num_classification_num_overfit_items = 0.0
    num_classification_num_overfit_items_labels = 0.0
    num_classification_num_overfit_correct_labels = 0.0
    num_classification_num_overfit_F_score = 0.0

    bool_rubbish = False
    if "@" not in preprocessed_tuple[0][1]:
        bool_rubbish = True
    elif any(str_rubbish in preprocessed_tuple[0][1] for str_rubbish in list_rubbish):
        bool_rubbish = True

    if bool_rubbish:
        tuple_type = (0, preprocessed_tuple[0][1])
    else:
        # if the item is in training data
        if preprocessed_tuple[0][0] in global_list_training_items:

            database_item = global_train_data.find_one({"augmented_item": preprocessed_tuple[0][0]})

            tuple_type = (1, preprocessed_tuple[0][1])

            # list_overfit_items.append(1)
            num_overfit_items = 1.0
            # if the whole sentence is in training data
            sentence_wo_eos_bos = re.sub('\<\|startoftext\|\>', '', database_item["training_data"])
            sentence_wo_eos_bos = re.sub('\<\|endoftext\|\>', '', sentence_wo_eos_bos)
            if sentence_wo_eos_bos == preprocessed_tuple[0][1]:
                # list_overfit_sentence.append(1)
                num_overfit_sentences = 1.0
            # list_classification_num_overfit_items.append(1)
            num_classification_num_overfit_items = 1.0
            # if dict_generated_items["labels"] == dict_db["labels"]
            if set(preprocessed_tuple[1]) == set(database_item["label"]):
                # list_classification_num_overfit_items_labels.append(1)
                num_classification_num_overfit_items_labels = 1.0
            # else if list for the generated item is a subset
            elif set(preprocessed_tuple[1]) <= set(database_item["label"]):
                # list_classification_num_overfit_correct_labels.append(1)
                num_classification_num_overfit_correct_labels = 1.0
            # else
            else:
                # list_classification_num_overfit_F_score.append(F-score)
                num_classification_num_overfit_F_score = F_score_item(preprocessed_tuple[1],
                                                                      database_item["label"])
        # else
        else:

            tuple_type = (0, preprocessed_tuple[0][1])

            # list_overfit_items.append(Levenshtein_distance)
            list_Levenshtein_metrics = []
            list_Levenshtein_metrics_sentences = []
            for valid_item in global_train_data.find():
                item_metrics = edlib.align(preprocessed_tuple[0][0], valid_item["augmented_item"])
                normalized_distance = 1 - (item_metrics['editDistance'] / max(len(preprocessed_tuple[0][0]),
                                                                              len(valid_item["augmented_item"])))
                list_Levenshtein_metrics.append(normalized_distance)

                # list_overfit_sentence.append(Levenshtein_distance)
                item_metrics = edlib.align(preprocessed_tuple[0][1], valid_item["training_data"])
                normalized_distance = 1 - (
                        item_metrics['editDistance'] / max(len(preprocessed_tuple[0][1]),
                                                           len(valid_item["training_data"])))
                list_Levenshtein_metrics_sentences.append(normalized_distance)

            num_overfit_items = max(list_Levenshtein_metrics)
            num_overfit_sentences = max(list_Levenshtein_metrics_sentences)
        return tuple_type, num_overfit_sentences, num_overfit_items, num_classification_num_overfit_items, num_classification_num_overfit_items_labels, num_classification_num_overfit_correct_labels, num_classification_num_overfit_F_score


def overfit_count(list_decoded_outputs, train_data, list_training_items, library, list_library_items):
    """
    Todo
    :param list_decoded_outputs:
    :param train_data:
    :param data_base:
    :return:
    """

    list_rubbish = ["\n", "#0", "#1", "#2", "#3", "#4", "#5", "#6", "#7", "#8", "#9", "#_", "##",
                    "0#", "1#", "2#", "3#", "4#", "5#", "6#", "7#", "8#", "9#"]

    list_preprocessed_tuples, overfit_repeated_items = preprocess_generated_items_tuples(list_decoded_outputs)

    num_cores = multiprocessing.cpu_count()

    # these are metrics to measure the generation overfit
    # evaluate items overfit via Levenshtein distance
    num_overfit_items = 0.0
    # evaluate sentence overfit through Levenshtein distance
    num_overfit_sentences = 0.0

    # these are metrics to measure the classification overfit
    # the number of generated items which are present in the database
    num_classification_num_overfit_items = 0.0
    # the number of generated items which are present in the database with the same labels
    num_classification_num_overfit_items_labels = 0.0
    # the number of generated items which are present in the database and were labeled correctly but maybe not with
    # all correct labels
    num_classification_num_overfit_correct_labels = 0.0
    # F beta score for generated items which are present in the database
    num_classification_num_overfit_F_score = 0.0

    # list_rubbish = ["\n", "#0", "#1", "#2", "#3", "#4", "#5", "#6", "#7", "#8", "#9", "#_", "##",
    #                 "0#", "1#", "2#", "3#", "4#", "5#", "6#", "7#", "8#", "9#"]

    len_list_decoded_outputs = len(list_decoded_outputs)

    # unique items
    list_0 = []
    # training items
    list_1 = []

    global global_list_training_items
    global global_train_data
    global global_list_library_items
    global global_library
    global_list_training_items = list_training_items
    global_train_data = train_data
    global_list_library_items = list_library_items
    global_library = library

    print(global_train_data)

    if library is not None:

        # library items
        list_2 = []

        # the number of generated items which are present in the library
        num_library_items = 0.0
        # F beta score for generated items which are present in the library
        num_classification_library_F_score = 0.0

        output = Parallel(n_jobs=num_cores, require='sharedmem')(delayed(overfit_iteration_library)(preprocessed_tuple) for preprocessed_tuple in list_preprocessed_tuples)

        for item in output:
            if item[0][0] == 0:
                list_0.append(item[0][1])
            elif item[0][0] == 1:
                list_1.append(item[0][1])
            else:
                list_2.append(item[0][1])
            num_overfit_sentences = num_overfit_sentences + item[1]
            num_overfit_items = num_overfit_items + item[2]
            num_classification_num_overfit_items = num_classification_num_overfit_items + item[3]
            num_classification_library_F_score = num_classification_library_F_score + item[4]
            num_classification_num_overfit_items_labels = num_classification_num_overfit_items_labels + item[5]
            num_classification_num_overfit_correct_labels = num_classification_num_overfit_correct_labels + item[6]
            num_library_items = num_library_items + item[7]
            num_classification_num_overfit_F_score = num_classification_num_overfit_F_score + item[8]

        overfit_repeated_sentences = len_list_decoded_outputs - len(set(list_decoded_outputs))

        metrics = {"overfit_items": num_overfit_items/len_list_decoded_outputs,
                   "overfit_sentences": num_overfit_sentences/len_list_decoded_outputs,
                   "overfit_repeated_items": overfit_repeated_items/len_list_decoded_outputs,
                   "overfit_repeated_sentences": overfit_repeated_sentences/len_list_decoded_outputs,
                   "classification_overfit_items": num_classification_num_overfit_items/len_list_decoded_outputs,
                   "classification_overfit_sentences": num_classification_num_overfit_items_labels/len_list_decoded_outputs,
                   "classification_labels": num_classification_num_overfit_correct_labels/len_list_decoded_outputs,
                   "classification_F_score": num_classification_num_overfit_F_score/len_list_decoded_outputs,
                   "library_items": num_library_items/len_list_decoded_outputs,
                   "classification_library_F_score": num_classification_library_F_score/len_list_decoded_outputs,
                   "classified_sentences": [list_0, list_1, list_2]}
    else:
        output = Parallel(n_jobs=num_cores, require='sharedmem')(delayed(overfit_iteration)(preprocessed_tuple) for preprocessed_tuple in list_preprocessed_tuples)

        for item in output:
            if item[0][0] == 0:
                list_0.append(item[0][1])
            else:
                list_1.append(item[0][1])
            num_overfit_sentences = num_overfit_sentences + item[1]
            num_overfit_items = num_overfit_items + item[2]
            num_classification_num_overfit_items = num_classification_num_overfit_items + item[3]
            num_classification_num_overfit_items_labels = num_classification_num_overfit_items_labels + item[4]
            num_classification_num_overfit_correct_labels = num_classification_num_overfit_correct_labels + item[5]
            num_classification_num_overfit_F_score = num_classification_num_overfit_F_score + item[6]

        overfit_repeated_sentences = len_list_decoded_outputs - len(set(list_decoded_outputs))

        metrics = {"overfit_items": num_overfit_items / len_list_decoded_outputs,
                   "overfit_sentences": num_overfit_sentences / len_list_decoded_outputs,
                   "overfit_repeated_items": overfit_repeated_items / len_list_decoded_outputs,
                   "overfit_repeated_sentences": overfit_repeated_sentences / len_list_decoded_outputs,
                   "classification_overfit_items": num_classification_num_overfit_items / len_list_decoded_outputs,
                   "classification_overfit_sentences": num_classification_num_overfit_items_labels / len_list_decoded_outputs,
                   "classification_labels": num_classification_num_overfit_correct_labels / len_list_decoded_outputs,
                   "classification_F_score": num_classification_num_overfit_F_score / len_list_decoded_outputs,
                   "classified_sentences": [list_0, list_1]}

    return metrics


def db_match(list_generated, list_db):
    """
    a function to calculate how many newly generated strings match the ones from the training file
    """

    list_generated = preprocess_generated_list(list_generated)

    set_list_db = set(list_db)
    overlap_count = len(list(set(list_generated) & set_list_db))

    list_overlap = [value for value in list_generated if value in set_list_db]

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
    print(y_pred)
    y_true = y_true.float()
    print(y_true)
    TP = (y_pred * y_true).sum(dim=1)
    prec = TP / (y_pred.sum(dim=1) + eps)
    print("Precision = " + str(prec.mean(0)))
    rec = TP / (y_true.sum(dim=1) + eps)
    print("Recall = " + str(rec.mean(0)))
    res = (prec * rec) / (prec * beta2 + rec + eps) * (1 + beta2)
    return res.mean().item()
