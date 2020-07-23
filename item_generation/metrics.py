from item_generation.utils import preprocess_generated_list, preprocess_generated_items, F_score_item
import numpy as np
from torch import Tensor
import edlib
import re


def overfit_count(list_decoded_outputs, train_data, list_training_items, library, list_library_items):
    """
    Todo
    :param list_decoded_outputs:
    :param train_data:
    :param data_base:
    :return:
    """

    dict_generated_items = preprocess_generated_items(list_decoded_outputs)

    current_num = 0

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

    len_list_decoded_outputs = len(list_decoded_outputs)

    # unique items
    list_0 = []
    # training items
    list_1 = []

    if library is not None:

        # library items
        list_2 = []

        # the number of generated items which are present in the library
        num_library_items = 0.0
        # F beta score for generated items which are present in the library
        num_classification_library_F_score = 0.0

        for item in dict_generated_items["items"]:

            # if the item is in training data
            if item in list_training_items:

                database_item = train_data.find_one({"augmented_item": item})

                list_1.append(list_decoded_outputs[current_num])

                # list_overfit_items.append(1)
                num_overfit_items = num_overfit_items + 1.0
                # if the whole sentence is in training data
                sentence_wo_eos_bos = re.sub('\<\|startoftext\|\>', '', database_item["training_data"])
                sentence_wo_eos_bos = re.sub('\<\|endoftext\|\>', '', sentence_wo_eos_bos)
                if sentence_wo_eos_bos == list_decoded_outputs[current_num]:
                    # list_overfit_sentence.append(1)
                    num_overfit_sentences = num_overfit_sentences + 1.0
                # list_classification_num_overfit_items.append(1)
                num_classification_num_overfit_items = num_classification_num_overfit_items + 1.0
                # if dict_generated_items["labels"] == dict_db["labels"]
                if set(dict_generated_items["labels"][current_num]) == set(database_item["label"]):
                    # list_classification_num_overfit_items_labels.append(1)
                    num_classification_num_overfit_items_labels = num_classification_num_overfit_items_labels + 1.0
                # else if list for the generated item is a subset
                elif set(dict_generated_items["labels"][current_num]) <= set(database_item["label"]):
                    # list_classification_num_overfit_correct_labels.append(1)
                    num_classification_num_overfit_correct_labels = num_classification_num_overfit_correct_labels + 1.0
                # else
                else:
                    # list_classification_num_overfit_F_score.append(F-score)
                    num_classification_num_overfit_F_score = num_classification_num_overfit_F_score + \
                                                             F_score_item(dict_generated_items["labels"][current_num],
                                                                          database_item["label"])
            # else
            else:
                # list_overfit_items.append(Levenshtein_distance)
                list_Levenshtein_metrics = []
                list_Levenshtein_metrics_sentences = []
                for valid_item in train_data.find():
                    item_metrics = edlib.align(item, valid_item["augmented_item"])
                    normalized_distance = 1 - (item_metrics['editDistance'] / max(len(item), len(valid_item["initial_item"])))
                    list_Levenshtein_metrics.append(normalized_distance)

                    # list_overfit_sentence.append(Levenshtein_distance)
                    item_metrics = edlib.align(list_decoded_outputs[current_num], valid_item["training_data"])
                    normalized_distance = 1 - (item_metrics['editDistance'] / max(len(list_decoded_outputs[current_num]),
                                                                                  len(valid_item["training_data"])))
                    list_Levenshtein_metrics_sentences.append(normalized_distance)

                if item in list_library_items:
                    library_item = library.find_one({"augmented_item": item})

                    num_library_items = num_library_items + 1.0
                    num_classification_library_F_score = num_classification_library_F_score + \
                                                             F_score_item(dict_generated_items["labels"][current_num],
                                                                          library_item["label"])

                    list_2.append(list_decoded_outputs[current_num])
                else:
                    list_0.append(list_decoded_outputs[current_num])


                num_overfit_items = num_overfit_items + max(list_Levenshtein_metrics)
                num_overfit_sentences = num_overfit_sentences + max(list_Levenshtein_metrics_sentences)

            current_num = current_num + 1

        overfit_repeated_items = len(dict_generated_items["items"]) - len(set(dict_generated_items["items"]))
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
                   "classified_itmes": [list_0, list_1, list_2]}
    else:
        for item in dict_generated_items["items"]:

            # if the item is in training data
            if item in list_training_items:

                database_item = train_data.find_one({"augmented_item": item})

                list_1.append(list_decoded_outputs[current_num])

                # list_overfit_items.append(1)
                num_overfit_items = num_overfit_items + 1.0
                # if the whole sentence is in training data
                sentence_wo_eos_bos = re.sub('\<\|startoftext\|\>', '', database_item["training_data"])
                sentence_wo_eos_bos = re.sub('\<\|endoftext\|\>', '', sentence_wo_eos_bos)
                if sentence_wo_eos_bos == list_decoded_outputs[current_num]:
                    # list_overfit_sentence.append(1)
                    num_overfit_sentences = num_overfit_sentences + 1.0
                # list_classification_num_overfit_items.append(1)
                num_classification_num_overfit_items = num_classification_num_overfit_items + 1.0
                # if dict_generated_items["labels"] == dict_db["labels"]
                if set(dict_generated_items["labels"][current_num]) == set(database_item["label"]):
                    # list_classification_num_overfit_items_labels.append(1)
                    num_classification_num_overfit_items_labels = num_classification_num_overfit_items_labels + 1.0
                # else if list for the generated item is a subset
                elif set(dict_generated_items["labels"][current_num]) <= set(database_item["label"]):
                    # list_classification_num_overfit_correct_labels.append(1)
                    num_classification_num_overfit_correct_labels = num_classification_num_overfit_correct_labels + 1.0
                # else
                else:
                    # list_classification_num_overfit_F_score.append(F-score)
                    num_classification_num_overfit_F_score = num_classification_num_overfit_F_score + \
                                                             F_score_item(dict_generated_items["labels"][current_num],
                                                                          database_item["label"])
            # else
            else:

                list_0.append(list_decoded_outputs[current_num])

                # list_overfit_items.append(Levenshtein_distance)
                list_Levenshtein_metrics = []
                list_Levenshtein_metrics_sentences = []
                for valid_item in train_data.find():
                    item_metrics = edlib.align(item, valid_item["augmented_item"])
                    normalized_distance = 1 - (item_metrics['editDistance'] / max(len(item), len(valid_item["augmented_item"])))
                    list_Levenshtein_metrics.append(normalized_distance)

                    # list_overfit_sentence.append(Levenshtein_distance)
                    item_metrics = edlib.align(list_decoded_outputs[current_num], valid_item["training_data"])
                    normalized_distance = 1 - (
                                item_metrics['editDistance'] / max(len(list_decoded_outputs[current_num]),
                                                                   len(valid_item["training_data"])))
                    list_Levenshtein_metrics_sentences.append(normalized_distance)

                num_overfit_items = num_overfit_items + max(list_Levenshtein_metrics)
                num_overfit_sentences = num_overfit_sentences + max(list_Levenshtein_metrics_sentences)

            current_num = current_num + 1

        overfit_repeated_items = len(dict_generated_items["items"]) - len(set(dict_generated_items["items"]))
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
