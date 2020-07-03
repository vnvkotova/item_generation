from item_generation.utils import preprocess_generated_list, compare_obj, preprocess_generated_items, F_score_item
import numpy as np
from torch import Tensor
import edlib


def overfit_count(list_decoded_outputs, train_data, data_base):

    # Go through each newly generated item

    # If item is in db, then divergence = 1
        # Check the labels, if the

    # Currently: if items are the same they get 1!

    # Todo: should return a dictionary entry for the current epoch with one single value for every metric

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
    # F beta score for generated items which are present in the databas
    num_classification_num_overfit_F_score = 0.0

    for item in dict_generated_items["items"]:

        # if the item is in training data
        database_item = train_data.find_one({"text": item})
        if database_item is not None:
            # list_overfit_items.append(1)
            num_overfit_items = num_overfit_items + 1.0
            # if the whole sentence is in training data
            training_item = train_data.find_one({"training_data": list_decoded_outputs[current_num]})
            if training_item != None:
                # list_overfit_sentence.append(1)
                num_overfit_sentences = num_overfit_sentences + 1.0
            # list_classification_num_overfit_items.append(1)
            num_classification_num_overfit_items = num_classification_num_overfit_items + 1
            # if dict_generated_items["labels"] == dict_db["labels"]
            # print("Tell me the type of dict_generated_items[\"labels\"] please")
            # print(dict_generated_items["labels"])
            # print("Tell me the type of database_item[\"label\"] please")
            # print(database_item["label"])
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
                item_metrics = edlib.align(item, valid_item["text"])
                normalized_distance = 1 - (item_metrics['editDistance'] / max(len(item), len(valid_item["text"])))
                list_Levenshtein_metrics.append(normalized_distance)

                # list_overfit_sentence.append(Levenshtein_distance)
                item_metrics = edlib.align(list_decoded_outputs[current_num], valid_item["training_data"])
                normalized_distance = 1 - (item_metrics['editDistance'] / max(len(list_decoded_outputs[current_num]),
                                                                              len(valid_item["training_data"])))
                list_Levenshtein_metrics_sentences.append(normalized_distance)

            num_overfit_items = num_overfit_items + max(list_Levenshtein_metrics)
            num_overfit_sentences = num_overfit_sentences + max(list_Levenshtein_metrics_sentences)

        current_num = current_num + 1

    metrics = {"overfit_items": num_overfit_items, "overfit_sentences": num_overfit_sentences,
               "classification_overfit_items": num_classification_num_overfit_items,
               "classification_overfit_sentences": num_classification_num_overfit_items_labels,
               "classification_labels": num_classification_num_overfit_correct_labels,
               "classification_F_score": num_classification_num_overfit_F_score}

    return metrics


def add_overfit_count(parsed_output, parsed_input, dichotomous=True):
    """Summary or Description of the Function
    TODO
    Parameters:
    param (type): MISSING DESC

    Returns:
    list:MISSING DESC

    """

    parsed_output = [parsed_output] if not isinstance(parsed_output, list) else parsed_output

    # Todo: the following is only temporarily needed
    # parsed_output - list of all entries for all epochs?
    # output_item - list of entries as in the log file which Björn shared with me, an element of output_entry['items']
    for output_entry in parsed_output:
        for output_item in output_entry['items']:
            divergent_matches = []
            convergent_matches = []

            for input_entry in parsed_input:
                divergent_match = compare_obj(input_entry['stems'], output_item['stems'], dichotomous=dichotomous)
                divergent_matches.append(divergent_match)

                # Todo: does Björn have only 1 construct per item???
                if input_entry['constructs'] == output_item['constructs']:
                    convergent_match = compare_obj(input_entry['stems'], output_item['stems'], dichotomous=dichotomous)
                    convergent_matches.append(convergent_match)
                else:
                    convergent_matches.append({'stems': None, 'similarity': 0})

            divergent_similarity_max = max([i['similarity'] for i in divergent_matches])
            output_item['divergent_similarity'] = divergent_similarity_max

            if not dichotomous:
                divergent_top_matches = list(
                    filter(lambda x: x['similarity'] == divergent_similarity_max, divergent_matches))
                output_item['divergent_matches'] = [i['stems'] for i in divergent_top_matches]
            else:
                output_item['divergent_matches'] = None

            convergent_similarity_max = max([i['similarity'] for i in convergent_matches])
            output_item['convergent_similarity'] = convergent_similarity_max

            if convergent_similarity_max and not dichotomous:
                convergent_top_matches = list(
                    filter(lambda x: x['similarity'] == convergent_similarity_max, convergent_matches))
                output_item['convergent_matches'] = [i['stems'] for i in convergent_top_matches]
            else:
                output_item['convergent_matches'] = None

    return parsed_output

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
