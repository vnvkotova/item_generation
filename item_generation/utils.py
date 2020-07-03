import re
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import edlib


def convert_into_training_file(df_to_convert):

    list_converted = []

    for index, row in df_to_convert.iterrows():
        str_temp = "<|startoftext|>"
        for label in row["label"]:
            str_temp = str_temp + "#" + label
        str_temp = str_temp + "@" + row["text"] + "<|endoftext|>"
        list_converted.append(str_temp)

    return list_converted


def F_score_item(list_actual, list_target):

    num_correctly_predicted = len(set(list_actual) & set(list_target))
    F_score = num_correctly_predicted/(len(list_actual) + len(list_target))

    return F_score


def preprocess_generated_items(list_generated_items):
    """
    Todo
    :param list_generated_items:
    :return:
    """

    list_items = []
    list_labels = []

    for item in list_generated_items:
        item = re.sub('\<\|startoftext\|\>#', '', item)
        item = re.sub('\<\|endoftext\|\>', '', item)
        if item.find("@") != -1:
            list_splited_str = item.split("@")
            item_with_dot = list_splited_str[1] + "."
            list_items.append(item_with_dot)
            list_labels.append(list_splited_str[0].split("#"))
        else:
            list_items.append(item)
            list_labels.append("")

    dict_generated_items = {"items": list_items, "labels": list_labels}

    return dict_generated_items


def levenshtein_distance(obj1, obj2):
    # Calculates the normalized Levenshtein distance as a similarity metrics between two strings

    # Parameters:
    #    obj1 (str):String or list
    #    obj2 (str):String or list

    # Returns:
    #    Inverted, normalized Levenshtein distance, whereas 1 = identical match
    obj1 = ''.join(obj1)
    obj2 = ''.join(obj2)
    x = edlib.align(obj1, obj2)

    normalized_distance = 1 - (x['editDistance'] / max(len(obj1), len(obj2)))

    return normalized_distance


def compare_obj(obj1, obj2, dichotomous=True):
    obj1 = [obj1] if isinstance(obj1, str) else obj1
    obj2 = [obj2] if isinstance(obj2, str) else obj2
    if not isinstance(obj1, list) or not isinstance(obj2, list):
        print('Error: `obj1` and obj2 must be strings or lists but are `', type(obj1), '` and `', type(obj2), '`!')
        return False

    if dichotomous:
        similarity = 1 if len([i for i in obj1 if i in obj2]) > 0 else 0
    else:
        similarity = min([levenshtein_distance(i, obj2) for i in obj1])

    result = {
        'stems': obj1,
        'similarity': similarity
    }

    return result

def preprocess_db(db_list):
    """
    Todo
    :param db_list:
    :return:
    """
    preprocessed_db_list = []
    for i, sentence in enumerate(db_list):
        sentence = sentence.lower()
        sentence = re.sub(r'.*@', r'', sentence)
        if sentence[:2] == "i ":
            sentence = sentence[2:]
        sentence = sentence.split(".", 1)[0]
        preprocessed_db_list.append(sentence.split("<|e", 1)[0])

    return preprocessed_db_list


def preprocess_generated_list(generated_list):
    """
    Todo
    :param generated_list:
    :return:
    """

    # deletes the labels in the beginning of generated sentences, lowercases the string, deletes "." from the end
    # of the sentence if it's there and similarly "I " from the beginning
    for i, sentence in enumerate(generated_list):
        sentence = sentence.lower()
        if sentence[-1] == ".":
            sentence = sentence[:-1]
        sentence = re.sub(r'.*@', r'', sentence)
        if sentence[:2] == "i ":
            generated_list[i] = sentence[2:]

    return generated_list


def input_file_to_list(file_path):
    """
    a function to convert an input training file with incorporated labels,  <|startoftext|> and <|endoftext|> into a list of items
    :param file_path: Todo
    :return: Todo
    """

    f = open(file_path, 'r+', encoding="utf-8")
    list_lines = []

    for line in f:
        list_lines.append(line)

    count = 0
    for item in list_lines:
        list_lines[count] = re.sub(r'.*@', r'', item)
        count += 1

    count = 0
    for item in list_lines:
        list_lines[count] = re.sub(r'<.*', '', item)
        count += 1

    return list_lines


def prepare_IPIP_file_for_Multilabel(xlsx_path):
    """

    :param xlsx_path: Todo
    :return: Todo
    """
    df_IPIP = pd.read_excel(xlsx_path)
    mlb = MultiLabelBinarizer()
    array_labels = mlb.fit_transform(df_IPIP.groupby("text").label.apply(list))
    array_items = np.sort(df_IPIP.text.unique())
    array_items = np.expand_dims(array_items, axis=1)
    array_data = np.concatenate((array_items, array_labels), axis=1)
    list_columns = ['item'] + mlb.classes_.tolist()
    df_output = pd.DataFrame(array_data, columns=list_columns)
    return df_output


def prepare_additional_IPIP(csv_IPIP_path, csv_additional_path):
    # set(df_val.columns) - set(df_new_Bert.columns)
    df_IPIP = pd.read_csv(csv_IPIP_path, index_col=0)
    labels = df_IPIP.columns[1:]

    df_val = pd.read_excel(csv_additional_path)
    mlb = MultiLabelBinarizer()
    array_labels = mlb.fit_transform(df_val.groupby("text").label.apply(list))
    array_items = np.sort(df_val.text.unique())
    array_items = np.expand_dims(array_items, axis=1)
    array_data = np.concatenate((array_items, array_labels), axis=1)
    list_columns = ['item'] + mlb.classes_.tolist()
    df_output = pd.DataFrame(array_data, columns=list_columns)

    for labels_item in labels:
        if labels_item not in df_output:
            df_output[labels_item] = pd.Series(np.zeros(array_labels.shape[0]), index=df_output.index)

    df_output.rename(columns={df_output.columns[0]: "item"}, inplace=True)
    df_output = df_output.reindex(sorted(df_output.columns), axis=1)
    df_output = df_output[['item'] + [col for col in df_output.columns if col != 'item']]
    # df_val = df_val[['id'] + [col for col in df_val.columns if col != 'id']]
    df_output.update(df_output[['item']].applymap('\"{}\"'.format))
    # df.to_csv(save_file_name) #df.to_csv("/Users/veronikakotova/Desktop/TUM/thesis/val.csv")
    return df_output


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x
