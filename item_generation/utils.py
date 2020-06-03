import re
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np


def preprocess_db(db_list):
    """
    Todo
    :param db_list:
    :return:
    """

    for i, sentence in enumerate(db_list):
        sentence = sentence.lower()
        sentence = re.sub(r'.*@', r'', sentence)
        if sentence[:2] == "i ":
            sentence = sentence[2:]
        db_list[i] = sentence.split(".", 1)[0]
        db_list[i] = sentence.split("<|e", 1)[0]

    return db_list


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
        if sentence[:2] == "i ":
            sentence = sentence[2:]
        generated_list[i] = re.sub(r'.*@', r'', sentence)

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
