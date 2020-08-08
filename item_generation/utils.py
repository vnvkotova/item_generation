import re
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import edlib
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
from transformers import pipeline


def synonym_generation(initial_data):
    """

    :param initial_data:
    :return:
    """

    nlp = pipeline("fill-mask")

    list_initial_items = []
    list_augmented_items = []
    list_instruments = []
    list_alphas = []
    list_labels = []
    list_keys = []
    list_training_data = []
    for document in initial_data.find():
        list_tags = pos_tag(word_tokenize(document["text"]))
        for tuple_word in list_tags:
            if tuple_word[1] == 'RB' or tuple_word[1] == 'JJ' or tuple_word[1] == 'NN':
                for syn in wordnet.synsets(tuple_word[0]):
                    for l in syn.lemmas():
                        synonym_item = re.sub(tuple_word[0], l.name(), document["text"])
                        if not any(map(str.isdigit, synonym_item)) and (synonym_item not in list_augmented_items) and (
                                document["text"] != synonym_item) and (":)" not in synonym_item):
                            list_initial_items.append(document["text"])
                            list_augmented_items.append(synonym_item)
                            list_instruments.append(document["instrument"])
                            list_alphas.append(document["alpha"])
                            list_labels.append(document["label"])
                            list_keys.append(document["key"])
                            new_training_data = document["training_data"].split("@", 1)[
                                                    0] + "@" + synonym_item + ".<|endoftext|>"
                            list_training_data.append(new_training_data)
                        if l.antonyms() and l.antonyms()[0].name() != "":
                            antomym_item = re.sub(tuple_word[0], l.antonyms()[0].name(), document["text"])
                            if not any(map(str.isdigit, antomym_item)) and (
                                    antomym_item not in list_augmented_items) and (
                                    document["text"] != antomym_item) and (":)" not in antomym_item):
                                list_initial_items.append(document["text"])
                                list_augmented_items.append(antomym_item)
                                list_instruments.append(document["instrument"])
                                list_alphas.append(document["alpha"])
                                list_labels.append(document["label"])
                                list_keys.append(document["key"])
                                new_training_data = document["training_data"].split("@", 1)[
                                                        0] + "@" + antomym_item + ".<|endoftext|>"
                                list_training_data.append(new_training_data)
                masked_item = re.sub(tuple_word[0], "<mask>", document["text"], 1)
                list_filled_outputs = nlp(masked_item)
                filled_sentence = ""
                for dict_filled in list_filled_outputs:
                    filled_sentence = dict_filled["sequence"]
                    filled_sentence = re.sub("<s> ", "", filled_sentence)
                    filled_sentence = re.sub("</s>", "", filled_sentence)
                unmasked_spaces = document["text"].count(' ')
                masked_spaces = filled_sentence.count(' ')
                if masked_spaces >= unmasked_spaces and (not any(map(str.isdigit, filled_sentence))) and (
                        filled_sentence not in list_augmented_items) and (document["text"] != filled_sentence) and (
                        ":)" not in filled_sentence):
                    list_initial_items.append(document["text"])
                    list_augmented_items.append(filled_sentence)
                    list_instruments.append(document["instrument"])
                    list_alphas.append(document["alpha"])
                    list_labels.append(document["label"])
                    list_keys.append(document["key"])
                    new_training_data = document["training_data"].split("@", 1)[
                                            0] + "@" + filled_sentence + ".<|endoftext|>"
                    list_training_data.append(new_training_data)

    dict_augmented = {"initial_item": list_initial_items, "augmented_item": list_augmented_items,
                      "instrument": list_instruments,
                      "alpha": list_alphas, "label": list_labels, "key": list_keys, "training_data": list_training_data}
    df_synonyms = pd.DataFrame(dict_augmented)

    return df_synonyms


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
            list_items.append(list_splited_str[1][:-1])
            list_labels.append(list_splited_str[0].split("#")[1:])
        else:
            list_items.append(item)
            list_labels.append([""])

    dict_generated_items = {"items": list_items, "labels": list_labels}

    return dict_generated_items


def preprocess_generated_items_tuples(list_generated_items):
    """
    Todo
    :param list_generated_items:
    :return:
    """

    list_tuples = []
    temp_list_items = []

    for item in list_generated_items:
        item = re.sub('\<\|startoftext\|\>#', '', item)
        item = re.sub('\<\|endoftext\|\>', '', item)
        if item.find("@") != -1:
            list_splited_str = item.split("@")
            list_tuples.append(((list_splited_str[1][:-1], item),list_splited_str[0].split("#")[1:]))
            temp_list_items.append(list_splited_str[1][:-1])
        else:
            list_tuples.append(((item, item), [""]))
            temp_list_items.append(item)

    overfit_repeated_items = len(temp_list_items) - len(set(temp_list_items))

    return (list_tuples, overfit_repeated_items)


def augment_data(train_data):

    # Todo should it return a dictionary???

    list_texts = []
    list_instruments = []
    list_alphas = []
    list_labels = []
    list_keys = []
    list_training_data = []
    for document in train_data.find():
        if document["text"][:2] == "I ":
            new_text = document["text"][2:].capitalize()
        else:
            new_text = "I " + document["text"][0].lower() + document["text"][1:]
        list_texts.append(new_text)
        list_instruments.append(document["instrument"])
        list_alphas.append(document["alpha"])
        list_labels.append(document["label"])
        list_keys.append(document["key"])
        new_training_data = document["training_data"].split("@", 1)[0] + "@" + new_text + ".<|endoftext|>"
        list_training_data.append(new_training_data)

    return None


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
