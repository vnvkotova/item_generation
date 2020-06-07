from transformers import BertTokenizer
from pathlib import Path
import torch

from box import Box
import pandas as pd
import collections
import os
from tqdm import tqdm, trange
import sys
import random
import numpy as np
import apex
from sklearn.model_selection import train_test_split

import logging

import datetime

from fast_bert.modeling import BertForMultiLabelSequenceClassification
from fast_bert.data_cls import BertDataBunch, InputExample, InputFeatures, MultiLabelTextProcessor, convert_examples_to_features
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy_multilabel, accuracy_thresh, fbeta, roc_auc


def train_multilabel_BERT(args):

    torch.cuda.empty_cache()

    pd.set_option('display.max_colwidth', -1)
    run_start_time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

    logfile = str(args["log_path"] / 'log-{}-{}.txt'.format(run_start_time, args["run_text"]))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler(sys.stdout)
        ])

    logger = logging.getLogger()

    device = torch.device('cuda')
    if torch.cuda.device_count() > 1:
        args.multi_gpu = True
    else:
        args.multi_gpu = False

    df_train_labels = pd.read_csv(args["train_labels"])
    label_cols = list(df_train_labels.columns)[2:]

    # takes training, validation and test csv files and converts the data into internal representation for BERT, it also instantiates the correct data loaders
    databunch = BertDataBunch(args['data_dir'], args["label_dir"], args.model_name,
                              train_file='train.csv', val_file='val.csv',
                              test_data='test.csv',
                              text_col="item", label_col=label_cols,
                              batch_size_per_gpu=args['train_batch_size'], max_seq_length=args['max_seq_length'],
                              multi_gpu=args.multi_gpu,
                              multi_label=True, model_type=args.model_type)

    num_labels = len(databunch.labels)

    metrics = []
    metrics.append({'name': 'accuracy_thresh', 'function': accuracy_thresh})
    metrics.append({'name': 'roc_auc', 'function': roc_auc})
    metrics.append({'name': 'fbeta_modified', 'function': fbeta_modified})

    # Sigmoid layer and Binary Cross Entropy loss (Todo: is it like Softmax activation plus a Cross-Entropy loss?)
    learner = BertLearner.from_pretrained_model(databunch, pretrained_path=args.model_name, metrics=metrics,
                                                device=device, logger=logger, output_dir=args.output_dir,
                                                finetuned_wgts_path=args["finetuned_dir"], warmup_steps=args.warmup_steps,
                                                multi_gpu=args.multi_gpu, is_fp16=args.fp16,
                                                multi_label=True, logging_steps=0)

    learner.fit(args.num_train_epochs, args.learning_rate, validate=True)

    return learner