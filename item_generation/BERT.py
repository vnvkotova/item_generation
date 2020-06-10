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

# from fast_bert.modeling import BertForMultiLabelSequenceClassification
from transformers import BertForSequenceClassification, BertConfig, AutoModelForSequenceClassification, AutoConfig

from fast_bert.data_cls import BertDataBunch, InputExample, InputFeatures, MultiLabelTextProcessor, convert_examples_to_features
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy_multilabel, accuracy_thresh, roc_auc

from .metrics import fbeta

from torch.nn import BCEWithLogitsLoss

# Todo I assume that it's used in BertLearner in model = model_class[1].from_pretrained(...)
class Modified_BertForMultiLabelSequenceClassification(BertForSequenceClassification):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].
    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_labels = 2
    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
    ):

        import pdb

        # pdb.set_trace()
        outputs = self.bert(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()

            loss = loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1, self.num_labels)
            )
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


MODEL_CLASSES = {
    "bert": (
        BertConfig,
        (BertForSequenceClassification, Modified_BertForMultiLabelSequenceClassification),
        BertTokenizer,
    ),
}

class ModifiedBertLearner(BertLearner):

    @staticmethod
    def from_pretrained_model(
        dataBunch,
        pretrained_path,
        output_dir,
        metrics,
        device,
        logger,
        finetuned_wgts_path=None,
        multi_gpu=True,
        is_fp16=True,
        loss_scale=0,
        warmup_steps=0,
        fp16_opt_level="O1",
        grad_accumulation_steps=1,
        multi_label=False,
        max_grad_norm=1.0,
        adam_epsilon=1e-8,
        logging_steps=100,
        freeze_transformer_layers=False
    ):

        model_state_dict = None

        model_type = dataBunch.model_type

        if torch.cuda.is_available():
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = 'cpu'

        if finetuned_wgts_path:
            model_state_dict = torch.load(finetuned_wgts_path, map_location=map_location)
        else:
            model_state_dict = None

        if multi_label is True:
            config_class, model_class, _ = MODEL_CLASSES[model_type]

            config = config_class.from_pretrained(
                str(pretrained_path), num_labels=len(dataBunch.labels)
            )

            model = model_class[1].from_pretrained(
                str(pretrained_path), config=config, state_dict=model_state_dict
            )
        else:
            config = AutoConfig.from_pretrained(
                str(pretrained_path), num_labels=len(dataBunch.labels)
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                str(pretrained_path), config=config, state_dict=model_state_dict
            )

        model.to(device)

        return ModifiedBertLearner(
            dataBunch,
            model,
            str(pretrained_path),
            output_dir,
            metrics,
            device,
            logger,
            multi_gpu,
            is_fp16,
            loss_scale,
            warmup_steps,
            fp16_opt_level,
            grad_accumulation_steps,
            multi_label,
            max_grad_norm,
            adam_epsilon,
            logging_steps,
            freeze_transformer_layers
        )

    def predict_batch(self, texts=None, return_dict=True):
        """
        Return unsorted Predictions
        :param texts:
        :param return_dict
        :return:
        """

        if texts:
            dl = self.data.get_dl_from_texts(texts)
        elif self.data.test_dl:
            dl = self.data.test_dl
        else:
            dl = self.data.val_dl

        all_logits = None

        self.model.eval()
        for step, batch in enumerate(dl):
            batch = tuple(t.to(self.device) for t in batch)

            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": None}

            if self.model_type in ["bert", "xlnet"]:
                inputs["token_type_ids"] = batch[2]

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs[0]
                if self.multi_label:
                    logits = logits.sigmoid()
                # elif len(self.data.labels) == 2:
                #     logits = logits.sigmoid()
                else:
                    logits = logits.softmax(dim=1)

            if all_logits is None:
                all_logits = logits.detach().cpu().numpy()
            else:
                all_logits = np.concatenate(
                    (all_logits, logits.detach().cpu().numpy()), axis=0
                )
        if return_dict:
            result_df = pd.DataFrame(all_logits, columns=self.data.labels)
            results = result_df.to_dict("record")
        else:
            results = all_logits

        # return [sorted(x.items(), key=lambda kv: kv[1], reverse=True) for x in results]
        return results


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
    metrics.append({'name': 'fbeta', 'function': fbeta})

    # Sigmoid layer and Binary Cross Entropy loss (Todo: is it like Softmax activation plus a Cross-Entropy loss?)
    learner = ModifiedBertLearner.from_pretrained_model(databunch, pretrained_path=args.model_name, metrics=metrics,
                                                device=device, logger=logger, output_dir=args.output_dir,
                                                finetuned_wgts_path=args["finetuned_dir"], warmup_steps=args.warmup_steps,
                                                multi_gpu=args.multi_gpu, is_fp16=args.fp16,
                                                multi_label=True, logging_steps=0)

    learner.fit(args.num_train_epochs, args.learning_rate, validate=True)

    print(learner)

    return learner
