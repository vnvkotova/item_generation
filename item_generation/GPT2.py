"""
Todo
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import logging
import math
import os
from dataclasses import dataclass, field

import pandas as pd
import re

import random

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
    GPT2Tokenizer,
    GPT2LMHeadModel,
)

from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, EvalPrediction, TrainOutput
from transformers.training_args import TrainingArguments

from typing import Callable, Dict, Optional, Tuple

from tqdm.auto import tqdm, trange

import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

from item_generation.utils import preprocess_db
from item_generation.metrics import db_match, overfit_count

try:
    from apex import amp

    _has_apex = True
except ImportError:
    _has_apex = False


def is_apex_available():
    return _has_apex

try:
    from torch.utils.tensorboard import SummaryWriter

    _has_tensorboard = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter

        _has_tensorboard = True
    except ImportError:
        _has_tensorboard = False


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """
    model_name_or_path: str
    config_name: str = None
    cache_dir: str = None


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training.
    """
    train_data_file: str
    line_by_line: bool
    block_size: int
    overwrite_cache: bool


def get_dataset(args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, evaluate=False, local_rank=-1):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)
    else:
        return TextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)


class ExtendedTrainer(Trainer):
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch,
    optimized for Transformers.
    """
    #
    # model: PreTrainedModel
    # args: TrainingArguments
    # data_collator: DataCollator
    # train_dataset: Optional[Dataset]
    # eval_dataset: Optional[Dataset]
    # compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None
    # prediction_loss_only: bool
    # tb_writer: Optional["SummaryWriter"] = None
    # optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None
    # global_step: Optional[int] = None
    # epoch: Optional[float] = None

    def train_augmentation(self, tokenizer, train_data, data_base, num_val_items, val_top_k, val_top_p, data_args,
                           training_args, model_path: Optional[str] = None):
        """
        Main training entry point.
        Args:
            model_path:
                (Optional) Local path to model if model to train has been instantiated from a local path
                If present, we will try reloading the optimizer/scheduler states from there.
            Todo
        """

        # Todo: decide what has to be done in case a txt file was passed as train_data

        if type(data_base) == str or type(train_data) == str:
            logger.warning("Sorry, not all functionality is available if training data and the database itself are"
                           "not in mongoDB.")

        current_item = train_data.find_one({"_id": 0})["initial_item"]
        list_items_intervals = []
        item_start = 0
        current_index = 0
        for item in train_data.find():
            if item["initial_item"] != current_item:
                list_items_intervals.append((item_start, current_index - 1))
                current_item = item["initial_item"]
                item_start = current_index
            current_index = current_index + 1
        list_items_intervals.append((item_start, current_index))

        list_training_data = []
        for item_interval in list_items_intervals:
            item_id = random.randint(item_interval[0], item_interval[1])
            list_training_data.append(train_data.find_one({"_id": item_id})["training_data"])
        f = open(data_args.train_data_file, 'w')
        for item in list_training_data:
            f.write(item + '\n')
        f.close()

        self.train_dataset = get_dataset(data_args, tokenizer=tokenizer, local_rank=training_args.local_rank)
        train_dataloader = self.get_train_dataloader()

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        optimizer, scheduler = self.get_optimizers(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(model_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model
        model.to(self.args.device)
        if self.args.fp16:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", self.args.to_json_string())
            self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        # Train!
        total_train_batch_size = (
            self.args.train_batch_size
            * self.args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1))
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_gpu_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = 0.0
        logging_loss = 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch", disable=not self.is_local_master()
        )

        list_losses = []
        prev_loss = 0
        # plt.ion()

        metric_overfit_items = []
        metric_overfit_sentences = []
        metric_overfit_repeated_items = []
        metric_overfit_repeated_sentences = []
        metric_classification_overfit_items = []
        metric_classification_overfit_sentences = []
        metric_classification_labels = []
        metric_classification_F_score = []
        current_epoch = 0

        metric_library_items = []
        metric_classification_library_F_score = []

        list_training_items = []
        for document in train_data.find():
            list_training_items.append(document["augmented_item"])
        list_training_items = list(set(list_training_items))

        list_library_items = None
        if data_base is not None:
            list_library_items = []
            for document in data_base.find():
                list_library_items.append(document["augmented_item"])
            list_library_items = list(set(list_library_items)-set(list_training_items))

        for epoch in train_iterator:

            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=not self.is_local_master())
            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                tr_loss += self._training_step(model, inputs, optimizer)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                        self.args.gradient_accumulation_steps >= len(epoch_iterator) == (step + 1)
                ):
                    if self.args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    optimizer.step()

                    scheduler.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if self.is_local_master():
                        if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                            self.global_step == 1 and self.args.logging_first_step
                        ):
                            logs: Dict[str, float] = {}
                            logs["loss"] = (tr_loss - logging_loss) / self.args.logging_steps
                            logs["learning_rate"] = scheduler.get_last_lr()[0]
                            logging_loss = tr_loss

                            self._log(logs)

                            if self.args.evaluate_during_training:
                                self.evaluate()

                        if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                            # In all cases (even distributed/parallel), self.model is always a reference
                            # to the model we want to save.
                            if hasattr(model, "module"):
                                assert model.module is self.model
                            else:
                                assert model is self.model
                            # Save model checkpoint
                            output_dir = os.path.join(
                                self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}"
                            )

                            self.save_model(output_dir)
                            self._rotate_checkpoints()
                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            logger.info("Saving optimizer and scheduler states to %s", output_dir)

                if 0 < self.args.max_steps < self.global_step:
                    epoch_iterator.close()
                    break

            depicted_loss = (tr_loss - prev_loss) / self.args.logging_steps
            prev_loss = tr_loss
            list_losses.append(depicted_loss)
            # plt.plot(list_losses, label='current loss')
            # plt.show()

            model.eval()
            text = "<|startoftext|>#"
            indexed_tokens = tokenizer.encode(text, return_tensors='pt')
            indexed_tokens = indexed_tokens.to('cuda')
            sample_outputs = model.generate(
                indexed_tokens,
                do_sample=True,
                max_length=50,
                top_k=val_top_k,
                top_p=val_top_p,
                num_return_sequences=num_val_items
            )

            decoded_outputs = []

            for i, sample_output in enumerate(sample_outputs):
                temp_sentence = tokenizer.decode(sample_output, skip_special_tokens=True)
                logger.info(temp_sentence)
                decoded_outputs.append(temp_sentence)
            model.train()

            logger.info("------------------------------------------ Metrics ------------------------------------------")

            dict_metrics_epoch = overfit_count(decoded_outputs, train_data, list_training_items, data_base, list_library_items)

            metric_overfit_items.append(dict_metrics_epoch["overfit_items"])
            metric_overfit_sentences.append(dict_metrics_epoch["overfit_sentences"])
            metric_overfit_repeated_items.append(dict_metrics_epoch["overfit_repeated_items"])
            metric_overfit_repeated_sentences.append(dict_metrics_epoch["overfit_repeated_sentences"])
            metric_classification_overfit_items.append(dict_metrics_epoch["classification_overfit_items"])
            metric_classification_overfit_sentences.append(dict_metrics_epoch["classification_overfit_sentences"])
            metric_classification_labels.append(dict_metrics_epoch["classification_labels"])
            metric_classification_F_score.append(dict_metrics_epoch["classification_F_score"])

            fig = plt.figure()
            fig.set_size_inches(12, 12)

            if data_base is not None:

                metric_library_items.append(dict_metrics_epoch["library_items"])
                metric_classification_library_F_score.append(dict_metrics_epoch["classification_library_F_score"])

                ax0 = plt.subplot2grid((3, 2), (0, 0), rowspan=1, colspan=2, title="Loss function")
                ax1 = plt.subplot2grid((3, 2), (1, 0), rowspan=1, colspan=1, title="Overfit metrics")
                ax2 = plt.subplot2grid((3, 2), (1, 1), rowspan=1, colspan=1, title="Classification metrics")
                ax3 = plt.subplot2grid((3, 2), (2, 0), rowspan=1, colspan=2, title="Semantics metrics")

                ax0.plot(list_losses)

                ax1.plot(metric_overfit_items)
                ax1.plot(metric_overfit_sentences)
                ax1.plot(metric_overfit_repeated_items)
                ax1.plot(metric_overfit_repeated_sentences)
                ax1.legend(["Similar items", "Similar sentences", "Repeated items", "Repeated sentences"])

                ax2.plot(metric_classification_overfit_items)
                ax2.plot(metric_classification_overfit_sentences)
                ax2.plot(metric_classification_labels)
                ax2.plot(metric_classification_F_score)
                ax2.legend(["Overfited items", "Overfited sentences", "Overfited labels", "F score"])

                ax3.plot(metric_library_items)
                ax3.plot(metric_classification_library_F_score)
                ax3.legend(["Library items", "F score"])
            else:
                ax0 = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=2, title="Loss function")
                ax1 = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1, title="Overfit metrics")
                ax2 = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1, title="Classification metrics")

                ax0.plot(list_losses)

                ax1.plot(metric_overfit_items)
                ax1.plot(metric_overfit_sentences)
                ax1.plot(metric_overfit_repeated_items)
                ax1.plot(metric_overfit_repeated_sentences)
                ax1.legend(["Similar items", "Similar sentences", "Repeated items", "Repeated sentences"])

                ax2.plot(metric_classification_overfit_items)
                ax2.plot(metric_classification_overfit_sentences)
                ax2.plot(metric_classification_labels)
                ax2.plot(metric_classification_F_score)
                ax2.legend(["Overfited items", "Overfited sentences", "Overfited labels", "F score"])

            plt.tight_layout()
            plt_name = self.args.output_dir + "/model_preformace.png"
            plt.savefig(plt_name)
            plt.show()

            if 0 < self.args.max_steps < self.global_step:
                train_iterator.close()
                break
            if self.args.tpu_metrics_debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

            list_training_data = []
            for item_interval in list_items_intervals:
                item_id = random.randint(item_interval[0], item_interval[1])
                list_training_data.append(train_data.find_one({"_id": item_id})["training_data"])
            f = open(data_args.train_data_file, 'w')
            for item in list_training_data:
                f.write(item + '\n')
            f.close()

            self.train_dataset = get_dataset(data_args, tokenizer=tokenizer, local_rank=training_args.local_rank)
            train_dataloader = self.get_train_dataloader()

        if self.tb_writer:
            self.tb_writer.close()

        if data_base is not None:
            dict_metrics = {"Losses": list_losses,
                            "Similar_items": metric_overfit_items, "Similar_sentences": metric_overfit_sentences,
                            "Overfit_repeated_items": metric_overfit_repeated_items,
                            "Overfit_repeated_sentences": metric_overfit_repeated_sentences,
                            "Class_overfited_items": metric_classification_overfit_items,
                            "Class_overfited_sentences": metric_classification_overfit_sentences,
                            "Class_overfited_labels": metric_classification_labels,
                            "Class_F_score": metric_classification_F_score,
                            "Library_items": metric_library_items,
                            "Classification_library_F_score": metric_classification_library_F_score}
        else:
            dict_metrics = {"Losses": list_losses,
                            "Similar_items": metric_overfit_items, "Similar_sentences": metric_overfit_sentences,
                            "Overfit_repeated_items": metric_overfit_repeated_items,
                            "Overfit_repeated_sentences": metric_overfit_repeated_sentences,
                            "Class_overfited_items": metric_classification_overfit_items,
                            "Class_overfited_sentences": metric_classification_overfit_sentences,
                            "Class_overfited_labels": metric_classification_labels,
                            "Class_F_score": metric_classification_F_score}
        df_metrics = pd.DataFrame(dict_metrics)
        excel_name = self.args.output_dir + "/metrics.xlsx"
        df_metrics.to_excel(excel_name)

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss / self.global_step)

    def train(self, tokenizer, train_data, data_base, num_val_items, val_top_k, val_top_p,
              model_path: Optional[str] = None):
        """
        Main training entry point.
        Args:
            model_path:
                (Optional) Local path to model if model to train has been instantiated from a local path
                If present, we will try reloading the optimizer/scheduler states from there.
            Todo
        """

        # Todo: decide what has to be done in case a txt file was passed as train_data

        if type(data_base) == str or type(train_data) == str:
            logger.warning("Sorry, not all functionality is available if training data and the database itself are"
                           "not in mongoDB.")

        train_dataloader = self.get_train_dataloader()

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        optimizer, scheduler = self.get_optimizers(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(model_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model
        model.to(self.args.device)
        if self.args.fp16:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", self.args.to_json_string())
            self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        # Train!
        total_train_batch_size = (
            self.args.train_batch_size
            * self.args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1))
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_gpu_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = 0.0
        logging_loss = 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch", disable=not self.is_local_master()
        )

        list_losses = []
        prev_loss = 0
        # plt.ion()

        metric_overfit_items = []
        metric_overfit_sentences = []
        metric_overfit_repeated_items = []
        metric_overfit_repeated_sentences = []
        metric_classification_overfit_items = []
        metric_classification_overfit_sentences = []
        metric_classification_labels = []
        metric_classification_F_score = []
        current_epoch = 0

        metric_library_items = []
        metric_classification_library_F_score = []

        list_training_items = []
        for document in train_data.find():
            list_training_items.append(document["augmented_item"])
        list_training_items = list(set(list_training_items))

        list_library_items = None
        if data_base is not None:
            list_library_items = []
            for document in data_base.find():
                list_library_items.append(document["augmented_item"])
            list_library_items = list(set(list_library_items)-set(list_training_items))

        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=not self.is_local_master())
            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                tr_loss += self._training_step(model, inputs, optimizer)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                        self.args.gradient_accumulation_steps >= len(epoch_iterator) == (step + 1)
                ):
                    if self.args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    optimizer.step()

                    scheduler.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if self.is_local_master():
                        if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                            self.global_step == 1 and self.args.logging_first_step
                        ):
                            logs: Dict[str, float] = {}
                            logs["loss"] = (tr_loss - logging_loss) / self.args.logging_steps
                            logs["learning_rate"] = scheduler.get_last_lr()[0]
                            logging_loss = tr_loss

                            self._log(logs)

                            if self.args.evaluate_during_training:
                                self.evaluate()

                        if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                            # In all cases (even distributed/parallel), self.model is always a reference
                            # to the model we want to save.
                            if hasattr(model, "module"):
                                assert model.module is self.model
                            else:
                                assert model is self.model
                            # Save model checkpoint
                            output_dir = os.path.join(
                                self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}"
                            )

                            self.save_model(output_dir)
                            self._rotate_checkpoints()
                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            logger.info("Saving optimizer and scheduler states to %s", output_dir)

                if 0 < self.args.max_steps < self.global_step:
                    epoch_iterator.close()
                    break

            depicted_loss = (tr_loss - prev_loss) / self.args.logging_steps
            prev_loss = tr_loss
            list_losses.append(depicted_loss)
            # plt.plot(list_losses, label='current loss')
            # plt.show()

            model.eval()
            text = "<|startoftext|>#"
            indexed_tokens = tokenizer.encode(text, return_tensors='pt')
            indexed_tokens = indexed_tokens.to('cuda')
            sample_outputs = model.generate(
                indexed_tokens,
                do_sample=True,
                max_length=50,
                top_k=val_top_k,
                top_p=val_top_p,
                num_return_sequences=num_val_items
            )

            decoded_outputs = []

            for i, sample_output in enumerate(sample_outputs):
                temp_sentence = tokenizer.decode(sample_output, skip_special_tokens=True)
                logger.info(temp_sentence)
                decoded_outputs.append(temp_sentence)
            model.train()

            logger.info("------------------------------------------ Metrics ------------------------------------------")

            dict_metrics_epoch = overfit_count(decoded_outputs, train_data, list_training_items, data_base, list_library_items)

            metric_overfit_items.append(dict_metrics_epoch["overfit_items"])
            metric_overfit_sentences.append(dict_metrics_epoch["overfit_sentences"])
            metric_overfit_repeated_items.append(dict_metrics_epoch["overfit_repeated_items"])
            metric_overfit_repeated_sentences.append(dict_metrics_epoch["overfit_repeated_sentences"])
            metric_classification_overfit_items.append(dict_metrics_epoch["classification_overfit_items"])
            metric_classification_overfit_sentences.append(dict_metrics_epoch["classification_overfit_sentences"])
            metric_classification_labels.append(dict_metrics_epoch["classification_labels"])
            metric_classification_F_score.append(dict_metrics_epoch["classification_F_score"])

            fig = plt.figure()
            fig.set_size_inches(12, 12)

            if data_base is not None:

                metric_library_items.append(dict_metrics_epoch["library_items"])
                metric_classification_library_F_score.append(dict_metrics_epoch["classification_library_F_score"])

                ax0 = plt.subplot2grid((3, 2), (0, 0), rowspan=1, colspan=2, title="Loss function")
                ax1 = plt.subplot2grid((3, 2), (1, 0), rowspan=1, colspan=1, title="Overfit metrics")
                ax2 = plt.subplot2grid((3, 2), (1, 1), rowspan=1, colspan=1, title="Classification metrics")
                ax3 = plt.subplot2grid((3, 2), (2, 0), rowspan=1, colspan=2, title="Semantics metrics")

                ax0.plot(list_losses)

                ax1.plot(metric_overfit_items)
                ax1.plot(metric_overfit_sentences)
                ax1.plot(metric_overfit_repeated_items)
                ax1.plot(metric_overfit_repeated_sentences)
                ax1.legend(["Similar items", "Similar sentences", "Repeated items", "Repeated sentences"])

                ax2.plot(metric_classification_overfit_items)
                ax2.plot(metric_classification_overfit_sentences)
                ax2.plot(metric_classification_labels)
                ax2.plot(metric_classification_F_score)
                ax2.legend(["Overfited items", "Overfited sentences", "Overfited labels", "F score"])

                ax3.plot(metric_library_items)
                ax3.plot(metric_classification_library_F_score)
                ax3.legend(["Library items", "F score"])
            else:
                ax0 = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=2, title="Loss function")
                ax1 = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1, title="Overfit metrics")
                ax2 = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1, title="Classification metrics")

                ax0.plot(list_losses)

                ax1.plot(metric_overfit_items)
                ax1.plot(metric_overfit_sentences)
                ax1.plot(metric_overfit_repeated_items)
                ax1.plot(metric_overfit_repeated_sentences)
                ax1.legend(["Similar items", "Similar sentences", "Repeated items", "Repeated sentences"])

                ax2.plot(metric_classification_overfit_items)
                ax2.plot(metric_classification_overfit_sentences)
                ax2.plot(metric_classification_labels)
                ax2.plot(metric_classification_F_score)
                ax2.legend(["Overfited items", "Overfited sentences", "Overfited labels", "F score"])

            plt.tight_layout()
            plt_name = self.args.output_dir + "/model_preformace.png"
            plt.savefig(plt_name)
            plt.show()

            if 0 < self.args.max_steps < self.global_step:
                train_iterator.close()
                break
            if self.args.tpu_metrics_debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

        if self.tb_writer:
            self.tb_writer.close()

        if data_base is not None:
            dict_metrics = {"Losses": list_losses,
                            "Similar_items": metric_overfit_items, "Similar_sentences": metric_overfit_sentences,
                            "Overfit_repeated_items": metric_overfit_repeated_items,
                            "Overfit_repeated_sentences": metric_overfit_repeated_sentences,
                            "Class_overfited_items": metric_classification_overfit_items,
                            "Class_overfited_sentences": metric_classification_overfit_sentences,
                            "Class_overfited_labels": metric_classification_labels,
                            "Class_F_score": metric_classification_F_score,
                            "Library_items": metric_library_items,
                            "Classification_library_F_score": metric_classification_library_F_score}
        else:
            dict_metrics = {"Losses": list_losses,
                            "Similar_items": metric_overfit_items, "Similar_sentences": metric_overfit_sentences,
                            "Overfit_repeated_items": metric_overfit_repeated_items,
                            "Overfit_repeated_sentences": metric_overfit_repeated_sentences,
                            "Class_overfited_items": metric_classification_overfit_items,
                            "Class_overfited_sentences": metric_classification_overfit_sentences,
                            "Class_overfited_labels": metric_classification_labels,
                            "Class_F_score": metric_classification_F_score}
        df_metrics = pd.DataFrame(dict_metrics)
        excel_name = self.args.output_dir + "/metrics.xlsx"
        df_metrics.to_excel(excel_name)

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss / self.global_step)


def train_GPT2(model_name_or_path, train_data, data_base, output_dir, augmentation = False,
               num_val_items = 30, val_top_k=30, val_top_p=0.95,
               config_name=None, cache_dir=None, line_by_line=True, block_size=-1,
               overwrite_cache=False, overwrite_output_dir=False, do_train=False, per_gpu_train_batch_size=8,
               gradient_accumulation_steps=1, learning_rate=5e-5, weight_decay=0.0, adam_epsilon=1e-8,
               max_grad_norm=1.0,
               num_train_epochs=3.0, max_steps=-1, warmup_steps=0, logging_dir=None, logging_first_step=False,
               logging_steps=500, save_steps=500, save_total_limit=None, no_cuda=False, seed=42, fp16=False,
               fp16_opt_level="O1", local_rank=-1, tpu_num_cores=None, tpu_metrics_debug=False,
               bos_token='<|endoftext|>'):
    """

    :param model_name_or_path: str, model argument, the model checkpoint for weights initialization
                                    on 16.05.2020 the following standard GPT-2 options were available:
                                    gpt2 – 12-layer, 768-hidden, 12-heads, 117M parameters, English
                                    gpt2-medium – 24-layer, 1024-hidden, 16-heads, 345M parameters, English
                                    gpt2-large – 36-layer, 1280-hidden, 20-heads, 774M parameters, English
                                    gpt2-xl – 48-layer, 1600-hidden, 25-heads, 1558M parameters, English
    :param train_data: str, data argument, the input training data file (a text file)
    :param data_base: Todo
    :param config_name: str, model argument, optional, pretrained config name or path if not the same as model_name
    :param cache_dir: str, model argument, optional, where do you want to store the pretrained models downloaded from s3
    :param line_by_line: bool, data argument, default to True, whether distinct lines of text in the dataset are to be
                                                                handled as distinct sequences
    :param block_size: int, data argument, default to the model max input length for single sentence inputs (take into
                                            account special tokens), sequence length after tokenization, the training
                                            dataset will be truncated in block of this size for training
    :param overwrite_cache: bool, data argument, default to False, overwrite the cached training set
    :param output_dir: str, training argument, the output directory where the model predictions and checkpoints will
                                                be written
    :param overwrite_output_dir: bool, training argument, overwrite the content of the output directory: use this to
                                                continue training if output_dir points to a checkpoint directory
    :param do_train: bool, training argument, whether to run training
    :param per_gpu_train_batch_size: int, training argument, batch size per GPU/CPU for training
    :param gradient_accumulation_steps: int, training argument, number of updates steps to accumulate before performing
                                                                a backward/update pass
    :param learning_rate: float, training argument, the initial learning rate for Adam
    :param weight_decay: float, training argument, weight decay if we apply some
    :param adam_epsilon: float, training argument, epsilon for Adam optimizer
    :param max_grad_norm: float, training argument, max gradient norm
    :param num_train_epochs: float, training argument, total number of training epochs to perform
    :param max_steps: int, training argument, if > 0: set total number of training steps to perform. Override num_train_epochs
    :param warmup_steps: int, training argument, linear warmup over warmup_steps
    :param logging_dir: str, tensorboard log dir
    :param logging_first_step: bool, training argument, log and eval the first global_step
    :param logging_steps: int, training argument, log every X updates steps
    :param save_steps: int, training argument, save checkpoint every X updates steps
    :param save_total_limit: int, training argument, limit the total amount of checkpoints: deletes the older
                                                        checkpoints in the output_dir. Default is unlimited checkpoints
    :param no_cuda: bool, training argument, do not use CUDA even when it is available
    :param seed: int, training argument, random seed for initialization
    :param fp16: bool, training argument, whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit
    :param fp16_opt_level: str, training argument, for fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                                                    See details at https://nvidia.github.io/apex/amp.html
    :param local_rank: int, training argument, for distributed training: local_rank
    :param tpu_num_cores: int, training argument, TPU: Number of TPU cores (automatically passed by launcher script)
    :param tpu_metrics_debug: bool, training argument, TPU: Whether to print debug metrics
    :param bos_token: str, the beginning of sequence token

    :return:
    """
    # Todo: try to add do_eval
    model_args = ModelArguments(model_name_or_path, config_name, cache_dir)

    train_data_file = ""
    # Todo get the train_data_file
    if type(train_data) == str:
        train_data_file = train_data
    else:
        train_data_file = output_dir + "GPT2_train_data.txt"
        df_mongoDB_train = pd.DataFrame(list(train_data.find()))
        list_mongoDB_train = df_mongoDB_train["training_data"].tolist()
        f = open(train_data_file, 'w')
        for item in list_mongoDB_train:
            f.write(item + '\n')
        f.close()

    logging.info('Passing the following training file to the trainer: %s', train_data_file)


    data_args = DataTrainingArguments(train_data_file, line_by_line, block_size, overwrite_cache)
    training_args = TrainingArguments(output_dir=output_dir, overwrite_output_dir=overwrite_output_dir,
                                      do_train=do_train,
                                      do_eval=False, do_predict=False, evaluate_during_training=False,
                                      per_gpu_train_batch_size=per_gpu_train_batch_size, per_gpu_eval_batch_size=8,
                                      gradient_accumulation_steps=gradient_accumulation_steps,
                                      learning_rate=learning_rate,
                                      weight_decay=weight_decay, adam_epsilon=adam_epsilon,
                                      max_grad_norm=max_grad_norm,
                                      num_train_epochs=num_train_epochs, max_steps=max_steps,
                                      warmup_steps=warmup_steps,
                                      logging_dir=logging_dir, logging_first_step=logging_first_step,
                                      logging_steps=logging_steps, save_steps=save_steps,
                                      save_total_limit=save_total_limit,
                                      no_cuda=no_cuda, seed=seed, fp16=fp16, fp16_opt_level=fp16_opt_level,
                                      local_rank=local_rank, tpu_num_cores=tpu_num_cores,
                                      tpu_metrics_debug=tpu_metrics_debug)

    if (os.path.exists(training_args.output_dir) and os.listdir(training_args.output_dir) and training_args.do_train
        and not training_args.overwrite_output_dir):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,)
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,)
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    else:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)

    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path, bos_token=bos_token)

    model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    # Get datasets
    train_dataset = (get_dataset(data_args, tokenizer=tokenizer, local_rank=training_args.local_rank) if training_args.do_train else None)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, mlm_probability=None)

    # Initialize our ExtendedTrainer
    trainer = ExtendedTrainer(model=model, args=training_args, data_collator=data_collator, train_dataset=train_dataset,
                              prediction_loss_only=True, )

    # Training
    if training_args.do_train:
        model_path = model_args.model_name_or_path
        # Todo the signature has changed!
        if augmentation:
            trainer.train_augmentation(tokenizer, train_data, data_base, num_val_items, val_top_k, val_top_p,
                                       data_args, training_args, model_path=model_path)
        else:
            trainer.train(tokenizer, train_data, data_base, num_val_items, val_top_k, val_top_p, model_path=model_path)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    os.remove(train_data_file)

    return None


def prompt_GPT2(train_data_file, model_dir, prompt_text, max_length, top_k, top_p, num_return_sequences, output_name):

    # Todo get rid of the train_data_file

    logging.basicConfig(level=logging.INFO)

    f = open(train_data_file, 'r+', encoding="utf-8")
    list_train_file = []
    for line in f:
        list_train_file.append(line)
    preprocessed_list_train_file = preprocess_db(list_train_file)

    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model.cuda()
    model.eval()

    # Encode a text inputs
    indexed_tokens = tokenizer.encode(prompt_text, return_tensors='pt')
    indexed_tokens = indexed_tokens.to('cuda')

    logging.info('     Generating items for the following prompt sentence: %s', prompt_text)

    # Todo change to args
    sample_outputs = model.generate(indexed_tokens,
                                    do_sample=True,
                                    max_length=max_length,
                                    top_k=top_k,
                                    top_p=top_p,
                                    num_return_sequences=num_return_sequences)

    full_decoded_outputs = []
    cropped_decoded_outputs = []
    db_comparisson = []
    multiple_generation = []
    multiple_cropped_generation = []

    # print("Output:\n" + 100 * '-')
    for i, sample_output in enumerate(sample_outputs):
        sentence = tokenizer.decode(sample_output, skip_special_tokens=True)
        logging.info('     %s', sentence)
        # Todo:
        # Step 1: keep only the item: no labels, no tokens, nothing
        temp_sentence = sentence.lower()
        if temp_sentence[-1] == ".":
            temp_sentence = temp_sentence[:-1]
        temp_sentence = re.sub(r'.*@', r'', temp_sentence)
        if temp_sentence[:2] == "i ":
            temp_sentence = temp_sentence[2:]

        # Step 2: check if it's in the db
        # Step 3: add it to the correspoding DataFrame row !!! BUT I do want to have an initial item to be able to compare the labels
        if temp_sentence in preprocessed_list_train_file:
            db_comparisson.append(list_train_file[preprocessed_list_train_file.index(temp_sentence)])
        else:
            db_comparisson.append("no")

        # Step 4: check if it's among the items generated before
        # Step 5: add it
        if sentence in full_decoded_outputs:
            multiple_generation.append(1)
        else:
            multiple_generation.append(0)

        if temp_sentence in cropped_decoded_outputs:
            multiple_cropped_generation.append(1)
        else:
            multiple_cropped_generation.append(0)

        cropped_decoded_outputs.append(temp_sentence)
        full_decoded_outputs.append(sentence)

    dict_df = {"Item": full_decoded_outputs, "Database": db_comparisson, "Fully-Repeated": multiple_generation,
               "Partially-Repeated": multiple_cropped_generation}
    df_output = pd.DataFrame.from_dict(dict_df)
    df_output.to_csv(output_name, index=False)

    return None
