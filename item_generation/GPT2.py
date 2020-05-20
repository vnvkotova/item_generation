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
from transformers.training_args import TrainingArguments, is_tpu_available

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

try:
    from apex import amp

    _has_apex = True
except ImportError:
    _has_apex = False


def is_apex_available():
    return _has_apex


if is_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

try:
    from torch.utils.tensorboard import SummaryWriter

    _has_tensorboard = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter

        _has_tensorboard = True
    except ImportError:
        _has_tensorboard = False


class ExtendedTrainer(Trainer):
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch,
    optimized for Transformers.
    """

    model: PreTrainedModel
    args: TrainingArguments
    data_collator: DataCollator
    train_dataset: Optional[Dataset]
    eval_dataset: Optional[Dataset]
    compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None
    prediction_loss_only: bool
    tb_writer: Optional["SummaryWriter"] = None
    optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None
    global_step: Optional[int] = None
    epoch: Optional[float] = None

    # def __init__(
    #     self,
    #     model: PreTrainedModel,
    #     args: TrainingArguments,
    #     data_collator: Optional[DataCollator] = None,
    #     train_dataset: Optional[Dataset] = None,
    #     eval_dataset: Optional[Dataset] = None,
    #     compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
    #     prediction_loss_only=False,
    #     tb_writer: Optional["SummaryWriter"] = None,
    #     optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None,
    # ):
    #     """
    #     Trainer is a simple but feature-complete training and eval loop for PyTorch,
    #     optimized for Transformers.
    #     Args:
    #         prediction_loss_only:
    #             (Optional) in evaluation and prediction, only return the loss
    #     """
    #     super().__init__(model, args, data_collator, train_dataset, eval_dataset, compute_metrics,
    #                      prediction_loss_only, tb_writer, optimizers)
    #
    # def get_train_dataloader(self):
    #     super().get_train_dataloader()
    #
    # def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None):
    #     super().get_eval_dataloader(eval_dataset)
    #
    # def get_test_dataloader(self, test_dataset: Dataset):
    #     super().get_test_dataloader(test_dataset)
    #
    # def get_optimizers(self, num_training_steps: int):
    #     """
    #     Setup the optimizer and the learning rate scheduler.
    #     We provide a reasonable default that works well.
    #     If you want to use something else, you can pass a tuple in the Trainer's init,
    #     or override this method in a subclass.
    #     """
    #     super().get_optimizers(num_training_steps)
    #
    # def _setup_wandb(self):
    #     """
    #     Setup the optional Weights & Biases (`wandb`) integration.
    #     One can override this method to customize the setup if needed.  Find more information at https://docs.wandb.com/huggingface
    #     You can also override the following environment variables:
    #     Environment:
    #         WANDB_WATCH:
    #             (Optional, ["gradients", "all", "false"]) "gradients" by default, set to "false" to disable gradient logging
    #             or "all" to log gradients and parameters
    #         WANDB_PROJECT:
    #             (Optional): str - "huggingface" by default, set this to a custom string to store results in a different project
    #         WANDB_DISABLED:
    #             (Optional): boolean - defaults to false, set to "true" to disable wandb entirely
    #     """
    #     super()._setup_wandb()
    #
    # def num_examples(self, dataloader):
    #     """
    #     Helper to get num of examples from a DataLoader, by accessing its Dataset.
    #     """
    #     super().num_examples(dataloader)

    def train(self, model_path: Optional[str] = None):
        """
        Main training entry point.
        Args:
            model_path:
                (Optional) Local path to model if model to train has been instantiated from a local path
                If present, we will try reloading the optimizer/scheduler states from there.
        """
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
        if is_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
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
        # plt.ion()

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

                    if is_tpu_available():
                        xm.optimizer_step(optimizer)
                    else:
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

                            list_losses.append(logs["loss"])
                            plt.plot(list_losses, label='current loss')
                            plt.legend()
                            plt.show();
                            # plt.plot(list_losses, label='current loss')
                            # plt.draw()
                            # plt.clf()


                            # generating items to see how the model is doing

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
            if 0 < self.args.max_steps < self.global_step:
                train_iterator.close()
                break
            if self.args.tpu_metrics_debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

        if self.tb_writer:
            self.tb_writer.close()

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss / self.global_step)

    # def _log(self, logs: Dict[str, float], iterator: Optional[tqdm] = None) -> None:
    #     super()._log(logs, iterator)
    #
    # def _training_step(self, model: nn.Module, inputs: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer):
    #     super()._training_step(model, inputs, optimizer)
    #
    # def is_local_master(self):
    #     super().is_local_master()
    #
    # def is_world_master(self):
    #     """
    #     This will be True only in one process, even in distributed mode,
    #     even when training on multiple machines.
    #     """
    #     super().is_world_master()
    #
    # def save_model(self, output_dir: Optional[str] = None):
    #     """
    #     Saving best-practices: if you use default names for the model,
    #     you can reload it using from_pretrained().
    #     Will only save from the master process.
    #     """
    #     super().save_model(output_dir)
    #
    # def _save_tpu(self, output_dir: Optional[str] = None):
    #     super().save_tpu(output_dir)
    #
    # def _save(self, output_dir: Optional[str] = None):
    #     super()._save()
    #
    # def _sorted_checkpoints(self, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False):
    #     super()._sorted_checkpoints(checkpoint_prefix, use_mtime)
    #
    # def _rotate_checkpoints(self, use_mtime=False):
    #     super()._rotate_checkpoints(use_mtime)
    #
    # def evaluate(self, eval_dataset: Optional[Dataset] = None, prediction_loss_only: Optional[bool] = None,):
    #     """
    #     Run evaluation and return metrics.
    #     The calling script will be responsible for providing a method to compute metrics, as they are
    #     task-dependent.
    #     Args:
    #         eval_dataset: (Optional) Pass a dataset if you wish to override
    #         the one on the instance.
    #     Returns:
    #         A dict containing:
    #             - the eval loss
    #             - the potential metrics computed from the predictions
    #     """
    #     super().evaluate(eval_dataset, prediction_loss_only)
    #
    # def predict(self, test_dataset: Dataset):
    #     """
    #     Run prediction and return predictions and potential metrics.
    #     Depending on the dataset and your use case, your test dataset may contain labels.
    #     In that case, this method will also return metrics, like in evaluate().
    #     """
    #     super().predict(test_dataset)
    #
    # def _prediction_loop(self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None):
    #     """
    #     Prediction/evaluation loop, shared by `evaluate()` and `predict()`.
    #     Works both with or without labels.
    #     """
    #     super()._prediction_loop(dataloader, description, prediction_loss_only)

# @dataclass
# class ModelArguments:
#     """
#     Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
#     """
#
#     model_name_or_path: Optional[str] = field(
#         default=None,
#         metadata={
#             "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
#         },
#     )
#     model_type: Optional[str] = field(
#         default=None,
#         metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
#     )
#     config_name: Optional[str] = field(
#         default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
#     )
#     tokenizer_name: Optional[str] = field(
#         default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
#     )
#     cache_dir: Optional[str] = field(
#         default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
#     )
#
#
# @dataclass
# class DataTrainingArguments:
#     """
#     Arguments pertaining to what data we are going to input our model for training and eval.
#     """
#
#     train_data_file: Optional[str] = field(
#         default=None, metadata={"help": "The input training data file (a text file)."}
#     )
#     eval_data_file: Optional[str] = field(
#         default=None,
#         metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
#     )
#     line_by_line: bool = field(
#         default=False,
#         metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
#     )
#
#     mlm: bool = field(
#         default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
#     )
#     mlm_probability: float = field(
#         default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
#     )
#
#     block_size: int = field(
#         default=-1,
#         metadata={
#             "help": "Optional input sequence length after tokenization."
#             "The training dataset will be truncated in block of this size for training."
#             "Default to the model max input length for single sentence inputs (take into account special tokens)."
#         },
#     )
#     overwrite_cache: bool = field(
#         default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
#     )
#
#
# def get_dataset(args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, evaluate=False, local_rank=-1):
#     file_path = args.eval_data_file if evaluate else args.train_data_file
#     if args.line_by_line:
#         return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)
#     else:
#         return TextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)
#
#
# def main():
#     # See all possible arguments in src/transformers/training_args.py
#     # or by passing the --help flag to this script.
#     # We now keep distinct sets of args, for a cleaner separation of concerns.
#
#     parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
#     model_args, data_args, training_args = parser.parse_args_into_dataclasses()
#
#     if data_args.eval_data_file is None and training_args.do_eval:
#         raise ValueError(
#             "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
#             "or remove the --do_eval argument."
#         )
#
#     if (
#         os.path.exists(training_args.output_dir)
#         and os.listdir(training_args.output_dir)
#         and training_args.do_train
#         and not training_args.overwrite_output_dir
#     ):
#         raise ValueError(
#             f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
#         )
#
#     # Setup logging
#     logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
#         datefmt="%m/%d/%Y %H:%M:%S",
#         level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
#     )
#     logger.warning(
#         "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
#         training_args.local_rank,
#         training_args.device,
#         training_args.n_gpu,
#         bool(training_args.local_rank != -1),
#         training_args.fp16,
#     )
#     logger.info("Training/evaluation parameters %s", training_args)
#
#     # Set seed
#     set_seed(training_args.seed)
#
#     # Load pretrained model and tokenizer
#     #
#     # Distributed training:
#     # The .from_pretrained methods guarantee that only one local process can concurrently
#     # download model & vocab.
#
#     if model_args.config_name:
#         config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
#     elif model_args.model_name_or_path:
#         config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
#     else:
#         config = CONFIG_MAPPING[model_args.model_type]()
#         logger.warning("You are instantiating a new config instance from scratch.")
#
#     if model_args.tokenizer_name:
#         tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
#     elif model_args.model_name_or_path:
#         tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
#     else:
#         raise ValueError(
#             "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
#             "and load it from here, using --tokenizer_name"
#         )
#
#     if model_args.model_name_or_path:
#         model = AutoModelWithLMHead.from_pretrained(
#             model_args.model_name_or_path,
#             from_tf=bool(".ckpt" in model_args.model_name_or_path),
#             config=config,
#             cache_dir=model_args.cache_dir,
#         )
#     else:
#         logger.info("Training new model from scratch")
#         model = AutoModelWithLMHead.from_config(config)
#
#     model.resize_token_embeddings(len(tokenizer))
#
#     if config.model_type in ["bert", "roberta", "distilbert", "camembert"] and not data_args.mlm:
#         raise ValueError(
#             "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
#             "flag (masked language modeling)."
#         )
#
#     if data_args.block_size <= 0:
#         data_args.block_size = tokenizer.max_len
#         # Our input block size will be the max possible for the model
#     else:
#         data_args.block_size = min(data_args.block_size, tokenizer.max_len)
#
#     # Get datasets
#     train_dataset = (
#         get_dataset(data_args, tokenizer=tokenizer, local_rank=training_args.local_rank)
#         if training_args.do_train
#         else None
#     )
#     eval_dataset = (
#         get_dataset(data_args, tokenizer=tokenizer, local_rank=training_args.local_rank, evaluate=True)
#         if training_args.do_eval
#         else None
#     )
#     data_collator = DataCollatorForLanguageModeling(
#         tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
#     )
#
#     # Initialize our Trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         data_collator=data_collator,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         prediction_loss_only=True,
#     )
#
#     # Training
#     if training_args.do_train:
#         model_path = (
#             model_args.model_name_or_path
#             if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
#             else None
#         )
#         trainer.train(model_path=model_path)
#         trainer.save_model()
#         # For convenience, we also re-save the tokenizer to the same directory,
#         # so that you can share your model easily on huggingface.co/models =)
#         if trainer.is_world_master():
#             tokenizer.save_pretrained(training_args.output_dir)
#
#     # Evaluation
#     results = {}
#     if training_args.do_eval and training_args.local_rank in [-1, 0]:
#         logger.info("*** Evaluate ***")
#
#         eval_output = trainer.evaluate()
#
#         perplexity = math.exp(eval_output["eval_loss"])
#         result = {"perplexity": perplexity}
#
#         output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
#         with open(output_eval_file, "w") as writer:
#             logger.info("***** Eval results *****")
#             for key in sorted(result.keys()):
#                 logger.info("  %s = %s", key, str(result[key]))
#                 writer.write("%s = %s\n" % (key, str(result[key])))
#
#         results.update(result)
#
#     return results


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


def train_GPT2(model_name_or_path, train_data_file, output_dir, config_name=None, cache_dir=None, line_by_line=True,
               block_size=-1,
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
    :param train_data_file: str, data argument, the input training data file (a text file)
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
    # tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)

    # model = AutoModelWithLMHead.from_pretrained(
    #         model_args.model_name_or_path,
    #         from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #         config=config,
    #         cache_dir=model_args.cache_dir,)
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
        trainer.train(model_path=model_path)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    return None
