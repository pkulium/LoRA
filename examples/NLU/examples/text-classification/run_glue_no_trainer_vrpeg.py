# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import logging
import math
import os
import random

import datasets
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from utils.schedulers import get_policy, assign_learning_rate
import numpy as np
from utils.net_utils import constrainScoreByWhole


logger = logging.getLogger(__name__)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_mask",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args

import torch
from torch import autograd, nn
import torch.nn.functional as F
import math
class VRPGE_Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.rand_like(self.weight))
        self.register_buffer('subnet', torch.zeros_like(self.scores))
        self.train_weights = False
        # nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        score_init_constant = 0.5
        self.scores.data = (
                torch.ones_like(self.scores) * score_init_constant
        )
        self.prune = True
        self.register_buffer("stored_mask_0", torch.zeros_like(self.scores))
        self.register_buffer("stored_mask_1", torch.zeros_like(self.scores))
        self.j = 0

    @property
    def clamped_scores(self):
        return self.scores

    def fix_subnet(self):
        self.subnet = (torch.rand_like(self.scores) < self.clamped_scores).float()

    def forward(self, x):
        if self.prune:
            if not self.train_weights:
                self.subnet = StraightThroughBinomialSampleNoGrad.apply(self.scores) 
                if self.j == 0:
                    self.stored_mask_0.data = (self.subnet-self.scores)/torch.sqrt((self.scores+1e-20)*(1-self.scores+1e-20))
                else:
                    self.stored_mask_1.data = (self.subnet-self.scores)/torch.sqrt((self.scores+1e-20)*(1-self.scores+1e-20))
                w = self.weight * self.subnet
                x = F.linear(x, w, self.bias)
            else:
                w = self.weight * self.subnet
                x = F.linear(x, w, self.bias)
        else:
            x = F.linear(x, self.weight, self.bias)
        return x

class StraightThroughBinomialSampleNoGrad(autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        output = (torch.rand_like(scores) < scores).float()
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        return torch.zeros_like(grad_outputs)

def make_VRPGE_layer(layer):
    new_layer = VRPGE_Lifnear(
        in_features=layer.in_features,
        out_features=layer.out_features,
        bias=layer.bias is None,
    )
    
    new_layer.weight = nn.Parameter(layer.weight.detach())
    
    if layer.bias is not None:
        new_layer.bias = nn.Parameter(layer.bias.detach())
    
    return new_layer


def make_VRPGE_replace(model, depth=1, path="", verbose=True):
    if depth > 10:
        return
    depth += 1
        
    if isinstance(model, nn.Linear) and "attention" in path:
        if verbose:
            print(f"Find linear {path}:{key} :", type(module))

        return make_VRPGE_layer(model)
    
    for key in dir(model):
        module = getattr(model, key)
        module_type = type(module)
            
        if not isinstance(module, nn.Module) or module is model:
            continue

        if isinstance(module, nn.Linear) and "attention" in path:
            layer = make_VRPGE_layer(module)
            setattr(model, key, layer)
            if verbose:
                print(f"Find linear {path}:{key} :", type(module))
            
        elif isinstance(module, nn.ModuleList):
            for i, elem in enumerate(module):
                layer = make_VRPGE_replace(elem, depth, path+":"+key+f"[{i}]", verbose=verbose)
                if layer is not None:
                    module[i] = layer
                
        elif isinstance(module, nn.ModuleDict):
            for module_key, item in module.items():
                layer = make_VRPGE_replace(item, depth, path+":"+key+":"+module_key, verbose=verbose)
                if layer is not None:
                    module[module_key] = layer
                
        else:
            layer = make_VRPGE_replace(module, depth, path+":"+key, verbose=verbose)
            if layer is not None:
                setattr(model, key, layer)
    
def get_optimizer(args, model):
    # for n, v in model.named_parameters():
    #     if v.requires_grad:
    #         print("<DEBUG> gradient to", n)

    #     if not v.requires_grad:
    #         print("<DEBUG> no gradient to", n)

    if args.optimizer == "sgd":
        if not args.train_weights_at_the_same_time:
            parameters = list(model.named_parameters())
            bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
            rest_params = [v for n, v in parameters if ("bn" not in n) and v.requires_grad]
            optimizer = torch.optim.SGD(
                [
                    {
                        "params": bn_params,
                        "weight_decay": 0 if args.no_bn_decay else args.weight_decay,
                    },
                    {"params": rest_params, "weight_decay": args.weight_decay},
                ],
                args.learning_rate_mask,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                nesterov=args.nesterov,
            )
        else:
            parameters = list(model.named_parameters())
            for n, v in parameters:
                if ("score" not in n) and v.requires_grad:
                    print(n, "weight_para")
            for n, v in parameters:
                if ("score" in n) and v.requires_grad:
                    print(n, "score_para")
            weight_params = [v for n, v in parameters if ("score" not in n) and v.requires_grad]
            score_params = [v for n, v in parameters if ("score" in n) and v.requires_grad]
            optimizer1 = torch.optim.SGD(
                score_params, lr=0.1, weight_decay=1e-6, momentum=0.9
            )
            optimizer2 = torch.optim.SGD(
                weight_params,
                args.learning_rate,
                momentum=0.9,
                weight_decay=5e-4,
                nesterov=args.nesterov,
            )
            return optimizer1, optimizer2

    elif args.optimizer == "adam":
        if not args.train_weights_at_the_same_time:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate_mask, weight_decay=args.weight_decay
            )
        else:
            parameters = list(model.named_parameters())
            for n, v in parameters:
                if ("score" not in n) and v.requires_grad:
                    print(n, "weight_para")
            for n, v in parameters:
                if ("score" in n) and v.requires_grad:
                    print(n, "score_para")
            weight_params = [v for n, v in parameters if ("score" not in n) and v.requires_grad]
            score_params = [v for n, v in parameters if ("score" in n) and v.requires_grad]
            optimizer1 = torch.optim.Adam(
                score_params, lr=args.learning_rate_mask, weight_decay=args.weight_decay
            )

            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if "score" not in n and not any(nd in n for nd in no_decay)],
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if "score" not in n and any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            optimizer2 = torch.optim.AdamW(
                optimizer_grouped_parameters,
                args.learning_rate,
            )

            return optimizer1, optimizer2
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate_mask, weight_decay=args.weight_decay
        )
    return optimizer, None

def calculateGrad(model, fn_avg, fn_list, args):
    for n, m in model.named_modules():
        if hasattr(m, "scores") and m.prune:
            m.scores.grad.data += 1/(args.K-1)*(fn_list[0] - fn_avg)*getattr(m, 'stored_mask_0') + 1/(args.K-1)*(fn_list[1] - fn_avg)*getattr(m, 'stored_mask_1')


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", args.task_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.valid_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )
    make_VRPGE_replace(model, verbose=True)

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.

    # no_decay = ["bias", "LayerNorm.weight"]
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in model.named_parameters() if "scores" not in n and not any(nd in n for nd in no_decay)],
    #         "weight_decay": args.weight_decay,
    #     },
    #     {
    #         "params": [p for n, p in model.named_parameters() if "scores" not in n and any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.0,
    #     },
        
    # ]
    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    args.optimizer = 'adam'
    args.learning_rate_mask = 12e-3
    args.K = 20
    args.train_weights_at_the_same_time = True
    args.nesterov = False
    optimizer, weight_opt = get_optimizer(args, model)

    # Prepare everything with our `accelerator`.
    model, weight_opt, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, weight_opt, optimizer, train_dataloader, eval_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab h`is length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Get the metric function
    if args.task_name is not None:
        metric = load_metric("glue", args.task_name)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    for epoch in range(args.num_train_epochs):
        model.train()
        assign_learning_rate(weight_opt, 0.5 * (1 + np.cos(np.pi * epoch / args.num_train_epochs)) * args.learning_rate)
        assign_learning_rate(optimizer, 0.5 * (1 + np.cos(np.pi * epoch / args.num_train_epochs)) * args.learning_rate_mask)
        for step, batch in enumerate(train_dataloader):
            fn_list = []
            l = 0
            if optimizer is not None:
                optimizer.zero_grad()
            if weight_opt is not None:
                weight_opt.zero_grad()
            for j in range(args.K):
                args.j = j
                outputs = model(**batch)
                original_loss = outputs.loss
                loss = original_loss/args.K
                fn_list.append(loss.item()*args.K)
                accelerator.backward(loss, retain_graph=True)
                l = l + loss.item()
            fn_avg = l
            calculateGrad(model, fn_avg, fn_list, args)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            if optimizer is not None:
                optimizer.step()
            if weight_opt is not None:
                weight_opt.step()
            with torch.no_grad():
                constrainScoreByWhole(model, None, None)

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        logger.info(f"epoch {epoch}: {eval_metric}")

    model.eval()
    for step, batch in enumerate(eval_dataloader):
        outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        metric.add_batch(
            predictions=accelerator.gather(predictions),
            references=accelerator.gather(batch["labels"]),
        )

    eval_metric = metric.compute()
    logger.info(f"epoch {epoch}: {eval_metric}")

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)

    if args.task_name == "mnli":
        # Final evaluation on mismatched validation set
        eval_dataset = processed_datasets["validation_mismatched"]
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
        eval_dataloader = accelerator.prepare(eval_dataloader)

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        logger.info(f"mnli-mm: {eval_metric}")


if __name__ == "__main__":
    main()
