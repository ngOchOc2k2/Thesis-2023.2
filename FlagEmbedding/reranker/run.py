import logging
import os
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer, TrainingArguments
from transformers import (
    HfArgumentParser,
    set_seed,
)

from .arguments import ModelArguments, DataArguments
from .data import TrainDatasetForCE, GroupCollator
from .modeling import CrossEncoder
from .trainer import CETrainer

logger = logging.getLogger(__name__)


def train_retrieval(config, data_path, model_path=None):
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments


    set_seed(training_args.seed)


    data_args.train_data = config.output_kaggle + data_path
    training_args.output_dir = './model_bge'
    
    if model_path != None:
        model_args.model_name_or_path = model_path

    num_labels = 1

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    _model_class = CrossEncoder

    model = _model_class.from_pretrained(
        model_args, data_args, training_args,
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    train_dataset = TrainDatasetForCE(data_args, tokenizer=tokenizer)
    _trainer_class = CETrainer
    trainer = _trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=GroupCollator(tokenizer),
        tokenizer=tokenizer
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    trainer.train()
    trainer.save_model()

