import logging
import os
from pathlib import Path
import torch.distributed as dist

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)
from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl
)

from .arguments import ModelArguments, DataArguments, \
    RetrieverTrainingArguments as TrainingArguments
from .data import SameDatasetTrainDataset, EmbedCollator
from .modeling import BGEM3Model
from .trainer import BiTrainer


logger = logging.getLogger(__name__)


class TrainerCallbackForDataRefresh(TrainerCallback):
    def __init__(self, train_dataset):
        self.train_dataset = train_dataset
        
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            self.train_dataset.refresh_epoch()
        

def train_retrieval(config, data_path, model_path=None):
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments


    # Set seed
    set_seed(training_args.seed)


    data_args.train_data = config.output_kaggle + data_path
    
    if model_path != None:
        model_args.model_name_or_path = model_path



    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    model = BGEM3Model(model_name=model_args.model_name_or_path,
                       normlized=training_args.normlized,
                       sentence_pooling_method=training_args.sentence_pooling_method,
                       negatives_cross_device=training_args.negatives_cross_device,
                       temperature=training_args.temperature,
                       enable_sub_batch=training_args.enable_sub_batch,
                       unified_finetuning=training_args.unified_finetuning,
                       use_self_distill=training_args.use_self_distill,
                       colbert_dim=training_args.colbert_dim,
                       self_distill_start_step=training_args.self_distill_start_step)



    if training_args.fix_position_embedding:
        for k, v in model.named_parameters():
            if "position_embeddings" in k:
                logging.info(f"Freeze the parameters for {k}")
                v.requires_grad = False
    if training_args.fix_encoder:
        for k, v in model.named_parameters():
            if "colbert_linear" in k or 'sparse_linear' in k:
                logging.info(f"train the parameters for {k}")
            else:
                v.requires_grad = False

    # print(f"===========================Rank {dist.get_rank()}: start loading data===========================")
    if data_args.same_task_within_batch:
        train_dataset = SameDatasetTrainDataset(args=data_args, 
                                                batch_size=training_args.per_device_train_batch_size, 
                                                seed=training_args.seed, 
                                                num_processes=training_args.world_size,
                                                process_index=training_args.process_index)
        training_args.per_device_train_batch_size = 1
        training_args.dataloader_num_workers = 0    # avoid multi-processes
    else:
        raise NotImplementedError("Not support `same_task_within_batch=False`")

    data_collator = EmbedCollator(
        tokenizer,
        query_max_len=data_args.query_max_len,
        passage_max_len=data_args.passage_max_len
    )
    
    trainer = BiTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    if data_args.same_task_within_batch:
        trainer.add_callback(TrainerCallbackForDataRefresh(train_dataset))
    
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    # print(f"===========================Rank {dist.get_rank()}: start training===========================")
    trainer.train()
    trainer.save_model()
    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)

