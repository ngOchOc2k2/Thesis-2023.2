import logging
import os
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

from .arguments import ModelArguments, DataArguments, \
    RetrieverTrainingArguments as TrainingArguments
from .data import TrainDatasetForEmbedding, EmbedCollator, TrainDatasetForEmbeddingCustom, EmbedCollatorCustom
from .modeling import BiEncoderModel, DistilationModel
from .modeling_custom import BiEncoderModelCustom, DistilationModelCustom
from .trainer import BiTrainer
logger = logging.getLogger(__name__)



def train_retrieval_custom(config, data_path, model_path=None, epochs=1, output_dir=None):
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


    training_args.epochs = epochs
        
        
    if output_dir != None:
        training_args.output_dir = output_dir
    
    tokenizer = AutoTokenizer.from_pretrained(
        'BAAI/bge-m3',
        # additional_special_tokens=["[E11]", "[E12]", "[E21]", "[E22]"],
        use_fast=False,
    )



    model = BiEncoderModelCustom(
        model_name=model_args.model_name_or_path,
        normlized=training_args.normlized,
        sentence_pooling_method=training_args.sentence_pooling_method,
        negatives_cross_device=training_args.negatives_cross_device,
        temperature=training_args.temperature,
        use_inbatch_neg=training_args.use_inbatch_neg,
    )

    if training_args.fix_position_embedding:
        for k, v in model.named_parameters():
            if "position_embeddings" in k:
                logging.info(f"Freeze the parameters for {k}")
                v.requires_grad = False


    train_dataset = TrainDatasetForEmbedding(args=data_args, tokenizer=tokenizer)

    trainer = BiTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=EmbedCollator(
            tokenizer,
            query_max_len=data_args.query_max_len,
            passage_max_len=data_args.passage_max_len
        ),
        tokenizer=tokenizer
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    trainer.train()
    trainer.save_model()

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)
        
        
def train_retrieval_distil_custom(config, data_path, model_teacher = None, epochs=None, model_path=None, output_dir=None):
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

    if epochs != None:
        training_args.num_train_epochs = epochs

    if output_dir != None:
        training_args.output_dir = output_dir
        
    tokenizer = AutoTokenizer.from_pretrained(
        'BAAI/bge-m3',
        # additional_special_tokens=["[E11]", "[E12]", "[E21]", "[E22]"],
        use_fast=False,
    )

    
    try:
        model = DistilationModelCustom(
            model_name=model_args.model_name_or_path,
            model_teacher=model_teacher,
            normlized=training_args.normlized,
            sentence_pooling_method=training_args.sentence_pooling_method,
            negatives_cross_device=training_args.negatives_cross_device,
            temperature=training_args.temperature,
            use_inbatch_neg=training_args.use_inbatch_neg,
        )
    except:
        model = BiEncoderModelCustom(
            model_name='BAAI/bge-m3',
            normlized=training_args.normlized,
            sentence_pooling_method=training_args.sentence_pooling_method,
            negatives_cross_device=training_args.negatives_cross_device,
            temperature=training_args.temperature,
            use_inbatch_neg=training_args.use_inbatch_neg,
        )    


    if training_args.fix_position_embedding:
        for k, v in model.named_parameters():
            if "position_embeddings" in k:
                logging.info(f"Freeze the parameters for {k}")
                v.requires_grad = False


    train_dataset = TrainDatasetForEmbedding(args=data_args, tokenizer=tokenizer)

    trainer = BiTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=EmbedCollator(
            tokenizer,
            query_max_len=data_args.query_max_len,
            passage_max_len=data_args.passage_max_len
        ),
        tokenizer=tokenizer
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    trainer.train()
    trainer.save_model()

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)
        