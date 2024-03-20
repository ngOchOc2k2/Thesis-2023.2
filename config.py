import os
import argparse
from typing import List

class Param:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser = self.all_param(parser)
        all_args, unknown = parser.parse_known_args()
        self.args = all_args

    def all_param(self, parser):
        # common parameters
        parser.add_argument("--gpu", default=0, type=int)
        parser.add_argument("--dataname", default="TACRED", type=str, help="Use TACRED or FewRel datasets.")
        parser.add_argument("--task_name", default="TACRED", type=str)
        parser.add_argument("--device", default="cuda", type=str)

        # training parameters
        parser.add_argument("--batch_size", default=64, type=int)
        parser.add_argument("--num_tasks", default=10)
        parser.add_argument("--rel_per_task", default=8)
        parser.add_argument("--pattern", default="entity_marker")
        parser.add_argument("--max_length", default=192, type=int)
        parser.add_argument("--encoder_output_size", default=768, type=int)
        parser.add_argument("--vocab_size", default=30522, type=int)
        parser.add_argument("--marker_size", default=4, type=int)
        parser.add_argument("--num_workers", default=0, type=int)
        parser.add_argument("--save_checkpoint", default="./checkpoint/", type=str)

        # epochs
        parser.add_argument("--classifier_epochs", default=10, type=int)

        # seed
        parser.add_argument("--seed", default=2021, type=int)
        
        # max gradient norm
        parser.add_argument("--max_grad_norm", default=10, type=float)

        # dataset path
        parser.add_argument("--data_path", default="/kaggle/input/data-relation/datasets/", type=str)
        
        # bert-base-uncased weights path
        parser.add_argument("--bert_path", default="bert-base-uncased", type=str)


        # model params
        parser.add_argument("--drop_p", type=float, default=0.1)
        parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
        parser.add_argument("--total_round", type=int, default=6)
        parser.add_argument("--drop_out", type=float, default=0.5)
        parser.add_argument("--use_gpu", default=True, type=bool)
        parser.add_argument("--hidden_size", default=768, type=int)

        # lora params
        parser.add_argument("--rank_lora", default=8, type=int)
        
        
        # retrieval configs
        parser.add_argument("--bge-model", default="BAAI/bge-m3", type=str)
        parser.add_argument("--description-path", default="/kaggle/input/train-bge/standard.json", type=str)
        parser.add_argument("--type-similar", default="colbert", type=str)
        parser.add_argument("--num-protos", default=5, type=int)

        return parser
