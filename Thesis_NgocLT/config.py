import os
import argparse


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
        parser.add_argument("--data_path", default="/home/luungoc/Thesis - 2023.2/Thesis_NgocLT/datasets/", type=str)
        
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
        parser.add_argument("--description-path", default="/home/luungoc/Thesis - 2023.2/Thesis_NgocLT/datasets/description/new.json", type=str)
        parser.add_argument("--type-similar", default="colbert", type=str)
        parser.add_argument("--num-protos", default=5, type=int)
        
        
        
        # train retrieval
        parser.add_argument("--tokenizer-name", default="BAAI/bge-m3", type=str)
        parser.add_argument("--cache-dir", default=True, type=bool)
        parser.add_argument("--config-name", default=None, type=str)
        parser.add_argument("--normlized", default=True, type=bool)
        
        parser.add_argument("--sentence-pooling-method", default="cls", type=str)
        parser.add_argument("--negatives-cross-device", default=True, type=bool)
        parser.add_argument("--temperature", default=0.02, type=float)
        parser.add_argument("--use-inbatch-neg", default=True, type=bool)
        
        parser.add_argument("--fix-position-embedding", default=False, type=bool)
        parser.add_argument("--query-max-len", default=1024, type=int)
        parser.add_argument("--passage-max-len", default=256, type=int)
        parser.add_argument("--output-dir-model-retrieval", default="./model_bge", type=str)

        parser.add_argument("--seed-retrieval", default=42, type=int)
        
        parser.add_argument("--train-data", default=None, type=str)
        parser.add_argument("--learning-rate", default=1e-5, type=float)
        parser.add_argument("--fp16", default=True, type=bool)
        parser.add_argument("--num-train-epochs", default=2, type=int)

        parser.add_argument("--train-data", default=None, type=str)
        parser.add_argument("--learning-rate", default=1e-5, type=float)
        parser.add_argument("--fp16", default=True, type=bool)
        parser.add_argument("--num-train-epochs", default=2, type=int)

        parser.add_argument("--per-device-train-batch-size", default=2, type=int)
        parser.add_argument("--per-device-eval-batch-size", default=8, type=int)
        parser.add_argument("--dataloader-drop-last", default=True, type=bool)
        parser.add_argument("--train-group-size", default=2, type=int)
        
        parser.add_argument("--logging-steps", default=10, type=int)
        parser.add_argument("--query-instruction-for-retrieval", default="", type=str)
        parser.add_argument("--save-steps", default=1500, type=int)
        parser.add_argument("--save-total-limit", default=3, type=int)
        
        parser.add_argument("--tf32", default=None, type=int)
        parser.add_argument("--eval-steps", default=None, type=float)
        parser.add_argument("--do-predict", default=False, type=bool)
        parser.add_argument("--deepspeeds", default=None, type=str)
        
        parser.add_argument("--ddp-timeout", default=1800, type=int)
        parser.add_argument("--bf16", default=False, type=float)
        parser.add_argument("--adam-beta1", default=0.9, type=float)
        parser.add_argument("--adam-beta2", default=0.999, type=float)
        
        parser.add_argument("--adam-epsilon", default=1e-08, type=float)
        parser.add_argument("--adafactor", default=False, type=bool)

        return parser
