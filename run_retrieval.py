import random
import torch
from methods.backbone import Bert_Encoder, Softmax_Layer, Dropout_Layer, Bert_LoRa
from dataloaders.data_loader import get_data_loader
from dataloaders.sampler import data_sampler
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import collections
from tqdm import tqdm
import logging
from config import Param
import re
import json

color_epoch = '\033[92m' 
color_loss = '\033[92m'  
color_number = '\033[93m'
color_reset = '\033[0m'


PROMPT = """
User information
----------------
Example 1: 
{{
    'context': {text_r1},
    'entity_1': {ett_r11},
    'entity_2': {ett_r12},
    'relation': {r1}
}}

Example 2: 
{{
    'context': {text_r2},
    'entity_1': {ett_r21},
    'entity_2': {ett_r22},
    'relation': {r2}
}}

Example 3: 
{{
    'context': {text_r3},
    'entity_1': {ett_r31},
    'entity_2': {ett_r32},
    'relation': {r3}
}}

Example 4: 
{{
    'context': {text_r4},
    'entity_1': {ett_r41},
    'entity_2': {ett_r42},
    'relation': {r4}
}}

----------------
Relation
----------------
[
    {r1},
    {r2},
    {r3},
    {r4}
]
----------------
You are a useful information extraction machine. Read the examples carefully and explain the sense of five relations above (note: not analysis the examples).
"""

def extract_text_between_tags(sentence):
    match_e11_e12 = re.search(r'\[E11\](.*?)\[E12\]', sentence)
    extracted_e11_e12 = match_e11_e12.group(1).strip() if match_e11_e12 else None
    
    match_e21_e22 = re.search(r'\[E21\](.*?)\[E22\]', sentence)
    extracted_e21_e22 = match_e21_e22.group(1).strip() if match_e21_e22 else None
    
    return extracted_e11_e12, extracted_e21_e22


param = Param()
args = param.args

# Device
# torch.cuda.set_device(args.gpu)
# args.device = torch.device(args.device)

# Num GPU
# args.n_gpu = torch.cuda.device_count()

# Task name
args.task_name = args.dataname

# rel_per_task
args.rel_per_task = 8 if args.dataname == "FewRel" else 4

if __name__ == '__main__':
    config = args

    config.step1_epochs = 10
    config.total_round = 1
    list_format = []
    
    
    for rou in range(config.total_round):
        test_cur = []
        test_total = []
        
        random.seed(config.seed + rou*100)
        sampler = data_sampler(config, seed=config.seed + rou*100)
        id2rel = sampler.id2rel
        rel2id = sampler.rel2id
            
        num_class = len(sampler.id2rel)

        list_relation = []
        
        for steps, (training_data, valid_data, test_data, current_relations, historic_test_data, seen_relations) in enumerate(sampler):
            
            list_relation.append(current_relations)
            
            temp_format = []
            for relation in current_relations:
                sample_first = training_data[relation][0]
                
                e11, e21 = extract_text_between_tags(sample_first['text'])
                temp_format.append({
                    'text': sample_first['text'],
                    'entity_1': e11,
                    'entity_2': e21,
                    'relation': id2rel[sample_first['relation']]
                })
                
            prompt_format = PROMPT.format_map({
                'text_r1': temp_format[0]['text'],
                'ett_r11': temp_format[0]['entity_1'],
                'ett_r12': temp_format[0]['entity_2'],
                'r1': temp_format[0]['relation'],
                
                'text_r2': temp_format[1]['text'],
                'ett_r21': temp_format[1]['entity_1'],
                'ett_r22': temp_format[1]['entity_2'],
                'r2': temp_format[1]['relation'],
                
                'text_r3': temp_format[2]['text'],
                'ett_r31': temp_format[2]['entity_1'],
                'ett_r32': temp_format[2]['entity_2'],
                'r3': temp_format[2]['relation'],
                
                'text_r4': temp_format[3]['text'],
                'ett_r41': temp_format[3]['entity_1'],
                'ett_r42': temp_format[3]['entity_2'],
                'r4': temp_format[3]['relation'],
            })
            
            list_format.append(prompt_format)
            
            json.dump(list_format, open('/home/luungoc/Thesis - 2023.2/Thesis_NgocLT/format/format.json', 'w'), ensure_ascii=False)