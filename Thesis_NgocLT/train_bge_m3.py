import json
import random
import os
import re
from dataloaders.data_loader import get_data_loader
from dataloaders.sampler import data_sampler
from FlagEmbedding import BGEM3FlagModel
from config import Param
import numpy as np
from tqdm import tqdm
import random

color_epoch = '\033[92m' 
color_loss = '\033[92m'  
color_number = '\033[93m'
color_reset = '\033[0m'



def extract_text_between_tags(sentence):
    match_e11_e12 = re.search(r'\[E11\](.*?)\[E12\]', sentence)
    extracted_e11_e12 = match_e11_e12.group(1).strip() if match_e11_e12 else None
    
    match_e21_e22 = re.search(r'\[E21\](.*?)\[E22\]', sentence)
    extracted_e21_e22 = match_e21_e22.group(1).strip() if match_e21_e22 else None
    
    return extracted_e11_e12, extracted_e21_e22


def save_to_jsonl(data, file_path):
    """
    Lưu dữ liệu vào một file JSONL.

    Parameters:
        data (list): Danh sách các đối tượng để lưu vào file JSONL.
        file_path (str): Đường dẫn tới file JSONL.

    Returns:
        None
    """
    with open(file_path, "w") as jsonl_file:
        for item in data:
            json.dump(item, jsonl_file)  # Ghi một đối tượng JSON vào file
            jsonl_file.write("\n")  # Viết dấu xuống dòng sau mỗi đối tượng

    print("Dữ liệu đã được lưu vào file JSONL:", file_path)
    

    

param = Param()
args = param.args

# Task name
args.task_name = args.dataname

# rel_per_task
args.rel_per_task = 8 if args.dataname == "FewRel" else 4

if __name__ == '__main__':
    config = args

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
        print("---"*30 + 'Loading Model!' + '---'*30)
        description_relation = json.load(open(config.description_path, 'r'))
        id2name = [item['relation'] for item in description_relation]
        
        if config.dataname == 'TACRED':
            convert2name = json.load(open(config.data_path + '/id2rel_tacred.json', 'r'))
        else:
            convert2name = json.load(open(config.data_path + '/id2rel.json', 'r'))
            
            
        text_description = [item['text'] for item in description_relation]
        
        accuracy_retrieval, total_retrieval = {}, {}

        task_accuracy = []
        accuracy_retrieval, total_retrieval = {}, {}
        for relation in id2name:
            accuracy_retrieval[relation] = 0
            total_retrieval[relation] = 0
            
        
        train_retrieval = []
        for steps, (training_data, valid_data, test_data, current_relations, historic_test_data, seen_relations) in enumerate(tqdm(sampler)):
            
            for relation in current_relations:
                description = ''
                
                
                for item in description_relation:
                    if item['relation'] == relation:
                        description = item['text']
                
                
                for sample in training_data[relation]:
                    neg_list = []
                    
                    for rela in current_relations:
                        if rela != relation:
                            x = random.randint(0, len(training_data[rela]) - 2)
                            neg_list.append(training_data[rela][x]['text'])
                    
                    train_retrieval.append({
                        'query': description,
                        'pos': [sample['text']],
                        'neg': neg_list
                    })
                    
        save_to_jsonl(train_retrieval, './with_neg.jsonl')
