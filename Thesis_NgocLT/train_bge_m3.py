import json
import random
import os
import re
import torch
from sklearn.cluster import KMeans
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
    with open(file_path, "w") as jsonl_file:
        for item in data:
            json.dump(item, jsonl_file)
            jsonl_file.write("\n")
    print("Dữ liệu đã được lưu vào file JSONL:", file_path)
    

def get_proto(config, encoder, relation_dataset):
    data_loader = get_data_loader(config, relation_dataset, shuffle=False, drop_last=False, batch_size=1)
    features = []
    encoder.eval()

    for step, batch_data in enumerate(data_loader):
        labels, tokens = batch_data
        tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
        with torch.no_grad():
            feature = encoder(tokens)
        features.append(feature)
        
    features = torch.cat(features, dim=0)
    proto = torch.mean(features, dim=0, keepdim=True).cpu()
    standard = torch.sqrt(torch.var(features, dim=0)).cpu()
    return proto, standard



def select_data(config, encoder, relation_dataset):
    data_loader = get_data_loader(config, relation_dataset, shuffle=False, drop_last=False, batch_size=1)
    features = []
    encoder.eval()
    
    for step, batch_data in enumerate(data_loader):
        labels, tokens = batch_data
        tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
        with torch.no_grad():
            feature = encoder(tokens).cpu()
        features.append(feature)

    features = np.concatenate(features)
    num_clusters = min(config.num_protos, len(relation_dataset))
    distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features)

    memory = []
    for k in range(num_clusters):
        sel_index = np.argmin(distances[:, k])
        instance = relation_dataset[sel_index]
        memory.append(instance)
    return memory

    

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
        # bge_m3 = BGEM3FlagModel(config.bge_model, use_fp16=True)
        # description_relation = json.load(open(config.description_path, 'r'))
        
        # id2name = [item['relation'] for item in description_relation]
        
        # if config.dataname == 'TACRED':
        #     convert2name = json.load(open(config.data_path + '/id2rel_tacred.json', 'r'))
        # else:
        #     convert2name = json.load(open(config.data_path + '/id2rel.json', 'r'))
            
    
        # text_description = [item['text'] for item in description_relation]
        
        # accuracy_retrieval, total_retrieval = {}, {}

        # task_accuracy = []
        # accuracy_retrieval, total_retrieval = {}, {}
        # for relation in id2name:
        #     accuracy_retrieval[relation] = 0
        #     total_retrieval[relation] = 0
            
        
        train_retrieval = []
        
        train, val, test = [], [], []
        for steps, (training_data, valid_data, test_data, current_relations, historic_test_data, seen_relations) in enumerate(tqdm(sampler)):
            
            train.append({
                'task': steps + 1,
                'relations': current_relations,
                'data': training_data,
            })
            
            val.append({
                'task': steps + 1,
                'relations': current_relations,
                'data': valid_data,
            })
            
            test.append({
                'task': steps + 1,
                'relations': current_relations,
                'data': test_data,
            })
            
            
            # for relation in current_relations:
            #     description = ''
                
                
            #     for item in description_relation:
            #         if item['relation'] == relation:
            #             description = item['text']
                
                
            #     for sample in training_data[relation]:
            #         neg_list = []
                    
            #         for rela in current_relations:
            #             if rela != relation:
            #                 for _ in range(2):
            #                     x = random.randint(0, len(training_data[rela]) - 2)
            #                     neg_list.append(training_data[rela][x]['text'])
                    
            #         train_retrieval.append({
            #             'query': description,
            #             'pos': [sample['text']],
            #             'neg': neg_list
            #         })
                    
        # random.shuffle(train_retrieval)
        # save_to_jsonl(train_retrieval, './with_neg.jsonl')

        json.dump(train, open('/home/luungoc/Thesis - 2023.2/Thesis_NgocLT/hoang/train.json', 'w'), ensure_ascii=False)

        json.dump(val, open('/home/luungoc/Thesis - 2023.2/Thesis_NgocLT/hoang/val.json', 'w'), ensure_ascii=False)

        json.dump(test, open('/home/luungoc/Thesis - 2023.2/Thesis_NgocLT/hoang/test.json', 'w'), ensure_ascii=False)
