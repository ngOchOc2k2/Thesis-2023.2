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


color_epoch = '\033[92m' 
color_loss = '\033[92m'  
color_number = '\033[93m'
color_reset = '\033[0m'



PROMPT_TASK_TACRED = """The task involves relation extraction for two entities within a given sentence. 
There are five classes: {re1}, {re2}, {re3}, {re4}, each representing different types of relationships that can exist between the two entities. 
The goal is to classify the relationship between the entities into one of these classes based on the context provided by the sentence
{description_class}
Example: {example}
"""



def extract_text_between_tags(sentence):
    match_e11_e12 = re.search(r'\[E11\](.*?)\[E12\]', sentence)
    extracted_e11_e12 = match_e11_e12.group(1).strip() if match_e11_e12 else None
    
    match_e21_e22 = re.search(r'\[E21\](.*?)\[E22\]', sentence)
    extracted_e21_e22 = match_e21_e22.group(1).strip() if match_e21_e22 else None
    
    return extracted_e11_e12, extracted_e21_e22



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
        bge_m3 = BGEM3FlagModel(config.bge_model, use_fp16=True)
        description_relation = json.load(open(config.description_path, 'r'))
        id2name = [item['relation'] for item in description_relation]
        
        if config.dataname == 'TACRED':
            convert2name = json.load(open(config.data_path + '/id2rel_tacred.json', 'r'))
        else:
            convert2name = json.load(open(config.data_path + '/id2rel.json', 'r'))
            
            
        text_description = [item['text'] for item in description_relation]
        
        accuracy_retrieval, total_retrieval = {}, {}
        dict_des = {}
        for sample in description_relation:
            dict_des[sample['relation']] = sample['text']
            

        # embedding description
        # embedding_description = bge_m3.encode(text_description, return_dense=True, return_colbert_vecs=True)
        task_accuracy = []
        accuracy_retrieval, total_retrieval = {}, {}
        for relation in id2name:
            accuracy_retrieval[relation] = 0
            total_retrieval[relation] = 0
        
        store_data = []
        
        print(f"\n" + "---"*25 + "Step 1: Prepare Data!" + "---"*25)
        for steps, (training_data, valid_data, test_data, current_relations, historic_test_data, seen_relations) in enumerate(tqdm(sampler)):
            
            print(f"Task {steps + 1}: {current_relations}")
            
            test_data_combine = []
            for relation in current_relations:
                
                for sample in test_data[relation]:
                    sample['task'] = steps
                    sample['task_relation'] = current_relations
                    
                for sample in training_data[relation]:
                    sample['task'] = steps
                    sample['task_relation'] = current_relations
                    
                store_data += training_data[relation]
                
        store_text, store_task = [], []
        
        for sample in store_data:
            store_text += PROMPT_TASK_TACRED.format_map({
                're1': sample['task_relation'][0],
                're2': sample['task_relation'][1],
                're3': sample['task_relation'][2],
                're4': sample['task_relation'][3],
                
                'description_class': dict_des[convert2name[sample['relation']]],
                'example': sample['text']
            })
                
            store_task += sample['task']
            
        
        embedding_store = bge_m3.encode(store_text, return_dense=True, return_colbert_vecs=True)
                
                
        print(f"\n" + "---"*25 + "Step 2: Eval Data!" + "---"*25)
        for steps, (training_data, valid_data, test_data, current_relations, historic_test_data, seen_relations) in enumerate(tqdm(sampler)):
            print(f"Length test: {len(test_data_combine)}")
            
            for sample in test_data_combine:
                text = sample['text']
                embedding_text = bge_m3.encode(text, max_length=512, return_dense=True, return_colbert_vecs=True)
                
                # consider similarity function
                if config.type_similar == 'dense':
                    similarity = embedding_text['dense_vecs'] @ embedding_store['dense_vecs'].T
                    total_retrieval[convert2name[sample['relation']]] += 1
                    
                    index = np.argmax(similarity)

                    predict_relation = id2name[index]
                    if convert2name[sample['relation']] == predict_relation:
                        accuracy_retrieval[predict_relation] += 1

                        

                elif config.type_similar == 'colbert':
                    result = []
                    
                    for idx in range(len(embedding_store['colbert_vecs'])):
                        a = float(bge_m3.colbert_score(embedding_text['colbert_vecs'], embedding_store['colbert_vecs'][idx]))
                        result.append(a)
                        
                    total_retrieval[convert2name[sample['relation']]] += 1
                    index = result.index(max(result))
                    predict_relation = id2name[index]
                        
                    if convert2name[sample['relation']] == predict_relation:
                        accuracy_retrieval[predict_relation] += 1

            
            acc_task = 0
            for relation in current_relations:
                acc_task += accuracy_retrieval[relation]


            print(f"Acc task: {acc_task} / {len(test_data_combine)} = {acc_task / len(test_data_combine)}")
            task_accuracy.append(acc_task / len(test_data_combine))
            
                            
                            
        for keys, value in enumerate(accuracy_retrieval):
            accuracy_retrieval[value] = accuracy_retrieval[value] / total_retrieval[value] if total_retrieval[value] != 0 else 0.0
            print(f"{value}: {accuracy_retrieval[value]}")

        json.dump(accuracy_retrieval, open('./log_output.json', 'w'), ensure_ascii=False)
        print(f"Accuracy tasks: {task_accuracy}")
        
        