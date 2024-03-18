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
        bge_m3 = BGEM3FlagModel(config.bge_model, use_fp16=True)
        description_relation = json.load(open(config.description_path, 'r'))
        id2name = [item['relation'] for item in description_relation]
        
        if config.dataname == 'FewRel':
            convert2name = json.load(open('./datasets/id2rel_tacred.json', 'r'))
        else:
            convert2name = json.load(open('./datasets/id2rel.json', 'r'))
            
            
        text_description = [item['text'] for item in description_relation]
        
        accuracy_retrieval, total_retrieval = {}, {}

        # embedding description
        embedding_description = bge_m3.encode(text_description, return_dense=True, return_colbert_vecs=True)
        
        for steps, (training_data, valid_data, test_data, current_relations, historic_test_data, seen_relations) in enumerate(tqdm(sampler)):
            
            test_data_combine = []
            for relation in current_relations:
                sample_first = training_data[relation][0]
                accuracy_retrieval[relation] = 0
                total_retrieval[relation] = 0
                
            
                # get total test data
                test_data_combine += test_data[current_relations]
                
            
            for sample in test_data_combine:
                text = sample['text']
                embedding_text = bge_m3.encode(text, max_length=512, return_dense=True, return_colbert_vecs=True)
                
                
                # consider similarity function
                if config.typy_similar == 'dense':
                    similarity = embedding_text @ embedding_description['dense'].T
                    total_retrieval[convert2name[sample['relation']]] += 1
                    
                    index = np.argmax(similarity)

                    predict_relation = id2name[index]
                    if sample['relation'] == predict_relation:
                        accuracy_retrieval[predict_relation] += 1
                        

                elif config.typy_similar == 'colbert':
                    result = []
                    
                    for idx in range(len(embedding_description['colbert_vecs'])):
                        a = float(bge_m3.colbert_score(embedding_text['colbert_vecs'], embedding_description['colbert_vecs'][idx]))
                        result.append(a)
                        
                        
                        result = np.array(result)
                        total_retrieval[convert2name[sample['relation']]] += 1
                        index = np.argmax(result)
                        predict_relation = id2name[index]
                        
                        if sample['relation'] == predict_relation:
                            accuracy_retrieval[predict_relation] += 1
                            
                            
        for keys, value in enumerate(accuracy_retrieval):
            print(f"{value}: {accuracy_retrieval[value] / total_retrieval[value]}" if total_retrieval[value] != 0 else f"{value}: 0.00")
