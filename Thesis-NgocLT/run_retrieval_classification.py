import random
import torch
from methods.backbone import Bert_LoRa, Classifier_Layer
from dataloaders.data_loader import get_data_loader
from sklearn.cluster import KMeans
from dataloaders.sampler import data_sampler
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import collections
import json
from tqdm import tqdm
import logging
from config import Param
from FlagEmbedding import BGEM3FlagModel
from FlagEmbedding.baai_general_embedding.finetune import BiEncoderModel, BiTrainer
from FlagEmbedding.baai_general_embedding.finetune.data import TrainDatasetForEmbedding, EmbedCollator
from FlagEmbedding.baai_general_embedding.finetune.run import train_retrieval
from transformers import AutoConfig, AutoTokenizer
from transformers import set_seed
import logging
import os
from pathlib import Path
from collections import Counter


logger = logging.getLogger(__name__)


color_epoch = '\033[92m' 
color_loss = '\033[92m'  
color_number = '\033[93m'
color_reset = '\033[0m'



PROMPT_TASK_TACRED = """The task involves relation extraction for two entities within a given sentence. 
There are five classes: {re1}, {re2}, {re3}, {re4}, each representing different types of relationships that can exist between the two entities. 
The goal is to classify the relationship between the entities into one of these classes based on the context provided by the sentence
{relation}
Example: {example}
"""

PROMPT_TASK_TACRED_ALL = """The task involves relation extraction for two entities within a given sentence. 
There are five classes: {re1}, {re2}, {re3}, {re4}, each representing different types of relationships that can exist between the two entities. 
The goal is to classify the relationship between the entities into one of these classes based on the context provided by the sentence
{relation1}
{relation2}
{relation3}
{relation4}
Example {example_relation}: {example}
"""



PROMPT_NO_RELATION = """In the extracted passage, tokens [E11], [E12], [E21], [E22] appear to mark the positions of entities. However, the words or phrases between them are not linked or refer to any relationship between these entities.
Example: {example}"""

def set_seed_classifier(config, seed):
    config.n_gpu = torch.cuda.device_count()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if config.n_gpu > 0 and torch.cuda.is_available() and config.use_gpu:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        

class CELoss(nn.Module):
    def __init__(self, temperature=2.0):
        super(CELoss, self).__init__()
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pros, label):
        bce_loss = self.ce_loss(pros, label)
        return bce_loss




def save_jsonl(filename, data):
    with open(filename, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')
            
            
    
def save_model(config, lora_model, classifier_model, file_name, task):
    lora_model.save_lora('./' + config.save_checkpoint + file_name)
    torch.save(classifier_model, './' + config.save_checkpoint + file_name + f'/checkpoint_task_{task}.pt')
    

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
    distances = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto').fit_transform(features)

    memory = []
    for k in range(num_clusters):
        sel_index = np.argmin(distances[:, k])
        instance = relation_dataset[sel_index]
        memory.append(instance)
    return memory




def train_simple_model(config, encoder, classifier, training_data, epochs, map_relid2tempid, test_data, seen_relations, steps):
    data_loader = get_data_loader(config, training_data, shuffle=True)
    encoder.train()
    classifier.train()

    optim_acc = 0.0
    criterion = CELoss()
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': 0.0001},
        {'params': classifier.parameters(), 'lr': 0.001}
    ])
    
    for epoch_i in range(epochs):
        losses = []
        for step, batch_data in enumerate(tqdm(data_loader)):
            optimizer.zero_grad()
            labels, tokens = batch_data
            labels = labels.to(config.device)
            labels = [map_relid2tempid[x.item()] for x in labels]
            
            labels = torch.tensor(labels).to(config.device)
            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
            reps = encoder(tokens)

            logits = classifier(reps)

            loss = criterion(pros=logits, label=labels)    
            
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            
        acc = evaluate_strict_model(config, encoder, classifier, test_data, seen_relations, map_relid2tempid)[0]
        
        print(f"{color_epoch}Epoch:{color_reset} {color_number}{epoch_i}{color_reset}," 
            + f"{color_loss}Loss:{color_reset} {color_number}{np.array(losses).mean()}{color_reset}," 
            + f"{color_epoch}Accuracy:{color_reset} {color_number}{acc}{color_reset}")

        if acc >= optim_acc:
            optim_acc = acc
            state_classifier = {
                'state_dict': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            path_save = f"checkpoint_task_{steps}"            
            save_model(config, encoder, state_classifier, path_save, steps)

    return optim_acc

def construct_hard_triplets(output, labels, relation_data):
    positive = []
    negative = []
    pdist = nn.PairwiseDistance(p=2)
    for rep, label in zip(output, labels):
        positive_relation_data = relation_data[label.item()]
        negative_relation_data = []
        for key in relation_data.keys():
            if key != label.item():
                negative_relation_data.extend(relation_data[key])
        positive_distance = torch.stack([pdist(rep.cpu(), p) for p in positive_relation_data])
        negative_distance = torch.stack([pdist(rep.cpu(), n) for n in negative_relation_data])
        positive_index = torch.argmax(positive_distance)
        negative_index = torch.argmin(negative_distance)
        positive.append(positive_relation_data[positive_index.item()])
        negative.append(negative_relation_data[negative_index.item()])


    return positive, negative


def batch2device(batch_tuple, device):
    ans = []
    for var in batch_tuple:
        if isinstance(var, torch.Tensor):
            ans.append(var.to(device))
        elif isinstance(var, list):
            ans.append(batch2device(var))
        elif isinstance(var, tuple):
            ans.append(tuple(batch2device(var)))
        else:
            ans.append(var)
    return ans




def most_frequent(arr):
    counter = Counter(arr)
    most_common = counter.most_common(1) 
    return most_common[0][0]


def get_values_from_indices(array_n, array_m):
    return [array_n[index] for index in array_m if index < len(array_n)]


def top_k_indices(arr, k):
    indexed_arr = list(enumerate(arr))
    sorted_arr = sorted(indexed_arr, key=lambda x: x[1], reverse=True)
    top_indices = [index for index, _ in sorted_arr[:k]]
    return top_indices



def most_frequent_value(array):
    frequency_dict = {}
    max_frequency = 0
    most_frequent_value = None
    index_of_most_frequent_value = None
    
    for index, value in enumerate(array):
        if value in frequency_dict:
            frequency_dict[value] += 1
        else:
            frequency_dict[value] = 1
        
        if frequency_dict[value] > max_frequency:
            max_frequency = frequency_dict[value]
            most_frequent_value = value
            index_of_most_frequent_value = index

    return most_frequent_value, index_of_most_frequent_value



def evaluate_strict_all(config, steps, retrieval_path, test_data_all, memories_data, list_map_relid2tempid, description, id2rel):    
    # if first task
    if steps == 0:      
        result_retrieval, result_classifier, result_total = [], [], []
        
        map_relid2tempid = list_map_relid2tempid[steps]
        encoder = Bert_LoRa(config=config, path_adapter=f'/kaggle/working/checkpoint/checkpoint_task_{steps}').to(config.device)
        clasifier = Classifier_Layer(config=config, num_class=config.rel_per_task).to(config.device)
        clasifier.load_state_dict(torch.load(f'/kaggle/working/checkpoint/checkpoint_task_{steps}/checkpoint_task_{steps}.pt')['state_dict'])
        

        accuracy_classifier, count_accuracy = evaluate_strict_model(
            config,
            encoder,
            clasifier,
            test_data_all[steps]['data'],
            test_data_all[steps]['current_relation'],
            map_relid2tempid,
        )

        result_retrieval.append(1.0)
        result_classifier.append(accuracy_classifier)
        result_total.append(accuracy_classifier)
        
        print(f"Result for retrieval: {result_retrieval}")
        print(f"Result for classifier: {result_classifier}")
        print(f"Result for total: {result_total}")
        return {
            'task': steps,
            'retrieval': result_retrieval,
            'classifier': result_classifier,
            'total': result_total,
        }
        
    # if task > 0
    else:
        result_retrieval, result_classifier = [], []
        result_total = [0 for _ in range(steps + 1)]
        data_for_classifier_task = [[] for _ in range(steps + 1)]
        
        # Load model bge
        bge_model = BGEM3FlagModel(retrieval_path, use_fp16=True, device='cuda')
        # bge_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device='cuda')
        
        
        memories_data_text, memories_data_task, memories_data_relation = [], [], []
        
        for task in memories_data:
            for keys, values in enumerate(task['data']):
                for sample in task['data'][values]:
                    if config.task_name == 'TACRED':
                        memories_data_text.append(PROMPT_TASK_TACRED.format_map({
                                're1': task['relations_task'][-4],
                                're2': task['relations_task'][-3],
                                're3': task['relations_task'][-2],
                                're4': task['relations_task'][-1],
                                'relation': description[id2rel[sample['relation']]],
                                'example': sample['text']
                            }))
                        memories_data_task.append(task['task'])
                        memories_data_relation.append(sample['relation'])
            
        # Embedding memories data
        print(f"Length passage for retrieval: {len(memories_data_text)}")
        embedding_memories_data = bge_model.encode(memories_data_text, max_length=config.max_length_passage, return_dense=True, return_colbert_vecs=True)
        count_true_retrieval_total, count_total_text_data = 0, 0
        count_true_classifier_total, count_total_text_classifier = 0, 0
        
        # count_acc_per_relation = {}
        # for task, test_data in enumerate(test_data_all):
        #     for item in test_data:
        #         count_acc_per_relation[id2rel[item['relation']]] = {
        #             'true': 0,
        #             'total': len(test_data)
        #         }
            
        for task, test_data in enumerate(test_data_all):
            count_retrieval = 0
            test_data_text = [sample['text'] for sample in test_data['data']]
            test_data_relation = [sample['relation'] for sample in test_data['data']]
            count_total_text_data += len(test_data_text)

            
            if config.type_similar == 'colbert':
                # Embedding test data
                embedding_test_data = bge_model.encode(test_data_text, max_length=config.max_length_query, return_dense=True, return_colbert_vecs=True)
                result = embedding_test_data['dense_vecs'] @ embedding_memories_data['dense_vecs'].T
                result = result.tolist()
                    
                for idx_query, query_text in enumerate(test_data_text):
                    negative_indices = top_k_indices(result[idx_query], config.top_k_retrieval)
                    negative = get_values_from_indices(memories_data_text, negative_indices)
                    value_task, predict_task = most_frequent_value(negative)
                    
                    if predict_task == memories_data_task[predict_task]:
                        count_retrieval += 1
                        count_true_retrieval_total += 1
                        data_for_classifier_task[task].append(test_data['data'][idx_query])

            result_retrieval.append(count_retrieval / len(test_data_text))
      
      
        # eval classifier 
        for steps, data_task in enumerate(data_for_classifier_task):
            map_relid2tempid = list_map_relid2tempid[steps]
            id_relation = list(map_relid2tempid.keys())
            count_total_text_classifier += len(data_task)
            
            
            encoder = Bert_LoRa(config=config, path_adapter=f'/kaggle/working/checkpoint/checkpoint_task_{steps}').to(config.device)
            clasifier = Classifier_Layer(config=config, num_class=steps * 4 + 4).to(config.device)
            clasifier.load_state_dict(torch.load(f'/kaggle/working/checkpoint/checkpoint_task_{steps}/checkpoint_task_{steps}.pt')['state_dict'])
            
            accuracy_classifier, count_accuracy = evaluate_strict_model(
                config,
                encoder,
                clasifier,
                data_task,
                test_data_all[steps]['current_relation'],
                map_relid2tempid,
            )
            
            count_true_classifier_total += count_accuracy
            
            result_classifier.append(accuracy_classifier)
      

        print(f"Result for retrieval per task: {result_retrieval}")
        print(f"Result for classifier per task: {result_classifier}")
        print(f"Count retrieval: {count_true_retrieval_total}, Count Total: {count_true_classifier_total}")
        print(f"Mean retrieval: {count_true_retrieval_total / count_total_text_data},  Total: {count_true_classifier_total / count_total_text_data}")

        return {
            'task': steps,
            'retrieval': result_retrieval,
            'classifier': result_classifier,
            'total': result_total,
            'mean_retrieval': count_true_retrieval_total / count_total_text_data,
            'mean_classifier': count_true_classifier_total / count_total_text_classifier,
            'mean_total': count_true_classifier_total / count_total_text_data
        }
        
        
        
def evaluate_strict_model(config, encoder, classifier, test_data, seen_relations, map_relid2tempid):
    data_loader = get_data_loader(config, test_data, batch_size=1)
    encoder.eval()
    classifier.eval()
    n = len(test_data)

    correct = 0
    for step, batch_data in enumerate(data_loader):
        labels, tokens = batch_data
        labels = labels.to(config.device)
        labels = [map_relid2tempid[x.item()] for x in labels]
        labels = torch.tensor(labels).to(config.device)

        tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
        reps = encoder(tokens)
        logits = classifier(reps)

        seen_relation_ids = [rel2id[relation] for relation in seen_relations]
        seen_relation_ids = [map_relid2tempid[relation] for relation in seen_relation_ids]
        seen_sim = logits[:, seen_relation_ids].cpu().data.numpy()
        max_smi = np.max(seen_sim, axis=1)

        label_smi = logits[:,labels].cpu().data.numpy()

        if label_smi >= max_smi:
            correct += 1

    return correct/n, correct





def prepare_data_for_retrieval(config, steps, bge_m3, current_relation, description, current_data, memory_data, id2rel):
    retrieval_model = BGEM3FlagModel(bge_m3, use_fp16=True, device='cuda')
    
    print("---" * 23 + 'Preparing data for training retrieval model!' + "---" * 23 + '\n')
    data_train = []
    
    if steps == 0:
        no_relation = json.load(open(config.data_path + config.no_relation, 'r'))
        no_relation_text = [PROMPT_NO_RELATION.format_map({'example': " ".join(item["tokens"])}) for item in no_relation]
        
        query = [item['text'] for item in current_data]
        task_relation = [item['relation'] for item in current_data]
        
        embedding_query = retrieval_model.encode(query, max_length=config.max_length_query, return_dense=True, return_colbert_vecs=True)
        embedding_neg = retrieval_model.encode(no_relation_text, max_length=config.max_length_passage, return_dense=True, return_colbert_vecs=True)

        if config.type_similar == 'colbert':
            result = embedding_query['dense_vecs'] @ embedding_neg['dense_vecs'].T
            result = result.tolist()
            
            
            for idx_query, query_text in enumerate(query):
                negative_indices = top_k_indices(result[idx_query], config.top_k_negative)
                negative = get_values_from_indices(no_relation_text, negative_indices)
                random.shuffle(current_data)
                
                for sample in current_data:
                    if sample['text'] != query_text and sample['relation'] == task_relation[idx_query]:
                        positive = PROMPT_TASK_TACRED.format_map({
                            're1': current_relation[0],
                            're2': current_relation[1],
                            're3': current_relation[2],
                            're4': current_relation[3],
                            'relation': description[id2rel[sample['relation']]],
                            'example': sample['text']
                        })
                        data_train.append({
                            'query': query_text,
                            'pos': [positive],
                            'neg': negative,
                        })
                        break
    
    
    else:
        current_data_for_relation = {}
        for relation in current_relation:
            current_data_for_relation[relation] = []
            
        for sample in current_data:
            current_data_for_relation[id2rel[sample['relation']]].append(sample)
        
        memory_data.append({
            'relations_task': current_relations,
            'data': current_data_for_relation,
            'task': len(memory_data)
        })
        
        for task in range(len(memory_data)):
            current_query, task_relation, original_this_task, neg_query = [], [], [], []
            this_list_relation = memory_data[task]['relations_task']
            for relation_temp in this_list_relation:
                try:
                    original_this_task += memory_data[task]['data'][relation_temp]
                except:
                    print("Error relation: {relation_temp}")
                    continue
                
            for keys, re_task in enumerate(memory_data[task]['data']):
                current_query += [item['text'] for item in memory_data[task]['data'][re_task]]
                task_relation += [item['relation'] for item in memory_data[task]['data'][re_task]]
    
            for task_neg in range(len(memory_data)):
                if task != task_neg:
                    neg_relation = memory_data[task_neg]['relations_task']
                    for keys, re_task in enumerate(memory_data[task_neg]['data']):
                        neg_query += [PROMPT_TASK_TACRED.format_map({
                                're1': neg_relation[0],
                                're2': neg_relation[1],
                                're3': neg_relation[2],
                                're4': neg_relation[3],
                                'relation': description[id2rel[sample['relation']]],
                                'example': sample['text']
                        }) for sample in memory_data[task_neg]['data'][re_task]]

            embedding_query = retrieval_model.encode(current_query, max_length=config.max_length_query, return_dense=True, return_colbert_vecs=True)
            embedding_neg = retrieval_model.encode(neg_query, max_length=config.max_length_passage, return_dense=True, return_colbert_vecs=True)  

            if config.type_similar == 'colbert':
                result = embedding_query['dense_vecs'] @ embedding_neg['dense_vecs'].T
                result = result.tolist()
                
                for idx_query, query_text in enumerate(current_query):
                    negative_indices = top_k_indices(result[idx_query], config.top_k_negative)
                    negative = get_values_from_indices(neg_query, negative_indices)
                    
                    random.shuffle(original_this_task)
                    
                    for sample in original_this_task:
                        if sample['text'] != query_text and sample['relation'] == task_relation[idx_query]:
                            positive = PROMPT_TASK_TACRED.format_map({
                                're1': this_list_relation[0],
                                're2': this_list_relation[1],
                                're3': this_list_relation[2],
                                're4': this_list_relation[3],
                                'relation': description[id2rel[sample['relation']]],
                                'example': sample['text']
                            })
                            
                            data_train.append({
                                'query': query_text,
                                'pos': [positive],
                                'neg': negative,
                            })
                            
                            break

    print(f"Length data train retrieval: {len(data_train)}")
    print("---" * 25 + 'Saving data train retrieval!' + "---" * 25 + '\n')
    random.shuffle(data_train)
    save_jsonl(data=data_train, filename=f'./train_step_{steps}.jsonl')
    return data_train, f'/train_step_{steps}.jsonl', '/kaggle/working/model_bge'



param = Param()
args = param.args

# Device
torch.cuda.set_device(args.gpu)
args.device = torch.device(args.device)
args.n_gpu = torch.cuda.device_count()
args.task_name = args.dataname
args.rel_per_task = 8 if args.dataname == "FewRel" else 4
args.num_class = 80 if args.dataname == "FewRel" else 4
    
if __name__ == '__main__':
    config = args

    config.device = torch.device(config.device)
    config.n_gpu = torch.cuda.device_count()
    config.total_round = 1
    
    
    for rou in range(config.total_round):
        test_cur = []
        test_total = []
        
        random.seed(config.seed + rou*100)
        sampler = data_sampler(config, seed=config.seed + rou*100)
        id2rel = sampler.id2rel
        rel2id = sampler.rel2id
            
        
        num_class = len(sampler.id2rel)
        memorized_samples = []
        memory = collections.defaultdict(list)
        history_relations = []
        history_data = []
        prev_relations = []
        classifier = None
        prev_classifier = None
        prev_encoder = None
        prev_dropout_layer = None
        relation_standard = {}
        forward_accs = []
        total_acc = []
        all_test_data = []
        list_map_relid2tempid = []
        list_retrieval = []
        description_original = json.load(open(config.description_path, 'r'))
        description_class = {}
        
        
        for sample in description_original:
            description_class[sample['relation']] = sample['text']
        
        for steps, (training_data, valid_data, test_data, current_relations, historic_test_data, seen_relations) in enumerate(sampler):

            prev_relations = history_relations[:]
            train_data_for_initial = []
            training_fix_data, test_fix_data = {}, {}

            for relation in current_relations:
                training_fix_data[relation], test_fix_data[relation] = [], []
                history_relations.append(relation)
                
                # Remove data without token entity
                for item in training_data[relation]:
                    item['task'] = steps
                    if 30522 in item['tokens'] and 30523 in item['tokens'] and 30524 in item['tokens'] and 30525 in item['tokens']: 
                        train_data_for_initial.append(item)
                        training_fix_data[relation].append(item)


                for item in test_data[relation]:
                    item['task'] = steps
                    if 30522 in item['tokens'] and 30523 in item['tokens'] and 30524 in item['tokens'] and 30525 in item['tokens']: 
                        test_fix_data[relation].append(item)



            print(f'Current relation: {current_relations}')
            print(f"Task {steps}, Num class: {len(history_relations)}")
            temp_rel2id = [rel2id[x] for x in seen_relations]
            map_relid2tempid = {k: v for v, k in enumerate(temp_rel2id)}

            encoder = Bert_LoRa(config=config).to(config.device)

            for name, param in encoder.encoder.named_parameters():
                if name.find("loranew_") != -1:
                    param.requires_grad = True
                else: 
                    param.requires_grad = False
                                 
            
            def count_trainable_params(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f'Tranable Params: {count_trainable_params(encoder)}')
            
            classifier = Classifier_Layer(config=config, num_class=len(history_relations)).to(config.device)
    
            test_data_task = []
            for relation in current_relations:
                test_data_task += test_fix_data[relation]

            bge_m3_path = config.bge_model

            cur_acc = train_simple_model(
                config, 
                encoder, 
                classifier, 
                train_data_for_initial, 
                config.classifier_epochs, 
                map_relid2tempid,
                test_data_task,
                seen_relations,
                steps,
            )


            # Prepare data for training retrieval
            data_retrieval, path_data, retrieval_model = prepare_data_for_retrieval(
                config, 
                steps, 
                bge_m3_path, 
                current_relations, 
                description_class, 
                train_data_for_initial, 
                memorized_samples, 
                id2rel
            )
            
            if steps > 0:
                train_retrieval(config=config, data_path=path_data, model_path='/kaggle/working/model_bge')
            else:
                train_retrieval(config=config, data_path=path_data, model_path=None)

            # Get memories data
            this_task_memory = {}
            
            print('---'*23 + 'Get memories data!' + '---'*23 + '\n')
            for relation in current_relations:
                this_task_memory[relation] = select_data(config, encoder, training_fix_data[relation])


            
            all_test_data.append({
                'data': test_data_task, 
                'task': steps,
                'seen_relation': history_relations,
                'current_relation': current_relations,
            })

            
            temp_rel2id = [rel2id[x] for x in history_relations]
            map_relid2tempid = {k: v for v, k in enumerate(temp_rel2id)}
            list_map_relid2tempid.append(map_relid2tempid) 
            print(f"Task {steps}: {list_map_relid2tempid}")
            
            cur_acc = max(cur_acc, evaluate_strict_model(config, encoder, classifier, test_data_task, seen_relations, map_relid2tempid)[0])
            test_cur.append(cur_acc)
            total_acc.append(cur_acc)


            torch.cuda.empty_cache()
            
            print('---'*23 + 'Evaluating!' + '---'*23 + '\n')
            print(f'Task--{steps}:')
            print(f"Length train init: {len(train_data_for_initial)}")
            print(f"Length test current task: {len(test_data_task)}")
            print(f'Current test acc: {cur_acc}')
            print(f'Accuracy test all task: {test_cur}')
            list_retrieval.append(evaluate_strict_all(config, steps, retrieval_model, all_test_data, memorized_samples, list_map_relid2tempid, description_class, id2rel))
            print('---'*23 + f'Finish task {steps}!' + '---'*23 + '\n')
            
            # add data to memories
            memorized_samples.append({
                'relations_task': current_relations,
                'data': this_task_memory,
                'task': len(memorized_samples)
            })
            
            json.dump(list_retrieval, open(f'./task_{steps}.json', 'w'), ensure_ascii=False)
            
        print(f"Finish result: {list_retrieval}")