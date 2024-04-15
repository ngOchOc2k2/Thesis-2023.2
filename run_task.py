import random
import torch
from methods.backbone import Bert_LoRa, Classifier_Layer
from dataloaders.data_loader import get_data_loader
from sklearn.cluster import KMeans
from dataloaders.sampler import data_sampler
import torch.nn as nn
import torch.optim as optim
import numpy as np
import collections
import json
from tqdm import tqdm
import logging
from config import Param    
from FlagEmbedding import BGEM3FlagModel
from FlagEmbedding.baai_general_embedding.finetune.run import train_retrieval, train_retrieval_distil
import logging
from collections import Counter
import subprocess
import os
import re
from copy import deepcopy


logger = logging.getLogger(__name__)


color_epoch = '\033[92m' 
color_loss = '\033[92m'  
color_number = '\033[93m'
color_reset = '\033[0m'


PROMPT_TASK_TACRED = """The task involves relation extraction for two entities within a given sentence. 
There are four classes: {re1}, {re2}, {re3}, {re4}, each representing different types of relationships that can exist between the two entities. 
The goal is to classify the relationship between the entities into one of these classes based on the context provided by the sentence
{relation}
Example: {example}
"""


PROMPT_TASK_FEWREL = """The task involves relation extraction for two entities within a given sentence. 
There are eight classes: {re1}, {re2}, {re3}, {re4}, {re5}, {re6}, {re7}, {re8} each representing different types of relationships that can exist between the two entities. 
The goal is to classify the relationship between the entities into one of these classes based on the context provided by the sentence
{relation}
Example: {example}
"""



PROMPT_TASK_TACRED_ALL = """There are four relations: {re1}, {re2}, {re3}, {re4}, each representing different types of relationships that can exist between the two entities. 
{relation1}
{relation2}
{relation3}
{relation4}
Example {example_relation}: {example}
"""


PROMPT_NO_RELATION = """In the extracted passage, tokens [E11], [E12], [E21], [E22] appear to mark the positions of entities. However, the words or phrases between them are not linked or refer to any relationship between these entities.
Example: {example}"""


PROMPT_NEGATIVE = """The task involves relation extraction for two entities within a given sentence. 
There are four classes: place served by transport hub, located on terrain feature, musical conductor, participating team, each representing different types of relationships that can exist between the two entities. 
The goal is to classify the relationship between the entities into one of these classes based on the context provided by the sentence
{relation}
Example: {example}
"""


REMOVE_TOKEN = ['[E11]', '[E12]', '[E21]', '[E22]', '-rrb-', '-lrb-']

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
    criterion = CELoss(temperature=config.kl_temp)
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': config.lr_encoder},
        {'params': classifier.parameters(), 'lr': config.lr_classifier}
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


        # Get best model 
        if acc >= optim_acc:
            optim_acc = acc
            state_classifier = {
                'state_dict': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            path_save = f"checkpoint_task_{steps}"            
            save_model(config, encoder, state_classifier, path_save, steps)

    return optim_acc




def most_frequent(arr):
    return max(arr, key=arr.count)

def get_values_from_indices(array_n, array_m):
    n = len(array_n)
    return [array_n[index] for index in array_m if 0 <= index < n]

def top_k_indices(arr, k):
    return sorted(range(len(arr)), key=lambda i: arr[i], reverse=True)[:k]


def most_frequent_value(array):
    max_frequency = 0
    most_frequent_value = None
    index_of_most_frequent_value = None

    for index, value in enumerate(array):
        frequency = sum(1 for v in array if v == value)
        if frequency > max_frequency:
            max_frequency = frequency
            most_frequent_value = value
            index_of_most_frequent_value = index

    return most_frequent_value, index_of_most_frequent_value


def remove_words_in_list(sentence, word_list):
    sentence = re.sub(r'[^\w\s]', '', sentence)
    words = sentence.split()
    processed_words = [word for word in words if word not in word_list]
    processed_sentence = ' '.join(processed_words)
    return processed_sentence



def evaluate_strict_all(config, steps, test_data_all, memories_data, list_map_relid2tempid, description, data_for_retrieval, id2rel, retrieval_path=None):    
    
    # If first task
    if steps == 0:      
        result_retrieval, result_classifier, result_total = [], [], []
        
        map_relid2tempid = list_map_relid2tempid[steps]
        encoder = Bert_LoRa(config=config, path_adapter=config.checkpoint_kaggle + f'/checkpoint_task_{steps}').to(config.device)
        clasifier = Classifier_Layer(config=config, num_class=config.rel_per_task).to(config.device)
        clasifier.load_state_dict(torch.load(config.checkpoint_kaggle + f'/checkpoint_task_{steps}/checkpoint_task_{steps}.pt')['state_dict'])
        

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
        
        
    # If task > 0
    else:
        result_retrieval, result_classifier = [], []
        result_total = [0 for _ in range(steps + 1)]
        data_for_classifier_task = [[] for _ in range(steps + 1)]
        
        # Load model bge
        if config.trainable_retrieval and retrieval_path != None:
            bge_model = BGEM3FlagModel(retrieval_path, use_fp16=True, device='cuda')
        else:
            bge_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device='cuda')
        
        
        memories_data_text, memories_data_task, memories_data_relation = [], [], []
        
        for task in data_for_retrieval:
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
                        
                    else:
                        memories_data_text.append(PROMPT_TASK_FEWREL.format_map({
                                're1': task['relations_task'][-8],
                                're2': task['relations_task'][-7],
                                're3': task['relations_task'][-6],
                                're4': task['relations_task'][-5],
                                're5': task['relations_task'][-4],
                                're6': task['relations_task'][-3],
                                're7': task['relations_task'][-2],
                                're8': task['relations_task'][-1],
                                'relation': description[id2rel[sample['relation']]],
                                'example': sample['text']
                            }))
                        
                    memories_data_task.append(task['task'])
                    memories_data_relation.append(sample['relation'])

                    
            
        # Embedding memories data
        print(f"Length passage for retrieval: {len(memories_data_text)}")
        embedding_memories_data = bge_model.encode(memories_data_text, max_length=config.max_length_passage, return_dense=config.dense_vecs, return_colbert_vecs=config.colbert_vecs)
        count_true_retrieval_total, count_total_text_data = 0, 0
        count_true_classifier_total, count_total_text_classifier = 0, 0
        
            
        for task, test_data in enumerate(test_data_all):
            count_retrieval = 0
            test_data_text = [sample['text'] for sample in test_data['data']]
            count_total_text_data += len(test_data_text)

            
            if config.type_similar == 'dense':
                
                # Embedding test data
                embedding_test_data = bge_model.encode(test_data_text, max_length=config.max_length_query, return_dense=config.dense_vecs, return_colbert_vecs=config.colbert_vecs)
                result = embedding_test_data['dense_vecs'] @ embedding_memories_data['dense_vecs'].T
                result = result.tolist()
                    
                for idx_query, query_text in enumerate(result):
                    negative_indices = top_k_indices(result[idx_query], config.top_k_retrieval)
                    negative = get_values_from_indices(memories_data_task, negative_indices)
                    value_task, predict_task = most_frequent_value(negative)
                    
                    # if value_task == task:
                    if task == negative[predict_task]:
                        count_retrieval += 1
                        count_true_retrieval_total += 1
                        data_for_classifier_task[task].append(test_data['data'][idx_query])
                    
            result_retrieval.append(count_retrieval / len(test_data_text))
      
      
        # Eval classifier 
        for steps, data_task in enumerate(data_for_classifier_task):
            map_relid2tempid = list_map_relid2tempid[steps]
            count_total_text_classifier += len(data_task)
            
            
            encoder = Bert_LoRa(config=config, path_adapter=config.checkpoint_kaggle +  f'/checkpoint_task_{steps}').to(config.device)
            clasifier = Classifier_Layer(config=config, num_class=steps * config.rel_per_task + config.rel_per_task).to(config.device)
            clasifier.load_state_dict(torch.load(config.checkpoint_kaggle + f'/checkpoint_task_{steps}/checkpoint_task_{steps}.pt')['state_dict'])
            
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
            'mean_retrieval': count_true_retrieval_total / count_total_text_data,
            'mean_classifier': count_true_classifier_total / count_total_text_classifier,
            'mean_total': count_true_classifier_total / count_total_text_data
        }
        
        
        
def evaluate_strict_model(config, encoder, classifier, test_data, seen_relations, map_relid2tempid):
    if len(test_data) == 0:
        return 0, 0
    
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


def get_description(config, task, example, relation_type, description, data_type, id2rel):
    if data_type == 'TACRED':
        if config.description_type == 'single':
            return PROMPT_TASK_TACRED.format_map({
            're1': relation_type[0],
            're2': relation_type[1],
            're3': relation_type[2],
            're4': relation_type[3],
            'relation': description[id2rel[example['relation']]],
            'example': remove_words_in_list(example['text'], REMOVE_TOKEN)
        })
        else:
            return PROMPT_TASK_TACRED_ALL.format_map({
            're1': relation_type[0],
            're2': relation_type[1],
            're3': relation_type[2],
            're4': relation_type[3],
            'relation1': description[relation_type[0]],
            'relation2': description[relation_type[1]],
            'relation3': description[relation_type[2]],
            'relation4': description[relation_type[3]],
            'example_relation': description[id2rel[example['relation']]],
            'example': remove_words_in_list(example['text'], REMOVE_TOKEN)
        })    
    else:
        return PROMPT_TASK_FEWREL.format_map({
        're1': relation_type[0],
        're2': relation_type[1],
        're3': relation_type[2],
        're4': relation_type[3],
        're5': relation_type[4],
        're6': relation_type[5],
        're7': relation_type[6],
        're8': relation_type[7],
        'relation': description[id2rel[example['relation']]],
        'example': remove_words_in_list(example['text'], REMOVE_TOKEN)
    })
        

def get_des_fewrel(example, description):
    for des in description:
        if des['relation'] == example['relation']:
            return des['text']
    return None
    
    
    
def prepare_data_for_retrieval(config, steps, bge_m3, current_relation, description, current_data, data_memory, id2rel):
    if steps == 0:
        retrieval_model = BGEM3FlagModel(bge_m3, use_fp16=True, device='cuda')
    else:
        retrieval_model = BGEM3FlagModel('/kaggle/working/model_teacher', use_fp16=True, device='cuda')
        
        
    print("---" * 23 + 'Preparing data for training retrieval model!' + "---" * 23 + '\n')
    data_train, data_last_train = [], []

    # If first task
    if steps == 0:
        negative_relation = json.load(open(config.data_path + config.no_relation, 'r'))
        description_fewrel = json.load(open(config.data_path + config.description_fewrel, 'r'))
        no_relation_text = [PROMPT_NEGATIVE.format_map({
            'relation': get_des_fewrel(item, description_fewrel),
            'example': " ".join(item["tokens"]),
        }) for item in negative_relation]
        
        query = [item['text'] for item in current_data]
        task_relation = [item['relation'] for item in current_data]
        
        # Embedding query and no relation data
        embedding_query = retrieval_model.encode(query, max_length=config.max_length_query, return_dense=config.dense_vecs, return_colbert_vecs=config.colbert_vecs)
        embedding_neg = retrieval_model.encode(no_relation_text, max_length=config.max_length_passage, return_dense=config.dense_vecs, return_colbert_vecs=config.colbert_vecs)
        
        if config.type_similar == 'dense':
            result = embedding_query['dense_vecs'] @ embedding_neg['dense_vecs'].T
            result = result.tolist()
            
            for idx_query, (query_text, rel_id) in enumerate(zip(query, task_relation)):
                negative_indices = top_k_indices(result[idx_query], config.top_k_negative)
                negative = get_values_from_indices(no_relation_text, negative_indices)
                random.shuffle(current_data)
                
                # Filter out positive examples by relation
                filtered_positive_examples = filter(lambda sample: sample['relation'] == rel_id, current_data)
                positive_example = next(filtered_positive_examples, None)
                
                if positive_example:
                    positive = get_description(
                        config=config,
                        task=steps,
                        example=positive_example,
                        relation_type=current_relation,
                        description=description,
                        data_type=config.task_name,
                        id2rel=id2rel,
                        
                    )
                    data_train.append({
                        'query': query_text,
                        'pos': [positive],
                        'neg': negative,
                    })
        
    # If task > 0
    else:
        memory_data = deepcopy(data_memory)
        current_data_for_relation = {}
        for relation in current_relation:
            current_data_for_relation[relation] = []

        # Group samples by relation
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

            # Concatenate queries and task relations
            for re_task, samples in memory_data[task]['data'].items():
                current_query += [item['text'] for item in samples]
                task_relation += [item['relation'] for item in samples]
                original_this_task += samples

            # Construct negative queries
            for task_neg in range(len(memory_data)):
                if task != task_neg:
                    for re_task, samples in memory_data[task_neg]['data'].items():
                        neg_query += [get_description(config, steps, sample, memory_data[task_neg]['relations_task'], description, config.task_name, id2rel) for sample in samples]
            
            # Embedding query and negative
            embedding_query = retrieval_model.encode(current_query, max_length=config.max_length_query, return_dense=config.dense_vecs, return_colbert_vecs=config.colbert_vecs)
            embedding_neg = retrieval_model.encode(neg_query, max_length=config.max_length_passage, return_dense=config.dense_vecs, return_colbert_vecs=config.colbert_vecs)

            if config.type_similar == 'dense':
                result = embedding_query['dense_vecs'] @ embedding_neg['dense_vecs'].T
                result = result.tolist()

                for idx_query, (query_text, rel_id) in enumerate(zip(current_query, task_relation)):
                    negative_indices = top_k_indices(result[idx_query], config.top_k_negative)
                    negative = get_values_from_indices(neg_query, negative_indices)

                    random.shuffle(original_this_task)

                    # Find positive examples
                    for sample in original_this_task:
                        if sample['text'] != query_text and sample['relation'] == rel_id:
                            positive = get_description(
                                config=config,
                                task=steps,
                                example=sample,
                                relation_type=this_list_relation,
                                description=description,
                                data_type=config.task_name,
                                id2rel=id2rel,
                            )
                            
                            data_train.append({
                                'query': query_text,
                                'pos': [positive],
                                'neg': negative,
                            })

                            if task == len(memory_data) - 1:
                                data_last_train.append({
                                    'query': query_text,
                                    'pos': [positive],
                                    'neg': negative,
                                })

                            break

    print(f"Length total retrieval: {len(data_train)}, last data retrieval: {len(data_last_train)}")
    print("---" * 25 + 'Saving data train retrieval!' + "---" * 25 + '\n')
    random.shuffle(data_train)
    save_jsonl(data=data_train, filename=f'/kaggle/working/train_step_{steps}.jsonl')
    save_jsonl(data=data_train, filename=f'/kaggle/working/train_last_{steps}.jsonl')
    return data_train, data_last_train, f'/train_step_{steps}.jsonl', f'/train_last_{steps}.jsonl', '/kaggle/working/model_teacher'




param = Param()
args = param.args


# Device
torch.cuda.set_device(args.gpu)
args.device = torch.device(args.device)
args.n_gpu = torch.cuda.device_count()
args.task_name = args.dataname


if args.dataname == 'FewRel':
    args.rel_per_task = 8 
    args.num_class = 80
    args.max_length_passage = 1024
    args.batch_size = 32
    args.description_path = "/kaggle/input/data-relation/datasets/standard/description_fewrel.json" 
    
else:
    args.rel_per_task = 4
    args.num_class = 40
    args.batch_size = 16
    args.max_length_passage = 768
    args.description_path = "/kaggle/input/data-relation/datasets/standard/description_tacred.json" 

    
if __name__ == '__main__':
    config = args

    config.device = torch.device(config.device)
    config.n_gpu = torch.cuda.device_count()
    path = '/kaggle/working/results'
    if os.path.exists(path):
        os.mkdir(path)

    
    for rou in range(config.total_round):
        random.seed(config.seed + rou*100)
        sampler = data_sampler(config, seed=config.seed + rou*100)
        id2rel = sampler.id2rel
        rel2id = sampler.rel2id
            
        
        num_class = len(sampler.id2rel)
        memorized_samples = []
        memory = collections.defaultdict(list)
        history_relations, list_map_relid2tempid = [], []
        history_data, prev_relations = [], []
        test_cur, test_total = [], []
        classifier = None
        relation_standard, description_class = {}, {}
        total_acc, all_test_data = [], []
        data_for_retrieval, list_retrieval = [], []
        description_original = json.load(open(config.description_path, 'r'))
        
        
        for sample in description_original:
            description_class[sample['relation']] = sample['text']
        
        for steps, (training_data, valid_data, test_data, current_relations, historic_test_data, seen_relations) in enumerate(sampler):

            prev_relations = history_relations[:]
            train_data_for_initial = []
            training_fix_data, test_fix_data = {}, {}

            for relation in current_relations:
                training_fix_data[relation], test_fix_data[relation] = [], []
                history_relations.append(relation)
                
                
                # Remove data without entity tokens
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


            ################################# Prepare Bert and Classifier model #######################################
            encoder = Bert_LoRa(config=config).to(config.device)

            for name, param in encoder.encoder.named_parameters():
                if name.find("lora") != -1:
                    param.requires_grad = True
                else: 
                    param.requires_grad = False
            
            def count_trainable_params(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f'Tranable Params: {count_trainable_params(encoder)}')
            
            classifier = Classifier_Layer(config=config, num_class=len(history_relations)).to(config.device)
            ############################################################################################################
    
    
    
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

            torch.cuda.empty_cache()

            # Prepare data for training retrieval
            retrieval_model = None
            
            if config.trainable_retrieval: 
                data_total, data_last, path_total, path_last, retrieval_model = prepare_data_for_retrieval(
                    config, 
                    steps, 
                    bge_m3_path, 
                    current_relations, 
                    description_class, 
                    train_data_for_initial, 
                    memorized_samples, 
                    id2rel
                )
            

            if config.trainable_retrieval:
                if steps > 0:
                    train_retrieval(
                        config=config, 
                        data_path=path_total, 
                        model_path='/kaggle/working/model_teacher',
                        output_dir='/kaggle/working/model_bge',
                    )
                    
                    train_retrieval_distil(
                        config=config, 
                        data_path=path_last, 
                        model_path='/kaggle/working/model_bge', 
                        model_teacher='/kaggle/working/model_teacher',
                        output_dir='/kaggle/working/model_teacher',
                        epochs=3
                    )
                else:
                    train_retrieval(config=config, data_path=path_total, model_path=None)


            # Get memories data
            this_task_memory = {}
              
            
            print('---'*23 + 'Get memories data!' + '---'*23 + '\n')
            for relation in current_relations:
                this_task_memory[relation] = select_data(config, encoder, training_fix_data[relation])

            
            # Add data to memories
            data_for_retrieval.append({
                'relations_task': current_relations,
                'data': this_task_memory,
                'task': steps,
            })
            
            
            # Get data test full task
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
            list_retrieval.append(evaluate_strict_all(config, steps, all_test_data, memorized_samples, list_map_relid2tempid, description_class, data_for_retrieval, id2rel, retrieval_path=retrieval_model))
            print('---'*23 + f'Finish task {steps}!' + '---'*23 + '\n')
            
            memorized_samples.append({
                'relations_task': current_relations,
                'data': this_task_memory,
                'task': len(memorized_samples)
            })
            
            json.dump(list_retrieval, open(config.output_kaggle + f'/task_{steps}.json', 'w'), ensure_ascii=False)
            
        print(f"Finish result: {list_retrieval}")