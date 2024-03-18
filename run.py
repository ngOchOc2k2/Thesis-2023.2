import random
import torch
from methods.backbone import Bert_Encoder, Softmax_Layer, Dropout_Layer, Bert_LoRa, Classifier_Layer
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
import os


# logging.basicConfig(filename='./Ngoc/Tacred_5_O_LoRA_256.log',level=print, format='%(asctime)s - %(levelname)s - %(message)s')



color_epoch = '\033[92m' 
color_loss = '\033[92m'  
color_number = '\033[93m'
color_reset = '\033[0m'


def set_seed(config, seed):
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

    
    
def save_model(config, lora_model, classifier_model, file_name, task):
    current_file_path = os.path.abspath(__file__)
    parent_folder = os.path.dirname(current_file_path)


    if not os.path.exists(parent_folder + '/' + config.save_checkpoint + file_name):
        os.makedirs(parent_folder + '/' + config.save_checkpoint + file_name)
        
    lora_model.save_lora(parent_folder + '/' + config.save_checkpoint + file_name)
    torch.save(classifier_model, parent_folder + '/' + config.save_checkpoint + file_name + f'/checkpoint_task_{task}.pt')


def train_simple_model(config, encoder, classifier, training_data, epochs, map_relid2tempid, test_data, seen_relations, steps):
    data_loader = get_data_loader(config, training_data, shuffle=True)
    encoder.train()
    classifier.train()

    optim_acc = 0.0
    criterion = CELoss()
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': 0.00002},
        {'params': classifier.parameters(), 'lr': 0.0005}
    ])
    
    for epoch_i in range(epochs):
        losses = []
        for step, batch_data in enumerate(tqdm(data_loader)):
            optimizer.zero_grad()
            labels, tokens = batch_data
            labels = labels.to(config.device)
            labels = [map_relid2tempid[x.item()] for x in labels]
            
            labels = torch.tensor(labels).to(config.device)
            tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
            reps = encoder(tokens)

            logits = classifier(reps)

            loss = criterion(pros=logits, label=labels)    
            
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            
        acc = evaluate_strict_model(config, encoder, classifier, test_data, seen_relations, map_relid2tempid)
        
        print(f"{color_epoch}Epoch:{color_reset} {color_number}{epoch_i}{color_reset}," 
            + f"{color_loss}Loss:{color_reset} {color_number}{np.array(losses).mean()}{color_reset}," 
            + f"{color_epoch}Accuracy:{color_reset} {color_number}{acc}{color_reset}")

        # if acc >= optim_acc:
        #     optim_acc = acc
        #     state_classifier = {
        #         'state_dict': classifier.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #     }
        #     path_save = f"checkpoint_task_{steps}"            
        #     save_model(config, encoder, state_classifier, path_save, steps)


def compute_jsd_loss(m_input):
    m = m_input.shape[0]
    mean = torch.mean(m_input, dim=0)
    jsd = 0
    for i in range(m):
        loss = F.kl_div(F.log_softmax(mean, dim=-1), F.softmax(m_input[i], dim=-1), reduction='none')
        loss = loss.sum()
        jsd += loss / m
    return jsd


def contrastive_loss(hidden, labels):

    logsoftmax = nn.LogSoftmax(dim=-1)

    return -(logsoftmax(hidden) * labels).sum() / labels.sum()


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
        seen_sim = logits[:,seen_relation_ids].cpu().data.numpy()
        max_smi = np.max(seen_sim,axis=1)

        label_smi = logits[:,labels].cpu().data.numpy()

        if label_smi >= max_smi:
            correct += 1

    return correct/n


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
    config.total_round = 6
    
    for rou in range(config.total_round):
        test_cur = []
        test_total = []
        
        random.seed(config.seed + rou*100)
        sampler = data_sampler(config, seed=config.seed + rou*100)
        id2rel = sampler.id2rel
        rel2id = sampler.rel2id
            
        
        num_class = len(sampler.id2rel)
        memorized_samples = {}
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
        
        for steps, (training_data, valid_data, test_data, current_relations, historic_test_data, seen_relations) in enumerate(sampler):
            
            prev_relations = history_relations[:]
            train_data_for_initial = []
            count = 0
            
            for relation in current_relations:
                history_relations.append(relation)
                
                for item in training_data[relation]:
                    if 30522 in item['tokens'] and 30523 in item['tokens'] and 30524 in item['tokens'] and 30525 in item['tokens']: 
                        train_data_for_initial.append(item)
                        count += 1

            print(f'Current relation: {current_relations}')
            print(f"Task {steps + 1}, Num class: {len(history_relations)}")
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
    
            test_data_all = []
            for relation in current_relations:
                test_data_all += test_data[relation]


            train_simple_model(
                config, 
                encoder, 
                classifier, 
                train_data_for_initial, 
                config.classifier_epochs, 
                map_relid2tempid,
                test_data_all,
                seen_relations,
                steps + 1,
            )

            torch.cuda.empty_cache()
            
            cur_acc = evaluate_strict_model(config, encoder, classifier, test_data_all, seen_relations, map_relid2tempid)
            test_cur.append(cur_acc)
            total_acc.append(cur_acc)
            
            
            print('---'*23 + 'Evaluating' + '---'*23)
            print(f"Length expanded train init: {len(train_data_for_initial)}")
            print(f"Length test current task: {len(test_data_all)}")
            print(f'Task--{steps + 1}:')
            print(f'Current test acc: {cur_acc}')
            print(f'Accuracy_test_current: {test_cur}')
            print('---'*23 + f'Finish task {steps + 1}' + '---'*23)
            
            accuracy = []
            temp_rel2id = [rel2id[x] for x in history_relations]
            map_relid2tempid = {k: v for v, k in enumerate(temp_rel2id)}
        
        # means = [sum(pair) / len(pair) for pair in zip(*total_acc)]
        # print(f"Mean acc: {means}")
