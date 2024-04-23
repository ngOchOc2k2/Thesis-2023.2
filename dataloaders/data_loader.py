import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import random
import json
import os
from typing import List, Tuple
import datasets


class data_set(Dataset):
    def __init__(self, data, config=None):
        self.data = data
        self.config = config
        self.bert = True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], idx)

    def collate_fn(self, data):
        label = torch.tensor([item[0]["relation"] for item in data])
        tokens = [torch.tensor(item[0]["tokens"]) for item in data]
        text = [item[0]["tokens"] for item in data]
        ind = [item[1] for item in data]

        try:
            key = [torch.tensor(item[0]["key"]) for item in data]
            return (label, tokens)  
        except:
            return (label, tokens)


def get_data_loader(config, data, shuffle=False, drop_last=False, batch_size=None):
    
    dataset = data_set(data, config)
    if batch_size == None:
        batch_size = config.batch_size
    batch_size = min(batch_size, len(data))
    
    return DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        pin_memory=True, 
        num_workers=config.num_workers, 
        collate_fn=dataset.collate_fn, 
        drop_last=drop_last
    )
    


class data_set_retrieval(Dataset):
    def __init__(self, args):
        if os.path.isdir(args.train_data):
            train_datasets = []
            for file in os.listdir(args.train_data):
                temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.train_data, file), split='train')
                train_datasets.append(temp_dataset)    
            self.dataset = datasets.concatenate_datasets(train_datasets)
            
        else:
            self.dataset = datasets.load_dataset('json', data_files=args.train_data, split='train')

        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
        self.args = args
        self.total_len = len(self.dataset)
        self.passage_max_len = args.passage_max_len
        self.query_max_len = args.query_max_len

    def __len__(self):
        return self.total_len
    
    
    def __getitem__(self, item) -> Tuple[str, List[str]]:
        query = self.dataset[item]['query']
        
        passages_negative, passages_positive = [], []

        assert isinstance(self.dataset[item]['pos'], list)
        passages_positive.extend(self.dataset[item]['pos'])

        passages_negative.extend(self.dataset[item]['neg'])
        
        return query, passages_positive, passages_negative


    def collate_fn(self, batch):
        query, passages_positive, passages_negative = zip(*batch)

        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(positive[0], list):
            positive = sum(positive, [])
        if isinstance(negative[0], list):
            negative = sum(negative, [])

        query_token = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )
        positive_token = self.tokenizer(
            positive,
            padding=True,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
        )
        
        negative_token = self.tokenizer(
            positive,
            padding=True,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
        )
        return query_token, positive_token, negative_token
    