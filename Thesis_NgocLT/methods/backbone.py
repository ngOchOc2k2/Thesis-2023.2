import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import json
import torch.utils.checkpoint
from transformers.models.bert.configuration_bert import BertConfig
from transformers import BertModel, BertForMaskedLM, BertConfig
from peft import LoraConfig, get_peft_model
from peft.peft_model import PeftModel
from copy import deepcopy
import os
import numpy as np

class base_model(nn.Module):

    def __init__(self):
        super(base_model, self).__init__()
        self.zero_const = nn.Parameter(torch.Tensor([0]))
        self.zero_const.requires_grad = False
        self.pi_const = nn.Parameter(torch.Tensor([3.14159265358979323846]))
        self.pi_const.requires_grad = False

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(os.path.join(path)))
        self.eval()

    def save_parameters(self, path):
        f = open(path, "w")
        f.write(json.dumps(self.get_parameters("list")))
        f.close()

    def load_parameters(self, path):
        f = open(path, "r")
        parameters = json.loads(f.read())
        f.close()
        for i in parameters:
            parameters[i] = torch.Tensor(parameters[i])
        self.load_state_dict(parameters, strict=False)
        self.eval()

    def get_parameters(self, mode="numpy", param_dict=None):
        all_param_dict = self.state_dict()
        if param_dict == None:
            param_dict = all_param_dict.keys()
        res = {}
        for param in param_dict:
            if mode == "numpy":
                res[param] = all_param_dict[param].cpu().numpy()
            elif mode == "list":
                res[param] = all_param_dict[param].cpu().numpy().tolist()
            else:
                res[param] = all_param_dict[param]
        return res

    def set_parameters(self, parameters):
        for i in parameters:
            parameters[i] = torch.Tensor(parameters[i])
        self.load_state_dict(parameters, strict = False)
        self.eval()
        
        
class Bert_Encoder(base_model):

    def __init__(self, config, is_lora=True):
        super(Bert_Encoder, self).__init__()

        # get config
        self.bert_config = BertConfig.from_pretrained(config.bert_path)

        # get output size
        self.output_size = config.encoder_output_size

        # get model
        self.encoder = BertModel.from_pretrained(config.bert_path).cuda()

        
        # get LoRA  
        if is_lora == True:
            configs = LoraConfig(
                r=8,   
                lora_alpha=16, 
                target_modules=["query", "value", "key"],
                lora_dropout=0.05, 
            )
            self.encoder = PeftModel(self.encoder, configs)


        self.drop = nn.Dropout(config.drop_out)
            
        if config.pattern in ['standard', 'entity_marker']:
            self.pattern = config.pattern
        else:
            raise Exception('Wrong encoding.')
        config.hidden_size = self.bert_config.hidden_size
        config.output_size = config.encoder_output_size
        
        self.encoder.resize_token_embeddings(config.vocab_size + config.marker_size)


    def print_trainable_parameters(self, model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )
    
    
    def forward(self, inputs):
        e11 = []
        e21 = []

        for i in range(inputs.size()[0]):
            tokens = inputs[i].cpu().numpy()
            e11.append(np.argwhere(tokens == 30522)[0][0])
            e21.append(np.argwhere(tokens == 30524)[0][0])

        tokens_output = self.encoder(inputs)[0] # [B,N] --> [B,N,H]
        output = []

        for i in range(len(e11)):
            instance_output = torch.index_select(tokens_output, 0, torch.tensor(i).cuda())
            instance_output = torch.index_select(instance_output, 1, torch.tensor([e11[i], e21[i]]).cuda())
            output.append(instance_output)  # [B,N] --> [B,2,H]

        output = torch.cat(output, dim=0)
        output = output.view(output.size()[0], -1)  # [B,N] --> [B,H*2]

        return output


class Bert_LoRa(nn.Module):
    
    def __init__(self, config, is_lora=True):
        super(Bert_LoRa, self).__init__()

        # get config
        self.bert_config = BertConfig.from_pretrained(config.bert_path)

        # get output size
        self.output_size = config.encoder_output_size

        # get model
        self.encoder = BertModel.from_pretrained(config.bert_path).cuda()
        
        
        if is_lora == True:
            configs = LoraConfig(
                r=config.rank_lora, 
                lora_alpha=config.rank_lora * 2, 
                target_modules=["query", "value"],
                lora_dropout=0.05, 
            )
            self.encoder = get_peft_model(self.encoder, configs)
                       
        self.drop = nn.Dropout(config.drop_out)

        if config.pattern in ['standard', 'entity_marker']:
            self.pattern = config.pattern
        else:
            raise Exception('Wrong encoding.')
        config.hidden_size = self.bert_config.hidden_size
        config.output_size = config.encoder_output_size
        self.encoder.resize_token_embeddings(config.vocab_size + config.marker_size)


    def save_lora(self, save_path):
        self.encoder.save_pretrained(save_path)
    

    def forward(self, inputs):
        e11 = []
        e21 = []

        for i in range(inputs.size()[0]):
            try:
                tokens = inputs[i].cpu().numpy()
                e11.append(np.argwhere(tokens == 30522)[0][0])
                e21.append(np.argwhere(tokens == 30524)[0][0])
            except:
                print(inputs[i])

        tokens_output = self.encoder(inputs)[0] # [B,N] --> [B,N,H]
        output = []

        for i in range(len(e11)):
            instance_output = torch.index_select(tokens_output, 0, torch.tensor(i).cuda())
            instance_output = torch.index_select(instance_output, 1, torch.tensor([e11[i], e21[i]]).cuda())
            output.append(instance_output)  # [B,N] --> [B,2,H]

        output = torch.cat(output, dim=0)
        output = output.view(output.size()[0], -1)  # [B,N] --> [B,H*2]

        return output


    
    
class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)
            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1) 
            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 0.5
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs


class Dropout_Layer(base_model):
    def __init__(self, config):
        super(Dropout_Layer, self).__init__()
        self.drop = nn.Dropout(config.drop_out)
        self.linear_transform = nn.Linear(config.hidden_size * 2, config.output_size, bias=True)
        self.layer_normalization = nn.LayerNorm([config.output_size])

    def forward(self, input):
        output = self.drop(input)
        output1 = self.linear_transform(output)
        output = F.gelu(output1)
        output = self.layer_normalization(output)
        return output, output1



class Softmax_Layer(base_model):
    def __init__(self, input_size, num_class):
        super(Softmax_Layer, self).__init__()
        self.input_size = input_size
        self.num_class = num_class
        self.fc = nn.Linear(self.input_size, self.num_class, bias=False)

    def forward(self, input):
        logits = self.fc(input)
        return logits



class Dropout_Layer(base_model):
    def __init__(self, config):
        super(Dropout_Layer, self).__init__()
        self.drop = nn.Dropout(config.drop_out)
        self.linear_transform = nn.Linear(config.hidden_size * 2, config.output_size, bias=True)
        self.layer_normalization = nn.LayerNorm([config.output_size])

    def forward(self, input):
        output = self.drop(input)
        output = self.linear_transform(output)
        output = F.gelu(output)
        output = self.layer_normalization(output)
        return output


class Classifier_Layer(base_model):
    def __init__(self, config, num_class):
        super(Classifier_Layer, self).__init__()
        
        self.num_class = num_class
        
        self.head = nn.Sequential(
            nn.Dropout(config.drop_out),
            nn.Linear(config.hidden_size * 2, config.hidden_size, bias=True),
            nn.GELU(),
            nn.LayerNorm([config.hidden_size]),
            nn.Linear(config.hidden_size, self.num_class),
        )
        
        # self.drop = nn.Dropout(config.drop_out)
        # self.linear_transform = nn.Linear(config.hidden_size * 2, config.encoder_output_size, bias=True)
        # self.layer_normalization = nn.LayerNorm([config.encoder_output_size])
        # self.fc = nn.Linear(config.encoder_output_size, self.num_class, bias=True)
        
        
    def forward(self, input):
        # output = self.drop(input)
        # output = self.linear_transform(output)
        # output = F.gelu(output)
        # output = self.layer_normalization(output)
        # logits = self.fc(output)
        
        return self.head(input)


