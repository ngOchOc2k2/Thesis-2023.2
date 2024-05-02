import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import AutoModel, AutoConfig
from transformers.file_utils import ModelOutput
import torch.nn.functional as F
import numpy as np
import os

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None
    
@dataclass
class EncoderOutputCustom(ModelOutput):
    loss: Optional[Tensor] = None



class BiEncoderModelCustom(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 model_name: str = None,
                 normlized: bool = False,
                 sentence_pooling_method: str = 'cls',
                 negatives_cross_device: bool = False,
                 temperature: float = 1.0,
                 use_inbatch_neg: bool = True,
                 vocab_size = 250002,
                 ):
        super().__init__()
        self.vocab_size = vocab_size
        self.model_config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        if model_name == 'BAAI/bge-m3':
            self.model.resize_token_embeddings(self.vocab_size + 4)
            
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.normlized = normlized
        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        self.use_inbatch_neg = use_inbatch_neg
        self.config = self.model.config

        self.head = nn.Sequential(
            nn.Linear(self.model_config.hidden_size * 2, self.model_config.hidden_size, bias=True),
        )
        if model_name != 'BAAI/bge-m3':
            self.head.load_state_dict(torch.load(os.path.join(model_name, "head.pth")))
        
        if not normlized:
            self.temperature = 1.0
            logger.info("reset temperature = 1.0 due to using inner product to compute similarity")

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')

            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]


    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))


    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors
    
    
    def encode(self, features, query_mode=False):
        if features is None:
            return None
        
        if query_mode == False:
            psg_out = self.model(**features, return_dict=True)
            p_reps = self.sentence_embedding(psg_out.last_hidden_state, features['attention_mask'])
            if self.normlized:
                p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
            return p_reps.contiguous()

        elif query_mode == True:
            tokens = features['input_ids'].cpu().numpy()
            e11 = np.argwhere(tokens == 250003)[:, 0]
            e21 = np.argwhere(tokens == 250005)[:, 0]

            token_output = self.model(**features, return_dict=True)["last_hidden_state"]
            token_output = self.sentence_embedding(token_output, features['attention_mask'])

            query_reps = []
            for e11_idx, e21_idx in zip(e11, e21):
                instance_output = torch.cat((token_output[e11_idx], token_output[e21_idx]), dim=0)  # Concatenate hai biểu diễn
                query_reps.append(instance_output)
            
            
            query_reps = torch.stack(query_reps)
            if self.normlized:
                query_reps = torch.nn.functional.normalize(query_reps, dim=-1)
                
            return self.head(query_reps.contiguous())
          

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, teacher_score: Tensor = None):
        q_reps = self.encode(query, query_mode=False)
        p_reps = self.encode(passage)

        if self.training:
            if self.negatives_cross_device and self.use_inbatch_neg:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            group_size = p_reps.size(0) // q_reps.size(0)
            if self.use_inbatch_neg:
                scores = self.compute_similarity(q_reps, p_reps) / self.temperature # B B*G
                scores = scores.view(q_reps.size(0), -1)

                
                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                target = target * group_size
                loss = self.compute_loss(scores, target)
            else:
                scores = self.compute_similarity(q_reps[:, None, :,], p_reps.view(q_reps.size(0), group_size, -1)).squeeze(1) / self.temperature # B G

                scores = scores.view(q_reps.size(0), -1)
                target = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
                loss = self.compute_loss(scores, target)


            # Compute distillation loss
            if teacher_score is not None:
                distillation_loss = self.compute_distillation_loss(scores, teacher_score)
                loss += distillation_loss
                
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
            
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
                 v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)
        torch.save(self.head.state_dict(), os.path.join(output_dir, "head.pth"))


    def compute_distillation_loss(self, student_scores, teacher_scores):
        return nn.KLDivLoss()(torch.log_softmax(student_scores, dim=-1), torch.softmax(teacher_scores, dim=-1))



class DistilationModelCustom(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 model_name: str = None,
                 model_teacher: str = None,
                 normlized: bool = False,
                 sentence_pooling_method: str = 'cls',
                 negatives_cross_device: bool = False,
                 temperature: float = 1.0,
                 use_inbatch_neg: bool = True,
                 distil_loss=True,
                 ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.teacher = AutoModel.from_pretrained(model_teacher)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.distil_loss = distil_loss

        for k, v in self.teacher.named_parameters():
            v.requires_grad = False

        self.normlized = normlized
        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        self.use_inbatch_neg = use_inbatch_neg
        self.config = self.model.config

        self.head = nn.Sequential(
            nn.Linear(self.model_config.hidden_size * 2, self.model_config.hidden_size, bias=True),
        )
        if model_name != 'BAAI/bge-m3':
            self.head.load_state_dict(torch.load(os.path.join(model_name, "head.pth")))
            
        if not normlized:
            self.temperature = 1.0
            logger.info("reset temperature = 1.0 due to using inner product to compute similarity")

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')

            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]


    def encode(self, features, model, query_mode=False):
        if features is None:
            return None
        
        if query_mode == False:
            psg_out = model(**features, return_dict=True)
            p_reps = self.sentence_embedding(psg_out.last_hidden_state, features['attention_mask'])
            if self.normlized:
                p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
            return p_reps.contiguous()

        elif query_mode == True:
            tokens = features['input_ids'].cpu().numpy()
            e11 = np.argwhere(tokens == 250003)[:, 0]
            e21 = np.argwhere(tokens == 250005)[:, 0]

            token_output = model(**features, return_dict=True)["last_hidden_state"]
            token_output = self.sentence_embedding(token_output, features['attention_mask'])

            query_reps = []
            for e11_idx, e21_idx in zip(e11, e21):
                instance_output = torch.cat((token_output[e11_idx], token_output[e21_idx]), dim=0)  # Concatenate hai biểu diễn
                query_reps.append(instance_output)
            
            
            query_reps = torch.stack(query_reps)
            if self.normlized:
                query_reps = torch.nn.functional.normalize(query_reps, dim=-1)
            return self.head(query_reps.contiguous())


    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))


    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, teacher_score: Tensor = None):
        q_reps = self.encode(query, self.model, query_mode=False)
        p_reps = self.encode(passage, self.model)
        q_teach = self.encode(query, self.teacher, query_mode=False)
        p_teach = self.encode(passage, self.teacher)
        loss, loss_cl, loss_kd = 0.0, 0.0, 0.0
        llambda = 0.3

        if self.training:
            if self.negatives_cross_device and self.use_inbatch_neg:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)
                
                q_teach = self._dist_gather_tensor(q_teach)
                p_teach = self._dist_gather_tensor(p_teach)

            group_size = p_reps.size(0) // q_reps.size(0)
            if self.use_inbatch_neg:
                scores = self.compute_similarity(q_reps, p_reps) / self.temperature # B B*G
                scores = scores.view(q_reps.size(0), -1)

                
                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                target = target * group_size
                loss_cl = self.compute_loss(scores, target)
                
                
            else:
                scores = self.compute_similarity(q_reps[:, None, :,], p_reps.view(q_reps.size(0), group_size, -1)).squeeze(1) / self.temperature # B G

                scores = scores.view(q_reps.size(0), -1)
                target = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
                loss_cl = self.compute_loss(scores, target)


            scores_teach = self.compute_similarity(q_teach, p_teach) / self.temperature # B B*G
            scores_teach = scores_teach.view(q_teach.size(0), -1)

            loss_kd = self.compute_distillation_loss(scores, scores_teach)
            loss = llambda * loss_kd + (1 - llambda) * loss_cl
    
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
            
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=combined_reps if self.training else p_reps,
        )


    def compute_distillation_loss(self, student_scores, teacher_scores):
        student_scores_norm = F.normalize(student_scores, p=2, dim=-1)
        teacher_scores_norm = F.normalize(teacher_scores, p=2, dim=-1)
        return max(0, (1 - F.cosine_similarity(student_scores_norm, teacher_scores_norm, dim=-1).mean()))


    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
                 v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)
        torch.save(self.head.state_dict(), os.path.join(output_dir, "head.pth"))
