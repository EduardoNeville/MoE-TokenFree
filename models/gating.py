# Based on https://github.com/uclaml/MoE/blob/main/MoE_Visualization.ipynb

import torch
import torch.nn as nn
from torch.nn import functional as F
from models import model_types
from dataclasses import dataclass, field

from abc import ABC, abstractmethod

def gumbel_noise(n,k):
    unif = torch.distributions.Uniform(0,1).sample((n,k))
    g = -torch.log(-torch.log(unif))
    return g

class Gate(ABC):
    def __init__(self, device):
        super(Gate, self).__init__()
        self.device = device

    @abstractmethod
    def compute(self, x):
        """
        Compute the output of the gate.
        This method should be implemented by all subclasses.
        """
        pass

    def perturbation(self, shape, noise_type = model_types.NORMAL):
        """
        Noise perturbation required for gradient computation
        """
        if noise_type != model_types.NORMAL:
            raise ValueError(f"Noise type ({noise_type}) is not supported.")
        return torch.zeros(shape).to(self.device)
    
class SoftmaxGate(Gate):
    def __init__(self, device, t_func = lambda: 1.0):
        super(SoftmaxGate, self).__init__(device)
        self.device = device
        self.t_func = t_func

    def compute(self, x):
        gate_score = torch.nn.functional.softmax(x / self.t_func() , dim=-1)
        combine_tensor = gate_score[..., None, None]
        index = torch.arange(x.shape[1]).repeat(x.shape[0], 1)
        return combine_tensor, index.long(), gate_score

class TopKGate(Gate):
    def __init__(self, device, straight_through, k=1):
        super(TopKGate, self).__init__(device)
        self.k = k
        self.device = device
        self.straight_through = straight_through

    def compute(self, x):
        if self.k > 1:
            topk_gate_scores, indices = torch.topk(
                x, self.k)
            topk_gate_scores =F.softmax(
                topk_gate_scores,
                dim=1,
                dtype=torch.float,
            ).type_as(x)
            mask = F.one_hot(indices, x.shape[-1]).float()
            mask_flat = mask.sum(dim=-1)
            combine_tensor = (topk_gate_scores[..., None, None, None] * mask_flat[..., None, None, None] * F.one_hot(indices, x.shape[-1])[..., None, None])
            combine_tensor = combine_tensor.sum(1)
            return combine_tensor, indices, topk_gate_scores
        elif self.k == 1:
            x = F.softmax(x, dim=-1)
            topk_gate_scores, index = x.topk(k=self.k, dim=-1) # torch.nn.functional.softmax(x , dim=-1).topk(k=self.k, dim=-1)
            if self.straight_through:
                index_soft = F.softmax(x, dim=-1)
                index = (index - index_soft).detach() + index_soft
                index = index[:,0]
                topk_gate_scores, index = map(lambda x: x.squeeze(dim=-1), (topk_gate_scores, index))
            else:
                topk_gate_scores, index = map(lambda x: x.squeeze(dim=-1), (topk_gate_scores, index))
            
            mask = F.one_hot(index, x.shape[-1]).float()
            mask_flat = mask.sum(dim=-1)
            combine_tensor = (topk_gate_scores[..., None, None, None] * mask_flat[..., None, None, None] * F.one_hot(index, x.shape[-1])[..., None, None])
            return combine_tensor, index, topk_gate_scores
   

    def perturbation(self, shape, noise_type):
        """
        Noise perturbation required for gradient computation
        """
        if noise_type == model_types.NORMAL:
            return torch.rand(shape).to(self.device)
        elif noise_type == model_types.GUMBEL:
            '''
            using gumbel max trick
            '''
            return gumbel_noise(shape[0], shape[1]).to(self.device)
    
class AbstractRouter(nn.Module):
    def __init__(self, gate, is_tracking, noise_type = model_types.NORMAL):
        super(AbstractRouter, self).__init__()
        self.is_tracking = is_tracking
        self.expert_activation = None
        self.gate = gate
        self.noise_type = noise_type

    @abstractmethod
    def _forward(self, x):
        """
        Compute the output of the gate.
        This method should be implemented by all subclasses.
        """
        pass
    
    def forward(self, x):
        expert_scores = self._forward(x)
        combine_tensor, indices, topk_gate_scores = self.gate.compute(expert_scores + self.gate.perturbation(expert_scores.shape, self.noise_type))
        if self.is_tracking:
            self._track_expert_activations(indices, expert_scores.size(1))
        return combine_tensor, indices, topk_gate_scores
    
    def _track_expert_activations(self, indices, num_experts):
        num_used_experts = 1 if indices.dim() == 1 else indices.size(1)
        if self.expert_activation is None:
            self.expert_activation = torch.zeros((num_experts,), device=indices.device)
        for j in range(num_used_experts):
            index = indices if indices.dim() == 1 else indices[:, j]
            self.expert_activation[:].scatter_add_(0, index, torch.ones_like(index).float())
            

    def reset_expert_activations(self):
        self.expert_activation = None

    def expert_activations(self):
        return self.expert_activation

class Router(AbstractRouter):
    def __init__(self, input_dim, out_dim, gate, is_tracking=True, noise_type = model_types.NORMAL):
        super().__init__(gate, is_tracking,noise_type)
        self.linear = nn.Linear(input_dim, out_dim, bias=False)
        self.activation = nn.ReLU(out_dim)
        self.out_dim = out_dim
        self.expert_activation = None
        self.is_tracking = is_tracking
    
    def _forward(self, x):
        out = self.linear(x)
        return out.mean(dim=1)

class RandomRouter(AbstractRouter):
    def __init__(self, input_dim, out_dim, gate, is_tracking=True, noise_type = model_types.NORMAL, router_depth = 1):
        super(RandomRouter, self).__init__(gate, is_tracking, noise_type)
        self.l1 = nn.Linear(input_dim, out_dim, bias=False)
        if router_depth != 1:
            raise ValueError("Only router depth 1 supported")
        for param in self.l1.parameters():
            param.requires_grad = False
    
    def _forward(self, x):
        out = self.l1(x)
        return out.mean(dim=1)
    
class RandomReinitRouter(AbstractRouter):
    def __init__(self, input_dim, out_dim, gate, is_tracking=True, noise_type = model_types.NORMAL, router_depth = 1):
        super(RandomReinitRouter, self).__init__(gate, is_tracking, noise_type)
        self.l1 = nn.Linear(input_dim, out_dim, bias=False)
        if router_depth != 1:
            raise ValueError("Only router depth 1 supported")
        for param in self.l1.parameters():
            param.requires_grad = False
    
    def _forward(self, x):
        self.l1.reset_parameters()
        out = self.l1(x)
        return out.mean(dim=1)
    
class CachedRouter(AbstractRouter):
    def __init__(self, router):
        super().__init__(router.gate, router.is_tracking, router.noise_type)
        self.router = router
        self.state = {}
        self.ROUTING_KEY = 'routing'

    def init(self, idx, labels):
        self.router.init(idx, labels)
        self.state.clear()
    
    def _forward(self, x):
        out = self.router.l3(x)
        return out.mean(dim=1)

    def forward(self, x): 
        if self.ROUTING_KEY in self.state:
            combine_tensor, index, topk_gate_scores = self.state[self.ROUTING_KEY]
            return combine_tensor, index, topk_gate_scores
        combine_tensor, index,  topk_gate_scores = self.router.forward(x)
        self.state[self.ROUTING_KEY] = (combine_tensor, index, topk_gate_scores)
        return combine_tensor, index, topk_gate_scores

class MoE(nn.Module):
    def __init__(self, config, router, expert_module_factory):
        super(MoE, self).__init__()
        self.models = nn.ModuleList()
        self.expert_num = config.expert_num
        self.is_per_token = config.is_per_token
        for _ in range(config.expert_num):
            self.models.append(expert_module_factory(config))
        if not self.is_per_token:
            self.router = router
        else:
            self.per_token_router = nn.Linear(config.n_embd, config.expert_num, bias=False)
            if config.router_type == model_types.RANDOM or config.router_type == model_types.RANDOM2:
                for param in self.per_token_router.parameters():
                    param.requires_grad = False
        self.config = config
        self.expert_activation = None
        self.noise_type = config.noise_type
        self.device = config.device

    def _track_expert_activations(self, indices, num_experts):
        num_used_experts = 1 if indices.dim() == 1 else indices.size(1)
        if self.expert_activation is None:
            self.expert_activation = torch.zeros((num_experts,), device=indices.device)
        for j in range(num_used_experts):
            index = indices if indices.dim() == 1 else indices[:, j]
            self.expert_activation[:].scatter_add_(0, index, torch.ones_like(index).float())

    def perturbation(self, shape):
        """
        Noise perturbation required for gradient computation
        """
        if self.noise_type == model_types.NORMAL:
            return torch.rand(shape).to(self.device)
        elif self.noise_type == model_types.GUMBEL:
            '''
            using gumbel max trick
            '''
            return gumbel_noise(shape[0], shape[1]).to(self.device)

    def forward(self, x):
        if self.expert_num == 1:
            return self.models[0](x)
        
        if self.is_per_token:
            if self.config.router_type == model_types.RANDOM2:
                self.per_token_router.reset_parameters()
            x_squashed = x.view(-1, x.shape[-1])
            gate_logits = self.per_token_router(x_squashed)
            gate_logits += self.perturbation(gate_logits.shape)
            if self.config.topk_exp == 1:
                gate_logits = nn.functional.softmax(gate_logits,dim=1)
            weights, selected_experts = torch.topk(
                gate_logits, self.config.topk_exp
            )
            self._track_expert_activations(selected_experts, self.expert_num)
            if self.config.topk_exp > 1:
                weights = nn.functional.softmax(
                    weights,
                    dim=1,
                    dtype=torch.float,
                ).type_as(x)
            results = torch.zeros_like(x_squashed)
            for i, expert in enumerate(self.models):
                batch_idx, nth_expert = torch.where(selected_experts == i)
                results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                    x_squashed[batch_idx]
                )
            return results.view_as(x)

        else:
            combine_tensor, _, _ = self.router(x)
            
            dispatch_tensor = combine_tensor.bool().to(combine_tensor)
            expert_inputs = torch.einsum('bsn,beij->ebsn', x, dispatch_tensor)

            output = []
            for i in range(self.expert_num):
                output.append(self.models[i](expert_inputs[i]))

            output = torch.stack(output)            
            output = torch.einsum('beij,ebsn->bsn', combine_tensor, output)
            return output

    def get_load_balancing_loss(self, x):
        select, index = None, None
        if self.is_per_token:
            x_squashed = x.view(-1, x.shape[-1])
            gate_logits = self.per_token_router(x_squashed)
            if self.config.topk_exp == 1:
                gate_logits = nn.functional.softmax(gate_logits,dim=1)
            weights, index = torch.topk(
                gate_logits, self.config.topk_exp
            )
            weights_ = torch.zeros_like(gate_logits)
            weights_.scatter_(1, index, weights)
            if self.config.topk_exp >1:
                select = nn.functional.softmax(
                    weights_,
                    dim=1,
                    dtype=torch.float,
                ).type_as(x)
            else:
                select = weights_
        else:
            _, index, _ = self.router(x)
            select = F.softmax(self.router._forward(x), dim=-1)
        if self.config.topk_exp == 1:
            mask_ = F.one_hot(index.long(), self.expert_num).float()
        else:
            mask_ = F.one_hot(index.long(), self.expert_num).float().mean(1)
        mask_ = mask_.to(self.config.device)
        density = mask_.mean(dim=-2)
        density_proxy = select.mean(dim=-2)
        load_balancing_loss = (density_proxy * density).mean() * float(self.expert_num ** 2)
        return load_balancing_loss

    def reset_expert_activations(self):
        self.expert_activation = None

    def expert_activations(self):
        return self.expert_activation