import torch
from .lora import LoRALinear
from .gating import MoE, TopKGate, Router, RandomRouter, SoftmaxGate, RandomReinitRouter, CachedRouter
from .model_types import *
from dataclasses import replace


def _moe_factory(layer_id, config, module_factory):
    if layer_id < config.gated_layer_id:
        return module_factory(config), None
    else:
        gate = _gating_factory(config)()
        router_factory = get_router_factory(config.router_type, config)
        router = router_factory(gate)
        return MoE(config, router, module_factory)

def _gating_factory(config):
    if config.gating_type == model_types.TOPK:
        return lambda: TopKGate(config.device, config.straight_through, config.topk_exp)
    elif config.gating_type == model_types.SOFT:
        return lambda: SoftmaxGate(config.device)
    else:
        raise ValueError("unknown gating type {config.gating_type}")

def get_custom_module_factory(desc, config, module_factory):
    if desc == model_types.STANDARD:
        return lambda _: module_factory(config)
    elif desc == model_types.MOE:
        return lambda layer_id: _moe_factory(layer_id, config, module_factory)
    else:
        raise ValueError(f"unknown model type {desc}")

def get_router_factory(desc, config):
    factories = {
        **dict.fromkeys(
            [model_types.STANDARD, model_types.LINEAR],
            lambda gate: Router(config.n_embd, config.expert_num, gate, noise_type = config.noise_type)),
        model_types.RANDOM: lambda gate: RandomRouter(config.n_embd, config.expert_num, gate, noise_type = config.noise_type),
        model_types.RANDOM2: lambda gate: RandomReinitRouter(config.n_embd, config.expert_num, gate, noise_type = config.noise_type)
    }
    return factories[desc]

def get_linear_factory(desc, config):
    factories = {
        **dict.fromkeys([model_types.STANDARD, model_types.LINEAR], torch.nn.Linear),
        model_types.LORA: lambda in_features, out_features, bias: LoRALinear(
            in_features,
            out_features,
            bias,
            device=config.device,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout)
    }
    return factories[desc]