import sys
sys.path.insert(0, '..')
from models.model import GPT, GPTConfig
from models import model_types
import numpy as np

init_from = 'scratch'
# out_dir = 'out-wikitext-gpt-1705953648.4246285'
out_dir = 'out-wikitext-gpt-1705953648.4246285'
eval_interval = 1
eval_iters = 40
wandb_log = True
always_save_checkpoint = False
batch_size = 8
gradient_accumulation_steps = 16
max_iters = 10000
learning_rate = 0.00096
min_lr = 9.6e-05
weight_decay = 0.5
decay_lr = True
device = 'cuda'
compile = False
compute_grad_memory = True
attention_type = 'moe'
mlp_type = 'moe'
router_type = 'random2'
lin_type = 'standard'
lora_rank = 0
lora_alpha = 32.0
expert_num = 4
base_seed_offset = 2
noise_type = 'normal'
load_balancing = False
moe_lin_type = 'standard'
straight_through = False
global_routing = True
router_lr_scaling = 1.0
topk_exp = 2
is_cached = True
router_depth = 1
block_size = 1024

dropout = 0.1
lora_dropout = 0.0
load_balancing_lambda = 0.001
is_per_token = False

model_args = dict(
        dropout=dropout,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        batch_size=batch_size,
        device=device,
        attention_type=attention_type,
        mlp_type=mlp_type,
        router_type=router_type,
        lin_type=lin_type,
        expert_num=expert_num,
        moe_lin_type=moe_lin_type,
        gating_type=model_types.TOPK,
        is_cached=is_cached,
        noise_type=noise_type,
        load_balancing=load_balancing,
        load_balancing_lambda=load_balancing_lambda,
        topk_exp=topk_exp,
        straight_through=straight_through,
        global_routing=global_routing,
        is_per_token=is_per_token,
        router_lr_scaling=router_lr_scaling,
        n_layer=12,
        n_head=12,
        n_embd=768,
        router_depth=router_depth,
    )

import os
import torch
device = 'cuda'
out_dir = 'out-wikitext-gpt-1705953648.4246285'
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
checkpoint_model_args = checkpoint['model_args']
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)

print("========> loaded checkpoint and created model")

state_dict = checkpoint['model']
# fix the keys of the state dictionary :(
# honestly no idea how checkpoints sometimes get this prefix, have to debug more
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
iter_num = checkpoint['iter_num']
best_val_loss = checkpoint['best_val_loss']

print("========> loaded checkpoint to model")

train_file_names=["train/0/de.bin", "train/0/en.bin", "train/0/fr.bin", "train/0/it.bin"]
val_file_names=["test/0/de.bin", "test/0/en.bin", "test/0/fr.bin", "test/0/it.bin"]
data_dir='/scratch/homes/bmessmer/data/ec-llm/multilingual_wiki_test/multilingual_wiki_test'

print("========> dataset: ", os.path.join(data_dir, train_file_names[0]))
train_data = np.memmap(os.path.join(data_dir, train_file_names[0]), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, val_file_names[0]), dtype=np.uint16, mode='r')
print("====> single language length:", len(train_data), len(val_data))

def get_batch(split, label_indices = {}):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

print("========> loaded dataset")

model.eval()
model = model.to(device)
# model = torch.compile(model)
X, _ = get_batch('train')

print("========> running inference")
y = model(X)
print("========> DONE")