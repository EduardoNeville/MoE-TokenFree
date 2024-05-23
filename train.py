"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from models.model import GPTConfig, GPT
from models.lora import get_lora_model

#from models.gating import get_moe_model

from models import model_types

#from models.scheduler import TemperatureScheduler

from utils.metrics import AverageMeter
import torch._dynamo.config
torch._dynamo.config.cache_size_limit = 256
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import torch._inductor
torch._inductor.config.fallback_random = True
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O

out_dir = 'out'
eval_interval = 500
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'MoE-Tokenization'
wandb_org = 'neville-mlo'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
train_file_names = [ 'train.bin' ]
val_file_names = [ 'val.bin' ]
gradient_accumulation_steps = 5 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0 # for pretraining 0 is good, for finetuning try 0.1+
bias = True # do we use bias inside LayerNorm and Linear layers?
#---------
#vocab_size = 50257  # GPT
vocab_size = 256  # ByT5
#---------

# LoRA params
lora_rank = 0
lora_alpha = 0.0 # set alpha to the first rank which is tried, then keep it fixed, and don't further tune it (see the paper for more info)
lora_dropout = 0.0

compute_grad_memory = False # compute the memory usage of the gradients

# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 5000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster

router_type = model_types.STANDARD
mlp_type = model_types.STANDARD
lin_type = model_types.LINEAR
moe_lin_type = model_types.LORA
gating_type = model_types.TOPK

moe_target_modules = [ "mlp" ]
lora_target_modules = [ "c_attn", "att.c_proj", "mlp.c_fc", "mlp.c_proj" ]

is_cached = True

expert_num = 1 # number of experts
topk_exp = 1 # topk experts to activate
base_seed_offset = 0
noise_type = model_types.GUMBEL

load_balancing = False
load_balancing_lambda = 0.01
straight_through = False
is_per_token = False
router_lr_scaling = 100.0
global_routing = True
router_depth = 1

data_dir = 'data/openwebtext/byt5_tokenization'
temp_scheduler = None

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    gradient_accumulation_steps *= 8 # simulate 8 gpus
print("total number of tokens per iteration:", batch_size * block_size * gradient_accumulation_steps)

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + base_seed_offset + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
if data_dir == '':
    data_dir = os.path.join('data', dataset)

def create_dataset(file_names, data_dir):
    datasets = []
    index_range = {}
    acc = 0
    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)
        file_data = np.memmap(file_path, dtype=np.uint16, mode='r')
        datasets.append(np.array(file_data))  # Convert memmap object to a NumPy array
        acc += len(file_data)
        index_range[file_name] = acc
    concatenated_data = np.concatenate(datasets, axis=0)  # Concatenate along the desired axis
    return concatenated_data, sorted(index_range.items(), key=lambda x: x[1])

def to_label(i, sorted_index_range):
    previous_max = 0
    label_idx = 0
    for _, cumulative_index in sorted_index_range:
        if previous_max <= i < cumulative_index:
            return label_idx
        previous_max = cumulative_index
        label_idx += 1
    return None

label_indices = {}
if len(train_file_names) == 1:
    print("========> dataset: ", os.path.join(data_dir, train_file_names[0]))
    train_data = np.memmap(os.path.join(data_dir, train_file_names[0]), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, val_file_names[0]), dtype=np.uint16, mode='r')
    print("====> single language length:", len(train_data), len(val_data))
else:
    train_data, train_indices = create_dataset(train_file_names, data_dir)
    val_data, test_indices = create_dataset(val_file_names, data_dir)
    label_indices['val'] = test_indices
    label_indices['train'] = train_indices
    print("====> multilingual length:", len(train_data), len(val_data))

def get_batch(split, label_indices = {}):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    if len(label_indices) == 0:
        return x, y, None
    labels = torch.tensor([to_label(x, label_indices[split]) for x in ix])
    return x, y, labels

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')

#meta_vocab_size = 50257 # GPT
meta_vocab_size = 256 # ByT5
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
# Note: we only want to do LoRA fine-tuning when we resume or start with a pretrained model and NOT when we start from scratch
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, bias=bias, vocab_size=vocab_size, dropout=dropout) # start with model_args from command line

print("========> LORA RANK: ", lora_rank)
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")

    # Removed 
    # attention_type=attention_type
    # batch_size = batch_size
    #mlp_type=mlp_type,
    #lin_type=lin_type,
    #moe_lin_type=moe_lin_type,
    #is_cached=is_cached,
    #global_routing=global_routing,
    #router_lr_scaling=router_lr_scaling,
    #vocab_size=vocab_size,
    #router_depth=router_depth,
    model_args = dict(
        dropout=dropout,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        device=device,
        router_type=router_type,
        expert_num=expert_num,
        gating_type=gating_type,
        noise_type=noise_type,
        load_balancing=load_balancing,
        load_balancing_lambda=load_balancing_lambda,
        topk_exp=topk_exp,
        straight_through=straight_through,
        is_per_token=is_per_token,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=bias,
        lora_target_modules=lora_target_modules,
        moe_target_modules=moe_target_modules
    )

    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    model_args['device'] = device

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['dropout',
                'lora_rank',
                'lora_alpha',
                'lora_dropout',
                'batch_size',
                'device',
                'attention_type',
                'mlp_type',
                'router_type',
                'lin_type',
                'expert_num',
                'moe_lin_type',
                'gating_type',
                'is_cached',
                'noise_type',
                'load_balancing',
                'topk_exp',
                'straight_through',
                'global_routing',
                'is_per_token',
                'router_lr_scaling',
                'n_head',
                'n_layer',
                'n_embd',
                'block_size',
                'vocab_size',
                'device']:
        model_args[k] = checkpoint_model_args.get(k, 0)
    # create the model

    # LoRA fine-tuning?    
    if lora_rank > 0:
        model_args['lora_rank'] = lora_rank
        model_args['lora_alpha'] = lora_alpha
        model_args['lora_dropout'] = lora_dropout

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
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

    if lora_rank > 0:
        # Only make LoRA weights tunable
        print("Marking model as LoRA fine-tunable...")
        model = get_lora_model(model)
        print("Done.")

elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    #temp_scheduler=TemperatureScheduler(1000, 18, 0.1)

    # Removed attention_type=attention_type
    override_args = dict(
        dropout=dropout,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        device=device,
        router_type=router_type,
        expert_num=expert_num,
        gating_type=gating_type,
        noise_type=noise_type,
        load_balancing=load_balancing,
        load_balancing_lambda=load_balancing_lambda,
        topk_exp=topk_exp,
        straight_through=straight_through,
        is_per_token=is_per_token,
        lora_target_modules=lora_target_modules,
        moe_target_modules=moe_target_modules
    )
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'lora_rank', 'lora_alpha']:
        model_args[k] = getattr(model.config, k)
    

    if lora_rank > 0:
        # Only make LoRA weights tunable
        print("Marking model as LoRA fine-tunable...")
        model = get_lora_model(model)
        print("Done.")
    # elif expert_num > 1:
    #     print("Marking model as Moe fine-tunable...")
    #     model = get_moe_model(model)
    #     print("Done.")

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    class_out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        class_losses = {}
        for k in range(eval_iters):
            X, Y, L = get_batch(split, label_indices)
            _, loss_for_reporting = model(X, targets=Y)
            losses[k] = loss_for_reporting
        out[split] = losses.mean()
        class_out[split] = class_losses
    model.train()
    return out, class_out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def reset_expert_activations(network):
    layers = [
        module
        for module in network.modules()
        if not isinstance(module, torch.nn.Sequential)
    ]
    expert_activation_layers = [
        layer.reset_expert_activations()
        for layer in layers
        if getattr(layer, "reset_expert_activations", None)
        and layer.reset_expert_activations() is not None
    ]
    for expert_activation_layer in expert_activation_layers:
        expert_activation_layer.reset_expert_activations()

def expert_activations(network):
    layers = [
        module
        for module in network.modules()
        if not isinstance(module, torch.nn.Sequential)
    ]
    expert_activation_layers = [
        layer.expert_activations()
        for layer in layers
        if getattr(layer, "expert_activations", None)
        and layer.expert_activations() is not None
    ]
    return expert_activation_layers


model_params = sum(p.numel() for p in model.parameters())
model_opt_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"====> Model Params: {model_params}")
print(f"----> Model optimized Params: {model_opt_params}")
print(f"===> MODELS: {model}")
print(f"----> Vocabulary size: {vocab_size}")
print(f"====> frozen: ", [ n for n, p in model.named_parameters() if not p.requires_grad ])

# logging

if wandb_log and master_process:
    import wandb
    wand_args = config
    wand_args['model_params'] = model_params
    wand_args['model_opt_params'] = model_opt_params
    wandb.init(project=wandb_project, entity=wandb_org, name=wandb_run_name, config=config, )


# training loop
batch_time_m = AverageMeter()
data_time_m = AverageMeter()
global_time_m = AverageMeter()
eval_time_m = AverageMeter()

t0 = time.time()
X, Y, L = get_batch('train', label_indices) # fetch the very first batch
data_time_m.update(time.time() - t0)
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
current_params = {}
previous_params = {}
grad_magnitude = {}
tracked_parameters = [name for name, param in model.named_parameters() if param.requires_grad and ("router" in name and "weight" in name and param.shape[0] == expert_num) ]
print("=============> ", tracked_parameters)
print('=============> num of param groups:', len(optimizer.param_groups))

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for idx, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr
        if idx == 2:
            param_group['lr'] = router_lr_scaling*lr

    print('==========> learning rate checking:')
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"Group {i + 1}: Learning Rate = {param_group['lr']}")

    print('==========> Evaluation of loss on train/val:')
    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        te = time.time()
        losses, class_losses = estimate_loss()

        print(f"{wandb_log}")
        if wandb_log:
            metrics = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu * 100, # convert to percentage
            }
            print(f"Logging metrics: {metrics}")
            activations = expert_activations(model)
            for idx, activation in enumerate(activations):
                values = (activation / activation.sum()).cpu().numpy()
                bins = list(range(activation.size(0) + 1))
                metrics[f'expert_activations/{idx}'] = wandb.Histogram(values)
            update_angles = None
            if len(current_params) == 0:
                for name in tracked_parameters:
                    metrics[f'relative_update_size/{name}'] = wandb.Histogram(torch.zeros((expert_num,)).numpy())

            for label, class_loss in class_losses['train'].items():
                metrics[f'train/{train_file_names[label]}'] = class_loss
            for label, class_loss in class_losses['val'].items():
                metrics[f'val/{val_file_names[label]}'] = class_loss
            if temp_scheduler is not None and gating_type == model_types.SOFT:
                metrics[f'softmax_temp'] = temp_scheduler.get_temperature()
            wandb.log(metrics)

        eval_time_m.update(time.time() - te)
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, Time: {eval_time_m.val:.3f} ({eval_time_m.avg:.3f})")
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break
    
   
    previous_params = dict(current_params)
    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    tb = time.time()
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            loss_for_backprop, loss_for_reporting = model(X, targets=Y)
            loss_for_backprop = loss_for_backprop / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        td = time.time()
        X, Y, L = get_batch('train', label_indices)
        data_time_m.update(time.time() - td)

        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss_for_backprop).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()

    if temp_scheduler is not None:
        temp_scheduler.step()

    batch_time_m.update(time.time() - tb)

    if compute_grad_memory:
        # compute the gradient memory usage
        grad_memory = 0
        for p in model.parameters():
            if p.grad is not None:
                grad_memory += p.grad.numel() * p.grad.element_size()
        grad_memory = grad_memory / 1024**2
        print(f"grad memory usage: {grad_memory:.2f} MB")
    

    named_params = dict(model.named_parameters())
    for name in tracked_parameters:
        current_params[name] = named_params[name].detach()

        if named_params[name].grad is not None:
            grad = named_params[name].grad.detach()
            neuron_grad_norm = torch.linalg.vector_norm(grad.flatten(1), dim=1)
            grad_magnitude[name] = neuron_grad_norm

    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)
    reset_expert_activations(model)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    global_time_m.update(dt)
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss_for_backprop
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(
            f'Train: [{iter_num:>4d}/{max_iters} ({100 * (iter_num / max_iters):>3.0f}%)]  '
            f'Loss: {lossf:.4g} '
            f'Time: {batch_time_m.val:.3f}s, {X.size(0) / batch_time_m.val:>7.2f}/s  '
            f'({batch_time_m.avg:.3f}s, {X.size(0) / batch_time_m.avg:>7.2f}/s)  '
            f'LR: {lr:.3e}  '
            f'Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})  '
            f'Global: {global_time_m.val:.3f} ({global_time_m.avg:.3f})  '
            f'MFU: {running_mfu*100:.2f}% ')
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
