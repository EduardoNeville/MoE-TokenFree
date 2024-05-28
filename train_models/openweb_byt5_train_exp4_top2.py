import argparse
import os
from pathlib import Path
import subprocess
import time
from argparse import Namespace
import pathlib
import sys

 # setting path
#sys.path.append(pathlib.Path().parent.absolute())

# Base shell command
#change names of train and val dataset
#Removed 
#--attention_type='standard' \
#--out_dir='wikitexts-byt5-std-'$wandb_name \
shell_cmd = r"""python train.py \
--dataset='openwebtext' \
--init_from='scratch' \
--out_dir='openweb-byt5-exp4-top2' \
--eval_interval=10 \
--eval_iters=40 \
--wandb_log=False \
--wandb_project='MoE-Tokenization' \
--wandb_run_name=$wandb_name \
--always_save_checkpoint=False \
--is_cached=False \
--batch_size=8 \
--gradient_accumulation_steps=16 \
--max_iters=$num_iters \
--learning_rate=$lr \
--min_lr=$min_lr \
--weight_decay=$wd \
--decay_lr=$decay_lr \
--device='cuda' \
--compile=True \
--compute_grad_memory=True \
--router_type='standard' \
--lora_rank=0 \
--lora_alpha=32.0 \
--expert_num=4 \
--base_seed_offset=$seed \
--noise_type='normal' \
--load_balancing=True \
--straight_through=False \
--topk_exp=2 \
--is_per_token=True \
--load_balancing_lambda=0.01 \
--vocab_size=$vocab_size \
--data_dir=$data_dir \
"""

min_lk=6e-5
learning_rate=6e-4
min_lr=6e-5
weight_decay=1e-1
decay_lr=True

# Configure runs
def main():
    base_dir = os.getcwd()
    seeds = [ 2 ]
    num_iters = [ 4884 ] 
    # New name for wandb
    train_name = f"openwebtext_byt5_exp4_top2"

    vocab_size = 256
    data_dir = 'data/openwebtext/byt5_tokenization'

    idx = 0
    for seed in seeds:
        for num_iter in num_iters:
            if do_run(idx):
                print(f"===== Running iteration={idx} {learning_rate=:0.3e} {weight_decay=:0.3e} =====")
                current_time = str(time.time())
                subprocess.run(
                    ['/bin/bash', '-c', f"echo {shell_cmd} \n {shell_cmd}"],
                    env={k: str(v) for k, v in dict(
                        # The keys and values need to be strings
                        **os.environ,  # Inherit original variables (e.g. conda)
                        wandb_name=train_name,
                        num_iters=num_iter,
                        vocab_size=vocab_size,
                        data_dir=data_dir,
                        lr=f"{learning_rate:0.6}",
                        min_lr=f"{min_lr:0.6}",
                        wd=f"{weight_decay:0.6}",
                        iter=idx,
                        seed=seed,
                        decay_lr=decay_lr,
                        warmup_iters=int(0.03 * num_iter)
                    ).items()}
            )
            idx += 1

if __name__ == "__main__":
    # Arguments to run a subset of iterations (e.g. for parallelization or infra failures)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--range", nargs=2, type=int, default=None,
        help="Run a subrange of the iterations from A to B (inclusive)",
    )
    parser.add_argument(
        "--iters", nargs='+', type=int, default=None,
        help="Run the provided subset of iterations",
    )
    args = parser.parse_args()

    assert not (args.range and args.iters)
    if args.range:
        do_run = lambda idx: args.range[0] <= idx <= args.range[1]
    elif args.iters:
        do_run = lambda idx: idx in args.iters
    else:
        do_run = lambda idx: True

    main()
