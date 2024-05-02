import argparse
import os
from pathlib import Path
import subprocess
import time
from argparse import Namespace
import pathlib
import sys
 # setting path
sys.path.append(pathlib.Path().parent.absolute())

# Base shell command
shell_cmd = r"""python train.py \
--dataset='wikitexts' \
--init_from='scratch' \
--out_dir='out-wikitexts-gpt-'$wandb_name \
--eval_interval=5 \
--eval_iters=40 \
--is_cached=False \
--wandb_log=True \
--wandb_project='ec-llm' \
--wandb_run_name=$wandb_name \
--always_save_checkpoint=True \
--batch_size=8 \
--gradient_accumulation_steps=16 \
--max_iters=$num_iters \
--lr_decay_iters=$num_iters \
--warmup_iters=$warmup_iters \
--learning_rate=$lr \
--min_lr=$min_lr \
--weight_decay=$wd \
--decay_lr=$decay_lr \
--device='cuda' \
--compile=True \
--compute_grad_memory=True \
--mlp_type='moe' \
--router_type='standard' \
--lin_type='standard' \
--lora_rank=0 \
--lora_alpha=32.0 \
--expert_num=4 \
--base_seed_offset=$seed \
--noise_type='normal' \
--load_balancing=True \
--moe_lin_type='standard' \
--straight_through=False \
--global_routing=False \
--topk_exp=2 \
--is_per_token=True \
--router_lr_scaling=0.0 \
--router_depth=1 \
--load_balancing_lambda=0.01 \
"""

min_lk=6e-5
wd = 0.5
base_lr = 3e-5
min_lr=6e-5
learning_rate=6e-4
weight_decay=1e-1
decay_lr=True

# Configure runs
def main():
    base_dir = os.getcwd()
    # seeds = [ 0, 1, 2 ]
    # lr_factors = [ 8, 4, 2, 0.5, 0.25, 0.125]
    # num_iters = [ 80 ]
    # seeds = [0, 1, 2]
    # lr_factors = [16, 32]
    # num_iters = [ 200 ]
    seeds = [2]
    #lr_factors = [32]
    num_iters = [ 5616 ]

    idx = 0
    for seed in seeds:
        for num_iter in num_iters:
            #for factor in lr_factors:
            #    lr = factor * base_lr
            if do_run(idx):
                print(f"===== Running iteration={idx} {learning_rate=:0.3e} {weight_decay=:0.3e} =====")
                current_time = str(time.time())
                min_lr = learning_rate/10
                subprocess.run(
                    ['/bin/bash', '-c', f"echo {shell_cmd} \n {shell_cmd}"],
                    env={k: str(v) for k, v in dict(
                        # The keys and values need to be strings
                        **os.environ,  # Inherit original variables (e.g. conda)
                        wandb_name=f'wikitexts_exp_num_4_topk_exp_2_num_iter_{num_iter}_lr_{learning_rate}_wd_{weight_decay}_seed_{seed}_ts_{current_time}_{idx}',
                        num_iters=num_iter,
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
