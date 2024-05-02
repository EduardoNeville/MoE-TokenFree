import argparse
import os
from pathlib import Path
import subprocess
import time
from argparse import Namespace


# Base shell command
shell_cmd = r"""python train.py \
--train_file_names='["train/0/de.bin", "train/0/en.bin", "train/0/fr.bin"]' \
--val_file_names='["test/0/de.bin", "test/0/en.bin", "test/0/fr.bin"]' \
--data_dir='/mloscratch/homes/neville/MoE-TokenFree/data/multilingual_wiki/multilingual_wiki_test' \
--init_from='scratch' \
--out_dir=$out_put \
--eval_interval=1 \
--eval_iters=40 \
--wandb_log=True \
--wandb_project='ec-llm' \
--wandb_run_name=$wandb_name \
--always_save_checkpoint=False \
--batch_size=8 \
--gradient_accumulation_steps=16 \
--max_iters=$num_iters \
--learning_rate=$lr \
--min_lr=$min_lr \
--weight_decay=$wd \
--decay_lr=$decay_lr \
--device='cuda' \
--compile=False \
--compute_grad_memory=True \
--attention_type='moe' \
--mlp_type='moe' \
--router_type='random2' \
--lin_type='standard' \
--lora_rank=0 \
--lora_alpha=32.0 \
--expert_num=4 \
--base_seed_offset=$seed \
--noise_type='normal' \
--load_balancing=False \
--moe_lin_type='standard' \
--straight_through=False \
--global_routing=True \
--router_lr_scaling=1.0 \
--topk_exp=2 \
--is_cached=True \
--router_depth=1 \
"""

wd = 0.5
base_lr = 3e-5


# Configure runs
def main():
    base_dir = os.getcwd()
    seeds = [2]
    lr_factors = [ 32 ]
    num_iters = [ 10000 ]

    idx = 0
    for seed in seeds:
        for num_iter in num_iters:
            for factor in lr_factors:
                lr = factor * base_lr
                if do_run(idx):
                    print(f"===== Running iteration={idx} {lr=:0.3e} {wd=:0.3e} =====")
                    min_lr = lr/10
                    current_time = time.time()
                    subprocess.run(
                        ['/bin/bash', '-c', f"echo {shell_cmd} \n {shell_cmd}"],
                        env={k: str(v) for k, v in dict(
                            # The keys and values need to be strings
                            **os.environ,  # Inherit original variables (e.g. conda)
                            wandb_name=f'validate_4exp_per_layer_random2_routing_top2',
                            num_iters=num_iter,
                            lr=f"{lr:0.6}",
                            min_lr=f"{min_lr:0.6}",
                            wd=f"{wd:0.6}",
                            iter=idx,
                            seed=seed,
                            decay_lr=True,
                            out_put=f'out-wikitext-gpt-{current_time}'
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
