import argparse
import os
from pathlib import Path
import subprocess
import time

# Base shell command
shell_cmd = r"""python train.py \
--dataset='wikitexts' \
--init_from='gpt2' \
--out_dir='out-wikitext-gpt-${current_time}' \
--eval_interval=5 \
--eval_iters=40 \
--wandb_log=True \
--wandb_project='ec-llm' \
--wandb_run_name=$wandb_name \
--always_save_checkpoint=False \
--batch_size=8 \
--gradient_accumulation_steps=96 \
--max_iters=$num_iters \
--learning_rate=$lr \
--min_lr=$min_lr \
--weight_decay=$wd \
--decay_lr=$decay_lr \
--device='cuda' \
--compile=True \
--compute_grad_memory=True \
--attention_type='standard' \
--mlp_type='moe' \
--router_type='gpt' \
--lin_type='standard' \
--moe_lin_type='linear' \
--base_seed_offset=$seed
"""

wd = 0.5
base_lr = 3e-5

# Configure runs
def main():
    base_dir = os.getcwd()
    seeds = [ 0, 1, 2 ]
    lr_factors = [ 8, 16, 32, 4, 2, 0.5]
    num_iters = [ 30 ]

    idx = 0
    for seed in seeds:
        for num_iter in num_iters:
            for factor in lr_factors:
                lr = factor * base_lr
                if do_run(idx):
                    print(f"===== Running iteration={idx} {lr=:0.3e} {wd=:0.3e} =====")
                    current_time = str(time.time())
                    min_lr = lr/10
                    subprocess.run(
                        ['/bin/bash', '-c', f"echo {shell_cmd} \n {shell_cmd}"],
                        env={k: str(v) for k, v in dict(
                            # The keys and values need to be strings
                            **os.environ,  # Inherit original variables (e.g. conda)
                            wandb_name=f'ft_refactored_mlp_linear_gpt_moe_lr__{lr}_wd_{wd}_num_iter_{num_iter}_seed_{seed}_ts_{current_time}_{idx}',
                            num_iters=num_iter,
                            lr=f"{lr:0.6}",
                            min_lr=f"{min_lr:0.6}",
                            wd=f"{wd:0.6}",
                            iter=idx,
                            seed=seed,
                            decay_lr=False,
                            current_time=current_time,
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