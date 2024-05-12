import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2Tokenizer
import tiktoken

# Add parent directory to path
current_dir = os.path.dirname(__file__)
grandparent_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(grandparent_dir)
sys.path.append(os.path.abspath(os.path.join(current_dir, 'models')))

from models.model import GPT
import models.model_types as model_types

from utils.crop import crop
from utils.gather_results import gather_results

import sys
import os

choices = ["A", "B", "C", "D"]

gate_type = "TopKBalancedNoisyGate"
gate_network = "mlp"
gate_balance_loss_weight = 1e-2
calculator_type = "UniversalCalculator"
multiply_gate_scores = True
score_scale_factor = 1.0
use_cache = False
lora_rank = 16
lora_alpha = 32
lora_dropout= 0.0
dropout = 0.0
init_from = 'gpt2'
router_type = model_types.STANDARD
gating_type = model_types.TOPK
noise_type = model_types.GUMBEL
load_balancing = False
load_balancing_lambda = 0.01
straight_through = False
is_per_token = False
moe_target_modules = [ "mlp" ]
lora_target_modules = [ "c_attn", "att.c_proj", "mlp.c_fc", "mlp.c_proj" ]


def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator / denominator
    return softmax


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def eval(args, tokenizer, model, subject, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]

    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        for i in tqdm(range(test_df.shape[0]), desc=f"Processing {subject}"):
            # get prompt and make sure it fits
            k = args.ntrain
            prompt_end = format_example(test_df, i, include_answer=False)
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

            while crop(prompt) != prompt:
                k -= 1
                train_prompt = gen_prompt(dev_df, subject, k)
                prompt = train_prompt + prompt_end

            label = test_df.iloc[i, test_df.shape[1] - 1]

            scores = []
            for choice in choices:
                input_text = prompt + "\n" + choice
                if "Encoding" in str(type(tokenizer)) :
                    inputs = tokenizer.encode_ordinary(input_text)
                    inputs = torch.as_tensor(np.array([inputs]))
                    seqLen = min(1024, inputs.shape[1])
                    inputs = inputs[:,:seqLen]
                    print(inputs.shape)
                else:
                    inputs = tokenizer(input_text, return_tensors="pt")["input_ids"]
                    seqLen = min(1024, inputs.shape[1])
                    inputs = inputs[:, :seqLen]
                inputs = {"input_ids": v.to(model.config.device) for v in inputs}
                outputs = model(**inputs)[2]
                scores.append(outputs[0, -1, :].max().item())
            pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(scores)]
            probs = softmax(np.array(scores))

            cor = pred == label
            cors.append(cor)
            all_probs.append(probs)

        acc = np.mean(cors)
        cors = np.array(cors)

        all_probs = np.array(all_probs)
        print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs


def main(args):
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )
    subjects = subjects[:28]
    try:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
    except:
        print("args.save_dir has been made!")
    if os.path.isfile(os.path.join(args.save_dir, "all_datasets_0.txt")):
        os.remove(os.path.join(args.save_dir, "all_datasets_0.txt"))

    print(subjects, "\n")
    print(args, "\n")

    override_args = dict(
        dropout=dropout,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        router_type=router_type,
        expert_num=args.num_experts,
        gating_type=gating_type,
        noise_type=noise_type,
        load_balancing=load_balancing,
        load_balancing_lambda=load_balancing_lambda,
        topk_exp=args.num_selects,
        straight_through=straight_through,
        is_per_token=is_per_token,
        lora_target_modules=lora_target_modules,
        moe_target_modules=moe_target_modules
    )

    model = GPT.from_pretrained(init_from, override_args)
    client_weights = torch.load(args.model_path)
    model._init_weights(client_weights) 

    tokenizer = None
    if ("byt5" in args.tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    elif ("tiktoken" in args.tokenizer_path):
        tokenizer = tiktoken.get_encoding("gpt2")

    else:
        print("Tokenizer not found")
        sys.exit(1)

    # Removed pad token should be included
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # if args.select_num is not None:
    #     model.set_moe_num_selects(args.select_num)

    all_cors = []

    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, probs = eval(args, tokenizer, model, subject, dev_df, test_df)

        with open(os.path.join(args.save_dir, "all_datasets_0.txt"), "a+") as file:
            file.write("Average accuracy {:.3f} - {}\n".format(acc, subject))

        all_cors.append(cors)

        test_df["correct"] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["choice{}_probs".format(choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(args.save_dir, "{}.csv".format(subject)), index=None
        )

    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))
    gather_results(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="/mloscratch/homes/neville/MoE-TokenFree/evals/mmlu/mmlu/data") # CHANGE
    parser.add_argument("--save_dir", "-s", type=str, default="results") 
    parser.add_argument("--tokenizer_path", type=str, default="google/byt5-base") # Only for byt5
    # parser.add_argument("--model_path", type=str, default="openlm-research/open_llama_3b_v2")
    parser.add_argument("--model_path", type=str, default="/mloscratch/homes/neville/MoE-TokenFree/")
    #parser.add_argument("--model_path", type=str, default="/mloscratch/homes/neville/MoE-TokenFree/efficient-collaborative-instruction-tuning/out/local_training/3/0/local_output_0/pytorch_model.bin")
    parser.add_argument("--num_experts", type=int, default=1)
    parser.add_argument("--num_selects", type=int, default=1)
    
    args = parser.parse_args()
    main(args)
