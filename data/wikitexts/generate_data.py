from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import os
from tqdm import tqdm
import multiprocessing as mp
import tiktoken

tiktokenizer = tiktoken.get_encoding("gpt2")
byt5tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")
WIKITEXT_DATA_PATH = os.path.dirname(__file__)

def enc_gpt(example):
    ids = tiktokenizer.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
    ids.append(tiktokenizer.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
    return {'ids': ids}

def enc_byt5(example):
    ids = byt5tokenizer(example['text'])['input_ids']
    return {'ids': ids}

def get_wikitext_data():
    """Tokenize Wikitext data using ByT5-base tokenizer."""

    if not os.path.exists(os.path.join(WIKITEXT_DATA_PATH, 'gpt_train.bin')):
        os.makedirs(WIKITEXT_DATA_PATH, exist_ok=True)
        print("Tokenizing data...")

        print("Loading dataset...")
        # Load and tokenize the training data
        dataset_train = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        print(f"Dataset loaded: {dataset_train['text']}")

        num_proc = mp.cpu_count() # 2 
        tokenized_train = dataset_train.map(enc_gpt, remove_columns=['text'], num_proc=num_proc, desc="Tokenizing the training data")

        print(f"Training data tokenisation complete. \n {tokenized_train.shape}")
        print(f"Tokenized data: {tokenized_train[:3]}")
        print(f"Tokenized data type: {type(tokenized_train[:3])}")

        # flattening the tokenized dataset_train
        lstTok = [item for sublist in tqdm(tokenized_train['ids']) for item in sublist]

        print(f"lstTok: {lstTok[:3]}")
        
        train_tokenized_np = np.array(lstTok, dtype=np.uint16)
        train_tokenized_np.tofile(os.path.join(WIKITEXT_DATA_PATH, 'gpt_train.bin'))
        print("gpt_train.bin created")

        # Load and tokenize the training data
        dataset_val = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")

        tokenized_val = dataset_val.map(enc_gpt, remove_columns=['text'], num_proc=num_proc, desc="Tokenizing the validation data")

        lstTok = [item for sublist in tqdm(tokenized_val['ids']) for item in sublist]

        val_tokenized_np = np.array(lstTok, dtype=np.uint16)
        val_tokenized_np.tofile(os.path.join(WIKITEXT_DATA_PATH, 'gpt_val.bin'))
        print("gpt_val.bin created")

    train_data = np.memmap(os.path.join(WIKITEXT_DATA_PATH, 'gpt_train.bin'), dtype=np.int16, mode='r')

    val_data = np.memmap(os.path.join(WIKITEXT_DATA_PATH, 'gpt_val.bin'), dtype=np.int16, mode='r')

    return {'train': train_data , 'val': val_data}

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    data = get_wikitext_data()
    print(data['train'].shape)
