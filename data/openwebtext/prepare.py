# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from re import split
from numpy.lib.utils import byte_bounds
from tqdm import tqdm
import numpy as np
from datasets import load_dataset # huggingface datasets
from transformers import AutoTokenizer
from datasets import DatasetDict, Dataset
import multiprocessing as mp
import time

tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")

def enc_process(example):
    ids = tokenizer(example['text']).input_ids
    print(f"Processed {len(ids)} tokens")
    return {'ids': ids, 'len': len(ids)}

def main():
    # number of workers in .map() call
    # good number to use is ~order number of cpu cores // 2
    num_proc = mp.cpu_count() # 2 

    # takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
    print("loading openwebtext dataset 54GB in huggingface .cache dir, about 8M documents (8,013,769)")
    dataset = load_dataset("openwebtext", split="train", num_proc=num_proc, trust_remote_code=True)

    # owt by default only contains the 'train' split, so create a test split
    print("splitting the dataset into train and test")
    split_dataset = dataset.train_test_split(test_size=0.0005, seed=2357, shuffle=True, load_from_cache_file=True)
    print(f"Split dataset: {split_dataset}")
    split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

    # this results in:
    # >>> split_dataset
    # DatasetDict({
    #     train: Dataset({
    #         features: ['text'],
    #         num_rows: 8009762
    #     })
    #     val: Dataset({
    #         features: ['text'],
    #         num_rows: 4007
    #     })
    # })
    
    print(f"Starting tokenization of the splits")
    start = time.time()
    tokenized = split_dataset.map(
        enc_process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )
    print(f"==========================================")
    print(f"==========================================")
    print(f"Tokenization took {time.time() - start} seconds")
    print(f"Tokenized dataset: {tokenized}")
    print(f"==========================================")
    print(f"==========================================")
    
    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'])
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()


# train.bin is ~17GB, val.bin ~8.5MB
# train has ~9B tokens (9,035,582,198)
# val has ~4M tokens (4,434,897)

# to read the bin files later, e.g. with numpy:
# m = np.memmap('train.bin', dtype=np.uint16, mode='r')
