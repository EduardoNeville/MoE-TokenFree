import numpy as np
import os
import sys
import random
import sys
import argparse
import warnings
import multiprocessing as mp

warnings.simplefilter("ignore", category=UserWarning)
from datasets import load_dataset, load_from_disk
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from transformers import AutoTokenizer
from dataset_utils import separate_data, split_data, save_file
from tqdm import tqdm

# The store_true option automatically creates a default value of False.
parser = argparse.ArgumentParser()
parser.add_argument('--n_clients', type=int, default = 1,
                    help='number of clients')
parser.add_argument('--dir', type=str, help='directory for storing the data',
                    default = parent_dir + "/multilingual_wiki_single_client/")
parser.add_argument('--niid',  action='store_false', help='sample non-iid data for each worker' )
parser.add_argument('--balance', action='store_true')
parser.add_argument('--partition',type=str,default='dir' )
parser.add_argument('--alpha',type=float ,default=0.5 ,help='needed when using dirichelet distr')

random.seed(1)
np.random.seed(1)
num_classes = 3 # TODO: change to 4

def tokenize(args):
    i, text, dataLen, tokenizer = args
    
    #print when one third of the data is processed
    tokenized_i = tokenizer(text)['input_ids']
    # Convert tokenized_i to a list of integers

    if i == 1 or (i == int(dataLen / 2)) or (i == int(3* dataLen / 4)):
        print(f"Text: {text[:10]}")
        print(f"Text type: {type(text)}")
        print(f"Tokenized text: {tokenized_i[:10]}")
        print(f"Tokenized text type: {type(tokenized_i)}")
        print(f"Tokenized text type of elts: {type(tokenized_i[1])}")

    return tokenized_i

def byt5_tokenizer(dataset):
    dataLen = len(dataset)
    ctx = mp.get_context('spawn')
    tokenized_dataset = np.array([])
    results = np.array([])

    tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")

    with ctx.Pool() as pool:
        for result in pool.map(tokenize, [(i, dataset[i], dataLen, tokenizer) for i in range(dataLen)]):
            print(f"---------------------------------------")
            print(f"---------------------------------------")
            print(f"Length of Length: {dataLen} ")
            print(f"Length of result: {len(result)} ")
            print(f"type of result : {type(result)}")
            print(f"---------------------------------------")
            print(f"---------------------------------------")

            results = np.array(result).flatten()
            print(f"type of results : {type(results)}")
            print(f"type of results[0] : {type(results[0])}")
            print(f"results[0] : {results[0]}")
    pool.close()
    pool.join()
    print(f"Concatenating results...")
    tokenized_dataset = np.concatenate((tokenized_dataset, results), axis=None)

    print(f"Tokenization complete. \n {tokenized_dataset.shape}")

    return tokenized_dataset

# Allocate data to users
def generate_multi_ling_wiki(dir_path, num_clients, num_classes, alpha ,niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    config_path = dir_path + "config.json"
    train_path = dir_path + "train"
    test_path = dir_path + "test"

    print(f"dir path: {dir_path}")
    print(f"train path: {train_path}")

    # Get multi-lingual wikipedia data
    print('This data downloading process might take a while... be patient.')
    dataset_text = []
    dataset_label = []
    i = 0
    # i am starting from de and fr, as it is just too slow :/
    # for dataset_idx in ["20220301.de", "20220301.fr"]:
    for dataset_idx in ["20220301.de", "20220301.en",
                        "20220301.fr"]: #, "20220301.it"]: TODO CHANGE!
    # for dataset_idx in ["20220301.frr"]:
        label = dataset_idx.split(".", 1)[-1]
        if os.path.isdir(dataset_idx):
            print('loading from disk: ',dataset_idx)
            data_one_lang = load_from_disk(dataset_idx)
            if data_one_lang is None:
                print('dataset not found so \n loading from web: ',dataset_idx)
                data_one_lang = load_dataset("wikipedia", dataset_idx, trust_remote_code=True)
                data_one_lang.save_to_disk(dataset_idx)
        else:
            print('loading from web: ',dataset_idx)
            data_one_lang = load_dataset("wikipedia", dataset_idx, trust_remote_code=True)
            data_one_lang.save_to_disk(dataset_idx)

        print(f"Dataset loaded for {dataset_idx}")
        dataset_text.extend(data_one_lang['train']['text'])
        l = len(data_one_lang['train']['text'])
        dataset_label.extend([i]*l)
        i = i+1


    dataset_label = np.array(dataset_label)
    # raw_tokenized = tokenizer.encode_ordinary_batch(dataset_text)

    print(f"Splitting data into {num_clients} clients")
    X, y, statistic = separate_data((dataset_text, dataset_label), num_clients, num_classes, alpha ,niid, balance, partition)
    train_data, test_data = split_data(X, y)
    tokenized_train_data = []
    tokenized_test_data = []

    print(f"Training data shape: {type(train_data)}")
    print(f"Training data: {train_data[0][0]}")
    
    tokenized_train_data = []
    tokenized_test_data = []
    print('================ Start tokenization: (another very long process)')
    for i in tqdm(range(len(train_data))):
        tokenized_train_data_ = []
        tokenized_test_data_ = []

        # Check if tokenized data already exists
        train_i_path = os.path.join(train_path, f"{str(i)}-train-de.npy")
        print(f"Path to train file {i}: {train_i_path}")
        if os.path.isfile(train_i_path) is False:
            print(f"Training data file not found at: {train_i_path}")
            print(f"Tokenizing data for client {i}...")

            if not os.path.exists(train_path):
                os.makedirs(train_path)
            tokenized_train_data_ = byt5_tokenizer(train_data[i])

            np.save(train_i_path, tokenized_train_data_)

            print(f"Print tokenized dataset: {tokenized_train_data_[:10]}")
        else:
            print(f"Training data file found at: {train_i_path}")
            tokenized_train_data_ = np.load(train_i_path)
            print(f"Tokenized training data: {tokenized_train_data_[:10]}")

        tokenized_train_path = os.path.join(train_path, 'tokenized_train_data.npy')
        if os.path.isfile(tokenized_train_path) is False:
            tokenized_train_data = []
        else:
            tokenized_train_data = np.load(tokenized_train_path)

        tokenized_train_data = np.append(tokenized_train_data, tokenized_train_data_)
        np.save(tokenized_train_path, tokenized_train_data)
        print(f"Tokenisation stored in {train_path}")

        # Check if tokenized data already exists
        test_i_path = os.path.join(test_path, f"{str(i)}-test-de.npy")
        if os.path.isfile(test_i_path) is False:
            if not os.path.exists(test_path):
                os.makedirs(test_path)
            tokenized_test_data_ = byt5_tokenizer(test_data[i])
            np.save(test_i_path, tokenized_test_data_)
        else:
            print(f"Testing data file found at: {test_i_path}")
            tokenized_test_data_ = np.load(test_i_path)
            print(f"Tokenized testing data: {tokenized_test_data_[:10]}")

        tokenized_test_path = os.path.join(test_path, 'tokenized_test_data.npy')
        if os.path.isfile(tokenized_test_path) is False: 
            tokenized_test_data = []
        else:
            tokenized_test_data = np.load(tokenized_test_path)
            print(f"Tokenized testing data: {tokenized_test_data_[:10]}")

        print(f"Tokenized testing data! \n Tokenized data: {tokenized_test_data_[:10]}")
        tokenized_test_data = np.append(tokenized_test_data, tokenized_test_data_)
        np.save(tokenized_test_path, tokenized_test_data)

    save_file(config_path, train_path, test_path, tokenized_train_data, tokenized_test_data, num_clients, num_classes, statistic, niid, balance, partition)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    args = parser.parse_args()
    niid = args.niid 
    balance = args.balance
    partition = args.partition
    alpha = args.alpha
    print("non iid:", args.niid)
    print("partition:", args.partition)
    if not os.path.exists(args.dir):
        os.makedirs(args.dir)
    generate_multi_ling_wiki(args.dir, args.n_clients, num_classes, alpha, niid, balance, partition)
