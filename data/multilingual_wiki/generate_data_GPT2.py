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

from dataset_utils import separate_data, split_data_multi_lingual, save_file_multi_lingual
# from transformers import BertTokenizer
from tqdm import tqdm
seed_value = 42
random.seed(seed_value)
# Set a seed for NumPy
np.random.seed(seed_value)


# The store_true option automatically creates a default value of False.
parser = argparse.ArgumentParser()
parser.add_argument('--n_clients', type=int, default = 1,
                    help='number of clients')
parser.add_argument('--dir', type=str, help='directory for storing the data',
                    default = parent_dir + "/multilingual_wiki_test/")
parser.add_argument('--niid',  action='store_false', help='sample non-iid data for each worker' )
parser.add_argument('--balance', action='store_true')
parser.add_argument('--partition',type=str,default='dir' )
parser.add_argument('--alpha',type=float ,default=0.5 ,help='needed when using dirichelet distr')


random.seed(1)
np.random.seed(1)
num_classes = 3

def tokenize(args):
    i, text, tokenizer = args

    #print when one third of the data is processed
    tokenized_i = tokenizer(text)['input_ids']
    # Convert tokenized_i to a list of integers

    if i == 1 or (i == 2_000_000) or (i == 4_000_000) or (i == 7_000_000):
        print(f"Text: {text[:10]}")
        print(f"Text type: {type(text)}")
        print(f"Tokenized text: {tokenized_i[:10]}")
        print(f"Tokenized text type: {type(tokenized_i)}")
        print(f"Tokenized text type of elts: {type(tokenized_i[1])}")

    return tokenized_i

def byt5_tokenizer(dataset):
    ctx = mp.get_context('spawn')
    tokenized_dataset = np.array([])
    results = np.array([])

    tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")

    with ctx.Pool() as pool:
        args_generator = ((i, data_point, tokenizer) for i, data_point in enumerate(dataset))
        for result in pool.imap(tokenize, args_generator):
            print(f"---------------------------------------")
            print(f"---------------------------------------")
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
    train_path = dir_path + "trainGPT2"
    test_path = dir_path + "testGPT2"

    print(f"dir path: {dir_path}")
    print(f"train path: {train_path}")
    print(f"Dir path: {current_dir}")

    # Get multi-lingual wikipedia data
    print('This data downloading process might take a while... be patient.')
    dataset_text = []
    dataset_label = []
    i = 0
    # i am starting from de and fr, as it is just too slow :/
    # for dataset_idx in ["20220301.de", "20220301.fr"]:
    for dataset_idx in ["20220301.de","20220301.en",
            "20220301.fr"]: #, "20220301.it"]:
    # for dataset_idx in ["20220301.frr"]:
        label = dataset_idx.split(".", 1)[-1]
        if os.path.isdir(dataset_idx):
            print('loading from disk: ',dataset_idx)
            data_one_lang = load_from_disk(dataset_idx)
        else:
            data_one_lang = load_dataset("wikipedia", dataset_idx)
            data_one_lang.save_to_disk(dataset_idx)
        dataset_text.extend(data_one_lang['train']['text'])
        l = len(data_one_lang['train']['text'])
        dataset_label.extend([i]*l)
        i = i+1

    # Tokenize the data
    # tokenizer = get_tokenizer('spacy',language)
    dataset_label = np.array(dataset_label)
    # sample data
    print('sample 10\% of the data')
    sampled_indices = np.random.choice(np.arange(len(dataset_label)), size=int(0.1*len(dataset_label)), replace = False).astype(int)


    X, y, statistic = separate_data(([dataset_text[ind] for ind in sampled_indices], dataset_label[sampled_indices]), num_clients, num_classes, alpha ,niid, balance, partition)
    train_data, test_data, train_label, test_label = split_data_multi_lingual(X, y)
    tokenized_train_data = []
    tokenized_test_data = []

    for i in range(len(train_data)):
        train_combined_texts = {}
        test_combined_texts = {}
        for j in range(4):
            train_combined_texts[j] =  " ".join([train_data[i][ind] for ind in np.where(train_label[i]==j)[0]])
            test_combined_texts[j] =  " ".join([test_data[i][ind] for ind in np.where(test_label[i]==j)[0]])

        train_data_tokenized = byt5_tokenizer(train_combined_texts[j] for j in range(4))
        tokenized_train_data.append(train_data_tokenized)
        test_data_tokenized = byt5_tokenizer(test_combined_texts[j] for j in range(4))
        tokenized_test_data.append(test_data_tokenized)

    save_file_multi_lingual(config_path, train_path, test_path, tokenized_train_data, tokenized_test_data, num_clients, num_classes, statistic, niid, balance, partition, alpha)

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
