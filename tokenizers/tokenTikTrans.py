# Regular imports
import csv
import sys
import time
from tqdm import tqdm

# Tokenizers
from transformers import AutoTokenizer
from blockTokenization import block_encoding, block_decoding, par_block_encoding
import sentencepiece as spm

def curate_text():
    file = open('train.csv', 'r')
    csvreader = list(csv.reader(file))
    rows = []
    for row in csvreader[1:]:
        rows.append(row[0])
    file.close()
    return rows

trained_txt = curate_text()
flat_str = ' '.join(trained_txt)
print(f"First 10 chars: {flat_str[:10]}")
print(f"First 10 chars in code_pt: {[ord(char) for char in flat_str[:10]]}")

# To get the tokeniser corresponding to a specific model in the OpenAI API:
#enc = tiktoken.encoding_for_model("gpt-4")
#from transformers import AutoTokenizer

byt5Tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")
#gpt2Tokenizer = tiktoken.get_encoding("gpt2")

text = "Hello how are you"

print(f"Starting Byt5 tokenization...")
byt5start = time.time()
byt5Ids = byt5Tokenizer(flat_str).input_ids
byt5end = time.time()
byt5_vocab_size = len(set(byt5Ids))
byt5TimeTokenize = byt5end - byt5start
print(f"ByT5 tokens: {byt5Ids[:10]}, number of tokens: {len(byt5Ids)}")
print(f"ByT5 tokenization size: {sys.getsizeof(byt5Ids) / 1_000_000} MB")
print(f"ByT5 tokenization time: {byt5TimeTokenize}")
print(f"ByT5 vocab_size time: {byt5_vocab_size}")
#byt5Decode = byt5Tokenizer.batch_decode(byt5Ids)

# Block tokenization
print(f"Starting Blk std tokenization...")
blkstart = time.time()
blkIds = block_encoding(flat_str)
blkend = time.time()
blk_vocab_size = len(set(blkIds))
blkTimeTokenize = blkend - blkstart
print(f"Block tokens: {blkIds[:10]}, number of tokens: {len(blkIds)}")
print(f"Block tokenization time: {blkTimeTokenize}")
print(f"Blk tokenization size: {sys.getsizeof(blkIds) / 1_000_000} MB")
print(f"Blk vocab_size time: {blk_vocab_size}")
diffBlk = 100 * (len(byt5Ids) - len(blkIds)) / len(byt5Ids)
print(f"Percentage difference: {diffBlk}%")

# Parallel block tokenization
#print(f"Starting Blk par tokenization...")
#parBlkstart = time.time()
#parBlkIds = par_block_encoding(flat_str)
#blkend = time.time()
#parBlkTimeTokenize = blkend - blkstart
#print(f"Block tokens: {parBlkIds[:10]}, number of tokens: {len(parBlkIds)}")
#print(f"Block tokenization time: {parBlkTimeTokenize}")
#diffParBlk = 100 * (len(byt5Ids) - len(parBlkIds)) / len(byt5Ids)
#print(f"Percentage difference: {100 - diffParBlk}%")


#gpt2Tokens = gpt2Tokenizer.encode_ordinary(text)

#if byt5Decode == flat_str:
#    print("ByT5 decoding is correct")
#
#if blkDecode == flat_str:
#    print("Block decoding is correct")


#print(f"GPT2 tokens: {gpt2Tokens}")



