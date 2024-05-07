import cupy as cp
from numba import cuda, uint8, uint16
import numpy as np
import csv
import sys

file = open('unicode_blocks.csv', 'r')
csvreader = list(csv.reader(file))
blocs = []
for row in csvreader[1:]:
    start = row[0]
    end = row[1]
    blkTp = (start, end)
    blocs.append(blkTp)
len_blocs = len(blocs)

print(f"First elt: {blocs[0]}")
print(f"Last elt: {blocs[-1]}")

# Assuming blocs is a list of tuples, we need to convert it to a CuPy array for efficient processing
# Convert blocs to a structured array for better handling in CUDA
#dtype = [('start', 'i4'), ('end', 'i4')]
blocs = cp.array([(int(start, 16), int(end, 16)) for start, end in blocs])

print(f"First elt: {blocs[0]}")
print(f"Last elt: {blocs[-1]}")

@cuda.jit
def find_block_kernel(chars, blocs, results):
    idx = cuda.grid(1)
    if idx < chars.size:
        char = chars[idx]
        for i in range(blocs.shape[0]):
            start = blocs[i]['start']
            end = blocs[i]['end']
            if start <= char <= end:
                idx_on_block = char - start
                if idx_on_block < 256:
                    idx_on_block = uint8(idx_on_block)
                else:
                    idx_on_block = uint16(idx_on_block)
                results[idx] = (uint8(i), idx_on_block)
                return
        results[idx] = (uint8(-1), uint8(-1))
def find_block(chars, blocs):
    results = cuda.device_array((chars.size,), dtype=np.int32)
    threadsperblock = 256
    blockspergrid = (chars.size + (threadsperblock - 1)) // threadsperblock
    find_block_kernel[blockspergrid, threadsperblock](chars, blocs, results)
    return results
def block_encoding(text, blocs):
    # Convert text to an array of Unicode code points
    chars = cp.array([ord(char) for char in text])
    results = find_block(chars, blocs)
    
    # Process results to form the final blocks array
    blocks = []
    curr_block = -1
    for block, idx_on_block in results:
        if block != curr_block:
            blocks.extend([uint8(0), uint8(block + 1), idx_on_block])
            curr_block = block
        else:
            blocks.append(idx_on_block)
    return blocks
def block_decoding(blocks, blocs):
    text = ""
    curr_block = -1
    idx = 0
    while idx < len(blocks):
        if blocks[idx] == 0:
            curr_block = blocks[idx + 1] - 1
            idx += 2
            continue
        else:
            start = blocs[curr_block]['start']
            int_utf8_char = start + blocks[idx]
            char = chr(int_utf8_char)
            text += char
        idx += 1
    return text
