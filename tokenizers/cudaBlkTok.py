import cupy as cp
from numba import cuda, uint8, uint16, uint32
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
                results[idx*2] = uint8(i)
                results[idx*2 + 1] = idx_on_block
                return
        results[idx*2] = uint8(-1)
        results[idx*2 + 1] = uint8(-1)

def find_block(chars, blocs):
    results = cuda.device_array((chars.size,))
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
    i = 0
    while i < cp.ceil(results.size):
        if results[i] != curr_block:
            blocks.append(uint8(0))              # 0 indicates a new block
            blocks.append(uint8(results[i] + 1)) # 1-based index of the block
            blocks.append(results[i + 1])        # Index of the character in the block
            curr_block = results[i]              # Update the current block
        else:
            blocks.append(results[i + 1])        # Index of the character in the block
        i += 2
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

def main():
    text = "Hello, World!"
    blocks = block_encoding(text, blocs)
    print(f"Encoded blocks: {blocks}")
    #text_decoded = block_decoding(blocks, blocs)
    print(f"Decoded text: {text_decoded}")

if __name__ == "__main__":
    main()
