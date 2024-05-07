import csv
import sys
from numpy import uint8, uint16
import concurrent.futures
from typing import List, Tuple

file = open('unicode_blocks.csv', 'r')
csvreader = list(csv.reader(file))
blocs = []
for row in csvreader[1:]:
    blocs.append(row)
len_blocs = len(blocs)

def find_block(char: int):
    """
        This function will find the block of a character in the csv file
        Each block has a start and end column if the character is in the block
        then the function will return the block
    """
    for idx in range(0, len_blocs):
        start = int(blocs[idx][0], 16)
        end = int(blocs[idx][1], 16)
        if  start <= char <= end:
            idx_on_block = char - start
            if idx_on_block < 256:
                idx_on_block = uint8(idx_on_block)
            else:
                idx_on_block = uint16(idx_on_block)
            return uint8(idx), idx_on_block
    return uint8(-1), uint8(-1)

def block_encoding(text)-> list[int]:
    """
        This function will tokenize a text into blocks
    """
    curr_block = -1
    blocks = []
    for char in text:
        code_pt = ord(char)
        block, idx_on_block = find_block(code_pt)
        if block != curr_block:
            blocks.append(uint8(0))
            blocks.append(uint8(block + 1))
            blocks.append(idx_on_block)
            curr_block = block
        else:
            blocks.append(idx_on_block)
    return blocks

def block_decoding(blocks: list[int])-> str:
    """
        This function will decode a list of idxs into text using the block information
    """
    curr_block = -1
    text = ""
    idx = 0
    while idx < len(blocks):
        if 0 == blocks[idx]:
            curr_block = blocks[idx + 1] - 1
            idx += 2
            continue
        else:
            start = int(blocs[curr_block][0], 16)
            int_utf8_char = start + blocks[idx]
            #print(f"int_utf8_char: {int_utf8_char}")
            char = chr(int_utf8_char)
            text += char
        idx += 1
    return text

def par_find_block(char: int):
    """
    Find the block of a character in the csv file.
    Each block has a start and end column; if the character is in the block,
    then the function will return the block.
    """
    for idx in range(0, len_blocs):
        start = int(blocs[idx][0], 16)
        end = int(blocs[idx][1], 16)
        if start <= char <= end:
            idx_on_block = char - start
            if idx_on_block < 256:
                idx_on_block = uint8(idx_on_block)
            else:
                idx_on_block = uint16(idx_on_block)
            return uint8(idx), idx_on_block
    return uint8(-1), uint8(-1)

def process_char(char: str):
    code_pt = ord(char)
    return par_find_block(code_pt)

def par_block_encoding(text: str) -> List[int]:
    """
    Tokenize a text into blocks using parallel processing.
    """
    blocks = []
    curr_block = -1
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_char, text))
    for block, idx_on_block in results:
        if block != curr_block:
            blocks.extend([uint8(0), uint8(block + 1), idx_on_block])
            curr_block = block
        else:
            blocks.append(idx_on_block)
    return blocks

def main():

    example = "Hello"
    someLetters = " ĺĻļĽľĿŀŁasdałŃńŅņŇňŉŊŋŌōŎŏŐő Œœ@ſ€@€ŔŕŖŗŘřŚśŜŝŞşŠšŢţüöäŤťŦŧŨũŪūŬ"
    moreLetters="ڸڹںڻڼڽھڿۀہۂۃۄۅۆۇۈۉۊۋیۍێۏېۑےۓ۔ە ۗ ۙ ۛ ۝۞ ۠ ۢ ۤۥۦ۩"
    lsts = [someLetters, moreLetters]


    before = someLetters
    print(f"Size in bytes before encoding: {str(len(before.encode('utf-8')))}")

    idxs = block_encoding(before)
    print(f"Indexes: {idxs}")
    print(f"Size of bytes after encoding: {str(idxs.__sizeof__())}")

    decoded = block_decoding(idxs)
    if before.encode("utf-8").hex() == decoded.encode("utf-8").hex():
        print("Test passed")
    else: 
        print(f"Expected: {before} | {before.encode('utf-8').hex()}")
        print(f"Got: {decoded} | {decoded.encode('utf-8').hex()}")
        print("Test failed")
    
    return

if __name__ == "__main__":
    main()
