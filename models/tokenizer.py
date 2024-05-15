from transformers import AutoTokenizer
import tiktoken
from torch import Tensor, as_tensor
import numpy as np

class Tokenizer:
    def __init__(self, tokenizer_name: str) -> None:
        self.tokenizer = self.tokenizer_selector(tokenizer_name)

    def tokenizer_selector(self, tokenizer_name: str):
        if tokenizer_name == "google/byt5-base":
            return AutoTokenizer.from_pretrained("google/byt5-base")
        elif tokenizer_name == "tiktoken":
            return tiktoken.get_encoding("gpt2")
        else:
            raise ValueError("Invalid tokenizer name")

    def tokenize(self, input_text: str)-> Tensor:
        if isinstance(self.tokenizer, AutoTokenizer):
            return self.huggingface_tokenization(input_text)
        elif isinstance(self.tokenizer, tiktoken.Encoding):
            return self.tiktoken_tokenization(input_text)
        else:
            raise TypeError("Tokenizer type not supported")

    def huggingface_tokenization(self, input_text: str)-> Tensor:
        inputs = self.tokenizer(input_text, return_tensors="pt")["input_ids"]
        return inputs

    def tiktoken_tokenization(self, input_text)-> Tensor:
        inputs = self.tokenizer.encode_ordinary(input_text)
        inputs = as_tensor(np.array([inputs]))
        return inputs

