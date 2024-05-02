import transformers import AutoTokenizer
import tiktoken
import torch
import numpy as np

class Tokenizer():
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer_selector(self, tokenizer)
        pass

    def huggingface_tokenization(self, input_text, ):
        inputs = self.tokenizer(input_text, return_tensors="pt")["input_ids"]

        
    def tiktoken_tokenization(self, input_text):
        inputs = self.tokenizer.encode_ordinary(input_text)
        inputs = torch.as_tensor(np.array([inputs]))
        return inputs

    def tokenizer_selector(self, tokenizer):
        match tokenizer:
            case "google/byt5-base":
                return AutoTokenizer.from_pretrained("google/byt5-base")
            case "tiktoken":
                return tiktoken.get_encoding("gpt2")
            case _:
                raise ValueError("Invalid tokenizer")
