import transformers
from contextlib import nullcontext
import os
from datasets import load_dataset
import copy
import numpy as np
from collections import OrderedDict
import torch
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
import time


def get_batch(data, seq_length, batch_size, device='cpu'):
    '''
    returns a batch of size ([seq_length, batch_size])
    '''
    ix = torch.randint(len(data) - seq_length, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+seq_length]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+seq_length]).astype(np.int64)) for i in ix])
    if "cuda" in torch.device(device).type:
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    return x, y

class GeneralClient:
    def __init__(self, args, client_id, model, data_path, output_dir, config, seq_length=128):
        self.client_id = client_id
        self.local_train_data_path = os.path.join(data_path, "train/{}.bin".format(self.client_id))
        self.local_test_data_path = os.path.join(data_path, "test/{}.bin".format(self.client_id))
        self.local_data_train = np.memmap(self.local_train_data_path, dtype=np.uint16, mode='r')
        self.local_data_test = np.memmap(self.local_test_data_path, dtype=np.uint16, mode='r')
        self.output_dir = output_dir
        self.seq_length = seq_length
        self.args = args
        self.model = model(config).to(args.device)
        self.local_output_dir = os.path.join(self.output_dir, "trainer_saved", "local_output_{}".format(self.client_id))
        self.optimizer = self.model.configure_optimizers(weight_decay=1e-2, learning_rate=1e-4, betas=(0.9, 0.95), device_type=args.device)

    def synchronize(self, server_model):
        # self.model = copy.deepcopy(server_model)
        for server_param, client_param in zip(server_model.parameters(), self.model.parameters()):
            client_param.data = server_param.data.clone() 


    def train(self,local_num_steps, acc_steps=4):
        type_ctx = nullcontext() if self.args.device== 'cpu' else torch.amp.autocast(
        device_type=self.args.device, dtype=torch.float16)  # extra_args.dtype)
        self.model.train()
        itr = 0
        for step in range(local_num_steps):  
            # for microstep_idx in range(acc_steps):# gradient accumulation
            x, y = get_batch(self.local_data_train, self.seq_length, batch_size=self.args.batch_size, device=self.args.device)
            _, loss = self.model(x, targets=y)
            loss.backward()
            
            self.optimizer.step()
            self.optimizer.zero_grad()

            # itr += 1
    
    @torch.no_grad()
    def eval(self, max_num_batches =100):
        self.model.eval()
        loss_list_val, acc_list = [], []
        for _ in range(max_num_batches): 
            x, y = get_batch(self.local_data_test, self.seq_length, batch_size=self.args.batch_size, device=self.args.device)
            logits, loss = self.model(x, targets=y)
            val_loss = loss
            loss_list_val.append(val_loss)
            acc_list.append((logits.argmax(-1) == y).float().mean())

        val_acc = torch.stack(acc_list).mean().item()
        val_loss = torch.stack(loss_list_val).mean().item()
        val_perplexity = 2.71828 ** val_loss
        print('val_acc:',val_acc, '\n',
              'val_loss:',val_loss, '\n',
              'val_perplexity:',val_perplexity)
        return val_acc, val_loss, val_perplexity
    