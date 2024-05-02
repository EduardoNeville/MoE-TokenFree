from tqdm import tqdm
import argparse
from collab_utils.clients import GeneralClient
from collab_utils.server import Server
from models.model import GPTConfig, GPT
import os

parser = argparse.ArgumentParser()
parser.add_argument("-gr", "--num_global_rounds", default = 20, type=int)
parser.add_argument('-le',"--num_local_epochs",default=5, type=int)
parser.add_argument("-model_path", "--model_path", type = str)
parser.add_argument('-lr',"--learning_rate",default=5e-3,type=float)
parser.add_argument('-mom',"--momentum",default=0.9)
parser.add_argument('-lam','--lambda_',default=0.5,type=float)
parser.add_argument('-ds','--dataset',default='agnews',type=str)
parser.add_argument('-nc','--num_clients',default=6,type=int)
parser.add_argument('-device','--device',default="cuda",type=str)
parser.add_argument('-bs','--batch_size',default=8,type=input)
args = parser.parse_args()

# prepare all clients
print('=============== initializing clients and server')
data_path = os.path.join("/mlodata1/dongyang/codes/EC-LLM/ec-llm/data", args.dataset)
clients = {}
for i in range(args.num_clients):
    model_args = dict(batch_size = args.batch_size) # start with model_args from command line
    gptconf = GPTConfig(**model_args)
    clients[i] = GeneralClient(args=args, client_id=i, model=GPT, data_path = data_path, 
                               output_dir = '/mlodata1/dongyang/codes/EC-LLM/ec-llm/out',
                               config = gptconf)

server = Server(args,GPT, config = model_args)
print('=============== collaborative finetuning')
# perform collaborative local fine tuning
for epoch in tqdm(range(args.num_global_rounds)):
    for id in range(args.num_clients):
        clients[id].synchronize(server.server_model)
        clients[id].train(local_num_steps = 100)
        clients[id].eval()
    server.aggregate_parameters([clients[i].model for i in range(args.num_clients)])
