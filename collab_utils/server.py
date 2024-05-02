
import copy
import time

class Server(object):
    def __init__(self, args, model, config):
        self.server_model = model.from_pretrained("gpt2",config).to(args.device)
        self.args = args

    def aggregate_parameters(self, uploaded_models):
        assert (len(uploaded_models) > 0)
        for param in self.server_model.parameters():
            param.data.zero_()
        weights = [1/self.args.num_clients for i in range(self.args.num_clients)]
        for w, client_model in zip(weights, uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.server_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

 