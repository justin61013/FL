import flwr as fl
import pruning
from efficient import *
# import effecientNetv2_l
from efficientv2 import *
from config import *

if __name__ == "__main__":
    while(True):
        # model = effnetv2_s()
        # pr = pruning.pruning(model, save_path_name)
        fl.server.start_server(server_address="127.0.0.1:12345", config=fl.server.ServerConfig(num_rounds=5))
        # pr.process()