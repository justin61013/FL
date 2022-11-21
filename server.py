import flwr as fl
import pruning
from efficient import *

pr = pruning.pruning(EfficientNetB0(), 'efficientnet-b0.pth')
if __name__ == "__main__":
    while(5):
        fl.server.start_server(server_address="127.0.0.1:12345", config=fl.server.ServerConfig(num_rounds=1))
        pr.process()