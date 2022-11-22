import flwr as fl
import pruning
from efficient import *

model = EfficientNetB0()
if __name__ == "__main__":
    while(5):
        pr = pruning.pruning(model, 'efficientnet-b0.pth')
        fl.server.start_server(server_address="127.0.0.1:12345", config=fl.server.ServerConfig(num_rounds=5))
        pr.process()