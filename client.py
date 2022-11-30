from collections import OrderedDict
from typing import Dict, List, Tuple
import numpy as np
import torch
import cifar
import flwr as fl
from efficient import *
import os
from efficientv2 import *
import time
from config import *
DEVICE = "cpu"

DEVICE: str = torch.device("cuda:"+ str(GPU) if torch.cuda.is_available() else "cpu")
# print(torch.cuda.is_available())
print(DEVICE)

def load_path(model, path):
    new_state_dict = OrderedDict()
    for key,value in torch.load(path).items():
        if 'weight_orig' in key:
            title = key.split('.')
            name = title[0]+'.weight'
            new_state_dict[name] = value
            value1 = value
            print(key)
        elif 'weight_mask' in key:
            title = key.split('.')
            name = title[0]+'.weight'
            new_state_dict[name] = value1
            print(key)
        else:
            new_state_dict[key] = value
    model.load_state_dict(new_state_dict)
    
class CifarClient(fl.client.NumPyClient):
    """Flower client implementing CIFAR-10 image classification using
    PyTorch."""

    def __init__(
        self,
        model: cifar.Net,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        num_examples: Dict,
        epoc: int
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples
        self.epoc = epoc
        self.MAX = 0

    def get_parameters(self, config) -> List[np.ndarray]:
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        # load efficientnet state_dict
        self.model.load_state_dict(state_dict, strict=False)
        # save efficientnet state_dict
        if cifar.test(self.model, self.testloader, device=DEVICE)[1] > self.MAX:
            torch.save(self.model.state_dict(), save_path_name)
            self.MAX = cifar.test(self.model, self.testloader, device=DEVICE)[1]
            # torch.save(state_dict, save_path_name)
            print('save efficientnet-b0.pth')
        # self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        cifar.train(self.model, self.trainloader, epochs=self.epoc, device=DEVICE)
        print('fit')
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = cifar.test(self.model, self.testloader, device=DEVICE)
        print('evaluate')
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}

def main() -> None:
    """Load data, start CifarClient."""
    
    # Load model and data
    model =  effnetv2_s()
    model.to(DEVICE)
    #  load path if path exist
    if os.path.exists(save_path_name):
        load_path(model, save_path_name)
    trainloader, testloader, num_examples = cifar.load_data()

    # Start client
    client = CifarClient(model, trainloader, testloader, num_examples, 1)
    for _ in range(5):
        fl.client.start_numpy_client(server_address="127.0.0.1:12345", client=client)
        time.sleep(2)


if __name__ == "__main__":
    main()
