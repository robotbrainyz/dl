import torch

def get_device():
    if torch.cuda.is_available():  
        dev = "cuda:0"
    else:  
        dev = "cpu"
    return torch.device(dev)
