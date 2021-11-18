
import torch


def recursive_detach(data):
    """Recursively goes through a structure of lists and dicts and detaches all tensors
    """
    if isinstance(data, dict):
        return {key: recursive_detach(val) for key, val in data.items()}
    elif isinstance(data, (list, tuple)):
        return [recursive_detach(elem) for elem in data]
    elif isinstance(data, torch.Tensor):
        return data.detach()
    else:
        return data
        
def recursive_to(data, device):
    """Recursively goes through a structure of lists and dicts and moves all tensors to requested device
    """
    if isinstance(data, dict):
        return {key: recursive_to(val, device) for key, val in data.items()}
    elif isinstance(data, (list, tuple)):
        return [recursive_to(elem, device) for elem in data]
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data

def recursive_tensor_sizes(data):
    if isinstance(data, dict):
        return {key: recursive_tensor_sizes(val) for key, val in data.items()}
    elif isinstance(data, (list, tuple)):
        return [recursive_tensor_sizes(elem) for elem in data]
    elif isinstance(data, torch.Tensor):
        return data.size()
    else:
        return data
