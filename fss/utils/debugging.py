import subprocess

import torch
import torch.nn.functional as F

def prod(lst):
    prod = 1
    for elem in lst:
        prod *= elem
    return prod

def get_tensor_statistics_str(tensor, name="", formatting="standard"):
    """ Returns string of formatted tensor statistics, contains min, max, mean, and std"""
    if isinstance(tensor, (torch.FloatTensor, torch.cuda.FloatTensor)):
        if tensor.numel() == 0:
            string = "size: {}".format(tensor.size())
        elif formatting == "standard":
            string = "elem in [{:9.3f}, {:9.3f}]    mean: {:9.3f}    std: {:9.3f}    size: {}".format(tensor.min().item(), tensor.max().item(), tensor.mean().item(), tensor.std().item(), tuple(tensor.size()))
        elif formatting == "short":
            string = "[{:6.3f}, {:6.3f}]  mu: {:6.3f}  std: {:6.3f}  {!s:17} {: 6.1f}MB".format(tensor.min().item(), tensor.max().item(), tensor.mean().item(), tensor.std().item(), tuple(tensor.size()), 4e-6 * prod(tensor.size()))
    elif isinstance(tensor, (torch.LongTensor, torch.ByteTensor, torch.cuda.LongTensor, torch.cuda.ByteTensor)):
        if tensor.numel() == 0:
            string = "size: {}".format(tensor.size())
        else:
            tensor = tensor.to('cpu')
            string = "elem in [{}, {}]    size: {}    HIST BELOW:\n{}".format(tensor.min().item(), tensor.max().item(), tuple(tensor.size()), torch.stack([torch.arange(0, tensor.max()+1), tensor.view(-1).bincount()], dim=0))
    elif isinstance(tensor, (torch.BoolTensor, torch.cuda.BoolTensor)):
        if tensor.numel() == 0:
            string = "size: {}".format(tensor.size())
        else:
            tensor = tensor.to('cpu').long()
            string = f"BoolTensor with {tensor.sum()} True values out of {tensor.numel()}, size: {tensor.size()}"
    elif isinstance(tensor, (float, int, bool)):
        string = f"{type(tensor)}: {tensor}"
    else:
        tensor_type = tensor.type() if isinstance(tensor, torch.Tensor) else type(tensor)
        raise NotImplementedError(f"A type of tensor not yet supported was input. Expected torch.FloatTensor or torch.LongTensor, got: {tensor_type}")
    string = string + "    " + name
    return string

def print_tensor_statistics(tensor, name="", formatting="standard"):
    print(get_tensor_statistics_str(tensor, name, formatting))

def get_weight_statistics_str(layer, name="", formatting="standard"):
    return get_tensor_statistics_str(layer.weight, name, formatting)

def get_memory_str():
    return "{:.2f} MB".format(torch.cuda.memory_allocated() / 1e6)
def print_memory():
    print(get_memory_str())

def visualize_modules(module, space=""):
    if hasattr(module, 'weight'):
        print(space, type(module), get_tensor_statistics_str(module.weight.data))
    else:
        print(space, type(module))
    for child_module in module.children():
        visualize_modules(child_module, space=space+"-")

def get_model_size_str(model):
    nelem = 0
    for module in model.modules():
        if hasattr(module, 'weight'):
            nelem += module.weight.numel()
        if hasattr(module, 'bias'):
            nelem += module.weight.numel()
    size_str = "{:.2f} MB".format(nelem * 4 * 1e-6)
    return size_str

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def print_nvidia_smi():
    print(subprocess.check_output('nvidia-smi').decode('UTF-8'))

def revert_imagenet_normalization(sample):
    """
    sample (Tensor): of size (nsamples,nchannels,height,width)
    """
    # Imagenet mean and std    
    mean = [.485,.456,.406]
    std = [.229,.224,.225]
    mean_tensor = torch.Tensor(mean).view(3,1,1).to(sample.device)
    std_tensor = torch.Tensor(std).view(3,1,1).to(sample.device)
    non_normalized_sample = sample*std_tensor + mean_tensor
    return non_normalized_sample

COLOR_RED = torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1)
COLOR_MAGENTA = torch.tensor([1.0, 0.0, 1.0]).view(3, 1, 1)
COLOR_WHITE = torch.tensor([1.0, 1.0, 1.0]).view(3, 1, 1)


