import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings





class KernelBase(nn.Module):
    """Kernel base class for kernels, all kernels inherit from this
    """
    def __add__(self,other):
        return AdditiveKernels(self,other)
    def __mul__(self,other):
        return MultiplicativeKernels(self,other)

class AdditiveKernels(KernelBase):
    def __init__(self,k1,k2):
        super().__init__()
        self.k1 = k1
        self.k2 = k2
    def __call__(self,x,y):
        return self.k1(x,y)+self.k2(x,y)

class MultiplicativeKernels(KernelBase):
    def __init__(self,k1,k2):
        super().__init__()
        self.k1 = k1
        self.k2 = k2
    def __call__(self,x,y):
        return self.k1(x,y)*self.k2(x,y)



class RBF(KernelBase):
    def __init__(self, length=1., scale=1., trainable=False):
        super().__init__()
        if trainable:
            self.length = nn.Parameter(torch.tensor(length)) 
            self.scale = nn.Parameter(torch.tensor(scale))
        else:
            self.length = torch.tensor(length)
            self.scale = torch.tensor(scale)
    def forward(self,x,y):
        """ 
        """
        B,N,D = x.shape
        return self.scale**2*torch.exp(-torch.cdist(x, y, p=2.0)**2/(2*self.length**2*D))

    def __str__(self):
        return "RBF-kernel"
        

class LinearKernel(KernelBase):
    def __init__(self,scale=1.):
        super().__init__()
        self.scale = torch.tensor(scale)
    def forward(self,x,y):
        """ 
        """
        B,N,D = x.shape
        return self.scale**2*torch.einsum('bnd,bmd->bnm',x,y)
    def __str__(self):
        return "linear-kernel"


class Laplace(KernelBase):
    def __init__(self, length=1., scale=1., trainable=False):
        super().__init__()
        if trainable:
            self.length = nn.Parameter(torch.tensor(length)) 
            self.scale = nn.Parameter(torch.tensor(scale))
        else:
            self.length = torch.tensor(length)
            self.scale = torch.tensor(scale)
    def forward(self,x,y):
        """ 
        """
        B,N,D = x.shape
        return self.scale**2*torch.exp(-torch.cdist(x, y, p=2.0)/(self.length*math.sqrt(D)))
    def __str__(self):
        return "Laplace-kernel"



