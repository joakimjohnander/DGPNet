import math
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.clear()

    def reset(self):
        self.avg = 0
        self.val = 0
        self.sum = 0
        self.count = 0

    def clear(self):
        self.reset()
        self.history = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count
        else:
            self.avg = 'nan'

    def new_epoch(self):
        self.history.append(self.avg)
        self.reset()


def PSNR(images, target_images, max_imval=1):
    """Calculate the average PSNR between the images and target_images.
    max_imval is the maximum possible image value."""

    batch_size = images.data.shape[0]
    sqr_diff = (images - target_images)**2
    mse = torch.mean(sqr_diff.view(batch_size,-1), dim=1)
    psnr = 10 / math.log(10) * torch.log(max_imval**2 / mse)
    mean_psnr = torch.mean(psnr)
    return mean_psnr.data[0]
