"""
DFN code (CAB, RRB, DFN) adapted from https://github.com/lxtGH/dfn_seg.git
ImageEncoder adapted from torchvision resnet
"""


import os
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torch.distributions as tdist

from fss.utils.debugging import print_tensor_statistics, revert_imagenet_normalization, COLOR_RED, COLOR_WHITE
from local_config import config
from functools import partial

LOG2PI = 1.8378
PI = 3.14159265359

GLOBALS = {
#    'number of forward calls': 0, 'times_learn': [], 'times_forward': []
}

    
class CAB(nn.Module):
    def __init__(self, in_channels, out_channels,use_silu=False):
        super(CAB, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        if use_silu:
            self.relu = nn.SiLU()
        else:
            self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1, x2 = x
        x = torch.cat([x1,x2],dim=1)
        x = self.global_pooling(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmod(x)
        x2 = x * x2
        res = x2 + x1
        return res

class RRB(nn.Module):
    def __init__(self, in_channels, out_channels,use_silu=False):
        super(RRB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if use_silu:
            self.relu = nn.SiLU()
        else:
            self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        res  = self.conv2(x)
        res = self.bn(res)
        res = self.relu(res)
        res = self.conv3(res)
        return self.relu(x + res)

class DFN(nn.Module):
    """This DFN code is quite adaptive and works with multi-scale GPs and shallow features of
    multiple levels.
    """
    def __init__(self, internal_dim, feat_input_modules, pred_input_modules,
                 rrb_d_dict, cab_dict, rrb_u_dict, terminal_module=None,
                 internal_upsample_mode='nearest', terminal_upsample_mode='bilinear'):
        super().__init__()
        self.internal_dim = internal_dim
        self.feat_input_modules = feat_input_modules
        self.pred_input_modules = pred_input_modules
        self.rrb_d = rrb_d_dict
        self.cab = cab_dict
        self.rrb_u = rrb_u_dict
        self.terminal_module = terminal_module if terminal_module is not None else nn.Identity()
        self.internal_upsample_mode = internal_upsample_mode
        self.terminal_upsample_mode = terminal_upsample_mode

    def forward(self, anno_enc_preds, feats):
        B, N, _, H32, W32 = feats['s32'].size()
        H = {f's{scale}': H32 * 32 // scale for scale in [32, 16, 8, 4, 1]}
        W = {f's{scale}': W32 * 32 // scale for scale in [32, 16, 8, 4, 1]}
        device = feats['s32'].device

        input_feats = {
            key: module(feats[key].view(B*N, *feats[key].size()[2:]))
            for key, module in self.feat_input_modules.items()
        }
        input_preds = {
            key: module(anno_enc_preds[key].view(B*N, *anno_enc_preds[key].size()[2:]))
            for key, module in self.pred_input_modules.items()
        }
        out = torch.zeros((B*N, self.internal_dim, H32, W32), device=device)
        for key in ['s32', 's16', 's8', 's4']:
            if key in input_feats and key in input_preds:
                new_stuff = torch.cat([input_feats[key], input_preds[key]], dim=1)
            elif key in input_feats:
                new_stuff = input_feats[key]
            elif key in input_preds:
                new_stuff = input_preds[key]
                
            if key in input_feats or key in input_preds:
                new_stuff = self.rrb_d[key](new_stuff)
                if out.size()[2:] != (H[key], W[key]):
                    out = F.interpolate(
                        out,
                        size=(H[key], W[key]),
                        mode=self.internal_upsample_mode,
                        align_corners = False if self.internal_upsample_mode == 'bilinear' else None)
                out = self.cab[key]([out, new_stuff])
                out = self.rrb_u[key](out)
                
        out = self.terminal_module(out)
        out = F.interpolate(out, size=(H['s1'], W['s1']), mode=self.terminal_upsample_mode, align_corners=False)
        out = out.view(B, N, -1, H['s1'], W['s1'])
        return out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

class ZeroModule(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)

class ImageEncoder(nn.Module):
    def __init__(self, resnet, terminal_module_dict, freeze_bn=False):
        super().__init__()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.freeze_bn = freeze_bn
        self.terminal_module_dict = terminal_module_dict
    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
    def forward(self, x):
        B, N, _, H, W = x.size()
        x = x.reshape(B*N, 3, H, W)
        feats = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        feats['s2'] = x
        x = self.maxpool(x)
        x = self.layer1(x)
        feats['s4'] = x
        x = self.layer2(x)
        feats['s8'] = x
        x = self.layer3(x)
        feats['s16'] = x
        x = self.layer4(x)
        feats['s32'] = x
        for name, layer in self.terminal_module_dict.items():
            feats[name] = layer(feats[name])
        feats['s2'] = feats['s2'].view(B, N, -1, H//2, W//2)
        feats['s4'] = feats['s4'].view(B, N, -1, H//4, W//4)
        feats['s8'] = feats['s8'].view(B, N, -1, H//8, W//8)
        feats['s16'] = feats['s16'].view(B, N, -1, H//16, W//16)
        feats['s32'] = feats['s32'].view(B, N, -1, H//32, W//32)
        return feats

def conv_block(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True,
               batch_norm=True, relu=True):
    layers = []
    layers.append(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=bias))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_planes))
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class BasicBlock(nn.Module):
    expansion = 1
    conv3x3 = lambda in_planes, out_planes, stride=1, dilation=1 : nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, bias=False, dilation=dilation)
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, use_bn=True):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = BasicBlock.conv3x3(inplanes, planes, stride, dilation=dilation)

        if use_bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = BasicBlock.conv3x3(planes, planes, dilation=dilation)

        if use_bn:
            self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class AnnoEncoder(nn.Module):
    def __init__(self, input_dim, init_dim, res_dims, out_dims,
                 use_bn=True, use_terminal_bn=True):
        super().__init__()
        self.conv_block = conv_block(input_dim, init_dim, kernel_size=3, stride=2, padding=1, batch_norm=use_bn)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ds1 = nn.Conv2d(init_dim, res_dims[0], kernel_size=3, padding=1, stride=2)
        self.res1 = BasicBlock(init_dim, res_dims[0], stride=2, downsample=ds1, use_bn=use_bn)
        ds2 = nn.Conv2d(res_dims[0], res_dims[1], kernel_size=3, padding=1, stride=2)
        self.res2 = BasicBlock(res_dims[0], res_dims[1], stride=2, downsample=ds2, use_bn=use_bn)
        ds3 = nn.Conv2d(res_dims[1], res_dims[2], kernel_size=3, padding=1, stride=2)
        self.res3 = BasicBlock(res_dims[1], res_dims[2], stride=2, downsample=ds3, use_bn=use_bn)

        self.label_pred1 = conv_block(
            res_dims[0], out_dims[0], kernel_size=3, stride=1, padding=1, batch_norm=use_terminal_bn
        )
        self.label_pred2 = conv_block(
            res_dims[1], out_dims[1], kernel_size=3, stride=1, padding=1, batch_norm=use_terminal_bn
        )
        self.label_pred3 = conv_block(
            res_dims[2], out_dims[2], kernel_size=3, stride=1, padding=1, batch_norm=use_terminal_bn
        )
        self.res_dims = res_dims

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, y, feature=None):
        """
        Args:
            y (FloatTensor): of size (B, S, C+1, H, W) with the background segmentation in channel 0
            feature (FloatTensor): optional
        Returns:
            FloatTensor: of size (B*S, D, H/16, W/16)
        """
        # label_mask: batch*support,prob,H,W
        assert y.dim() == 5
        B, N, Cp1, H, W = y.size()
        y = y.reshape(B*N, Cp1, H, W)
        y = y[:,1:]            # (B*S, 1, H, W)
        y = self.conv_block(y) # (B*S, _, H/2, W/2)
        y = self.pool(y)       # (B*S, _, H/4, W/4)

        to_out = {}
        y = self.res1(y)       # (B*S, _, H/8, W/8)
        to_out['s8']  = y
        y = self.res2(y)       # (B*S, _, H/16, W/16)
        to_out['s16'] = y
        y = self.res3(y)       # (B*S, _, H/32, W/32)
        to_out['s32'] = y

        out = {}
        out['s8']  = self.label_pred1(to_out['s8']).view(B, N, -1, H//8, W//8)
        out['s16'] = self.label_pred2(to_out['s16']).view(B, N, -1, H//16, W//16)
        out['s32'] = self.label_pred3(to_out['s32']).view(B, N, -1, H//32, W//32)
        
        return out
    
class AnnoEncoderAvgPool(nn.Module):
    """With this anno encoder, the GP output space is NOT learnt. The GP outputs _masks_.
    """
    def __init__(self):
        super().__init__()
        self.avg_pool_s8 = nn.AvgPool2d((8,8))
        self.avg_pool_s16 = nn.AvgPool2d((16,16))
        self.avg_pool_s32 = nn.AvgPool2d((32,32))

    def forward(self, y, feature=None):
        """
        Args:
            y (FloatTensor): of size (B, S, C+1, H, W) with the background segmentation in channel 0
            feature (FloatTensor): optional
        Returns:
            FloatTensor: of size (B*S, D, H/16, W/16)
        """
        # label_mask: batch*support,prob,H,W
        assert y.dim() == 5
        B, N, Cp1, H, W = y.size()
        y = y.reshape(B*N, Cp1, H, W)
        y = y[:,1:]
        out = {}
        out['s8']  = self.avg_pool_s8(y).view(B, N, -1, H//8, W//8)
        out['s16'] = self.avg_pool_s16(y).view(B, N, -1, H//16, W//16)
        out['s32'] = self.avg_pool_s32(y).view(B, N, -1, H//32, W//32)
        return out
    

class DGPModel(nn.Module):
    def __init__(self, kernel, covariance_output_mode='none', stride=2,
                covar_size=1, sigma_noise=1e-1):
        super().__init__()
        self.kernel = kernel
        self.sigma_noise = sigma_noise
        self.covariance_output_mode = covariance_output_mode
        self.covar_size = covar_size
        self.stride = stride
        assert covariance_output_mode in ['none', 'concatenate variance']
        if covariance_output_mode == 'concatenate variance':
            assert covar_size > 0, 'Covar neighbourhood size must be larger than 0 if using concatenate variance'
    def _sample_points_strided_grid(self, enc_images, enc_segmentations):
        """Downsample images by a factor of self.stride and pick those points
        """
        B, N, C, H, W = enc_segmentations.size()
        B, N, D, H, W = enc_images.size()
        stride = self.stride
        Hs, Ws = int(H//stride), int(W//stride)
        enc_segmentations = F.interpolate(enc_segmentations.view(B*N, C, H, W), size=(Hs, Ws), mode="nearest")
        enc_segmentations = enc_segmentations.reshape(B, N, C, Hs, Ws)
        enc_images = F.interpolate(enc_images.view(B*N, D, H, W), size=(Hs, Ws), mode="nearest")
        enc_images = enc_images.reshape(B, N, D, Hs, Ws)
        x_s = enc_images.permute(0,1,3,4,2).reshape(B, N*Hs*Ws, D)
        y_s = enc_segmentations.permute(0,1,3,4,2).reshape(B, N*Hs*Ws, C)
        return x_s, y_s
    def learn(self, enc_images, enc_segmentations,orig_segmentations=None):
        """
        """
        B, N, C, H, W = enc_segmentations.size()
        B, N, D, H, W = enc_images.size()
        E = 1
        enc_segmentations = enc_segmentations.reshape(B,N,E,C//E,H,W).permute(0,2,1,3,4,5).reshape(B*E,N,C//E,H,W)
        enc_images = enc_images.reshape(B,N,E,D//E,H,W).permute(0,2,1,3,4,5).reshape(B*E,N,D//E,H,W)

        x_s, y_s = self._sample_points_strided_grid(enc_images, enc_segmentations)
        B, S, _ = x_s.size()
        sigma_noise = self.sigma_noise * torch.eye(S, device=x_s.device)[None,:,:]
        K_ss = self.kernel(x_s, x_s) + sigma_noise
        L = torch.linalg.cholesky(K_ss)#torch.cholesky(K_ss)
        alpha = self.tri_solve(L.permute(0,2,1), self.tri_solve(L, y_s), lower=False)
        return L, alpha, x_s
    
    def tri_solve(self, L, b, lower=True):
        return torch.triangular_solve(b, L, upper=not lower)[0]

    def _get_covar_neighbours(self,v_q):
        """ Converts covariance of the form (B,H,W,H,W) to (B,H,W,K**2) where K is the covariance in a local neighbourhood around each point
        and M*M = Q
        """
        K = self.covar_size
        B,H,W,H,W = v_q.shape
        v_q = F.pad(v_q,4*(K//2,))#pad v_q
        delta = torch.stack(torch.meshgrid(torch.arange(-(K//2),K//2+1),torch.arange(-(K//2),K//2+1)),dim=-1)
        positions = torch.stack(torch.meshgrid(torch.arange(K//2,H+K//2),torch.arange(K//2,W+K//2)),dim=-1)
        neighbours = positions[:,:,None,None,:]+delta[None,:,:]
        points = torch.arange(H*W)[:,None].expand(H*W,K**2)
        v_q_neigbours = v_q.reshape(B,H*W,H+K-1,W+K-1)[:,points.flatten(),neighbours[...,0].flatten(),neighbours[...,1].flatten()].reshape(B,H,W,K**2)
        return v_q_neigbours

    def _get_covariance(self, x_q, K_qs, L):
        """
        Returns:
            torch.Tensor (B, Q, Q)
        """
        B, Q, S = K_qs.size()        
        K_qq = self.kernel(x_q, x_q) # B, Q, Q
        v = self.tri_solve(L,K_qs.permute(0,2,1))
        v_q = K_qq - torch.einsum('bsq,bsk->bqk',v,v)
        return v_q

    def forward(self, encoded_images, online_model):
        """
        """
        B,M,D,H,W = encoded_images.size()
        E = 1
        encoded_images = encoded_images.reshape((B,M,E,D//E,H,W)).permute(0,2,1,3,4,5).reshape((B*E,M,D//E,H,W))

        x_q = encoded_images.permute(0,1,3,4,2).reshape(B*E*M,H*W,D//E) # B*M go together because we don't want covariance between different query images
        L,alpha,x_s = online_model
        BE,S,C = alpha.size()
        K_qs = self.kernel(x_q,x_s)
        f_q = K_qs@alpha
        BE,Q,C = f_q.shape
        if self.covariance_output_mode == 'concatenate variance':
            v_q = self._get_covariance(x_q, K_qs, L)
            v_q = v_q.reshape(BE*M,H,W,H,W)
            v_q_neighbours = self._get_covar_neighbours(v_q)
            out = torch.cat([f_q, v_q_neighbours.view(BE, Q, self.covar_size**2)], dim=2)
            out = out.reshape(B, E, M, H, W, C+self.covar_size**2).permute(0, 2, 1, 5, 3, 4).reshape(B,M,E*(C+self.covar_size**2),H,W)
        else:
            f_q = f_q.reshape(B,E,M,H,W,C).permute(0,2,1,5,3,4).reshape(B,M,E*C,H,W)
            out = f_q
        return out
    def __str__(self):
        return f"{str(self.kernel)}"
    

class FSSLearner(nn.Module):
    def __init__(self, image_encoder, anno_encoder, model_dict, upsampler):
        super().__init__()
        self.image_encoder = image_encoder
        self.anno_encoder = anno_encoder
        self.model_dict = model_dict
        self.upsampler = upsampler
    def learn(self, images, segmaps, classes):
        """
        Args:
            images (Tensor(B N 3 H W)): 
            segmaps (LongTensor(B N C H W)):
            classes (LongTensor(B N C)): Maps annotation ids to class ids. Does not contain background.
        Returns:
            online_model
        """
        #events = [torch.cuda.Event(enable_timing=True) for i in range(3 + len(self.model_dict))]
        #events[0].record()
        encoded_images = self.image_encoder(images)        # dict of (B, N, D, H, W)
        #events[1].record()
        encoded_segmentations = self.anno_encoder(segmaps,feature=images) # dict of (B, N, C, H, W)
        #events[2].record()
        online_models = {}
        #i = 3
        for key, model in self.model_dict.items():
            online_models[key] = model.learn(
                encoded_images[key],
                encoded_segmentations[key],
                orig_segmentations=segmaps
            )
        #    events[i].record()
        #    i = i + 1
        #torch.cuda.synchronize()
        #new_times = [t0.elapsed_time(t1) for t0, t1 in zip(events[:-1], events[1:])]
        #if len(GLOBALS['times_learn']) == 0:
        #    GLOBALS['times_learn'] = new_times
        #else:
        #    GLOBALS['times_learn'] = [a+b for a, b in zip(GLOBALS['times_learn'], new_times)]
            
        return online_models

    def forward(self, images, online_models, segmaps):
        """
        Args:
            images (Tensor(B N 3 H W)): 
            online_models:
            segmaps (None or Tensor(B,N,C,H,W)):
        Returns:
            Tensor(B, N, C, H, W)
        """
        #events = [torch.cuda.Event(enable_timing=True) for i in range(3 + len(self.model_dict))]
        B, N, _, H, W = images.size()
        #events[0].record()
        encoded_images = self.image_encoder(images) # dict of (B, N, D, H, W)
        #events[1].record()
        predicted_segmentation_encodings = {}       # dict of (B, N, C, H, W)
        #i = 2
        for key, model in self.model_dict.items():
            predicted_segmentation_encodings[key] = model(
                encoded_images[key], online_models[key]
            )
        #    events[i].record()
        #    i = i + 1
        segscores = self.upsampler(predicted_segmentation_encodings, encoded_images)
        #events[i].record()
        #torch.cuda.synchronize()
        #new_times = [t0.elapsed_time(t1) for t0, t1 in zip(events[:-1], events[1:])]
        #if len(GLOBALS['times_forward']) == 0:
        #    GLOBALS['times_forward'] = new_times
        #else:
        #    GLOBALS['times_forward'] = [a+b for a, b in zip(GLOBALS['times_forward'], new_times)]

        #if GLOBALS['number of forward calls'] >= 1000:
        #    print("\nTime learn")
        #    print("  ".join([f"{time / 1000:7.1f} ms" for time in GLOBALS['times_learn']]))
        #    print("\nTime forward")
        #    print("  ".join([f"{time / 1000:7.1f} ms" for time in GLOBALS['times_forward']]))
        #    raise ValueError()

        return segscores

    def __str__(self):
        return  f"FSSlearner-{str(self.model)}"


class DGP(nn.Module):
    def __init__(self, learner, debugging=False, loss=None, support_set_sampler=None):
        super().__init__()
        self.learner = learner
        if loss is None:
            loss = nn.CrossEntropyLoss(ignore_index=255)
        self.loss = loss
        self.calculate_losses = True
        self.support_set_sampler = support_set_sampler
        GLOBALS['DEBUG'] = debugging

    def get_loss_idfs(self):
        return ['query seg CE', 'query seg iou']

    def get_losses(self, segscores, segs, segannos, classes):
        assert segs.size() == segannos.size(),'Segs and segannos must be of same shape'
        B, Q, Cp1, H, W = segscores.size()
        loss = self.loss(segscores.view(B*Q, Cp1, H, W), segannos.view(B*Q, H, W))
        iou = self._get_iou(segs, segannos, classes)
        return {'total': loss, 'partial': {'query seg CE': {'val': loss, 'N': B*Q}, 'query seg iou': iou}}

    def _get_iou(self, segs, segannos, classes):
        """Note that this is an IoU-measure used only during training. It is not the same as the
        IoU reported during evaluation and more reminiscent of the IoU-measure used in the Video
        Object Segmentation problem.
        Args:
            segs (LongTensor(B, Q, H, W))
            segannos (LongTensor(B, Q, H, W))
            classes (LongTensor(B, Q, C))
        Returns:
            iou (Tensor(B, Q, C))
        """
        B, Q, H, W = segs.size()
        _, _, C = classes.size()
        iou = torch.zeros((B, Q, C), device=classes.device)
        segs = segs.clone()
        segs[segannos == 255] = 255
        for c in range(1, C+1):
            mask_pred = (segs == c)
            mask_anno = (segannos == c)
            intersection = (mask_pred * mask_anno).sum(dim=(2,3))
            union = mask_pred.sum(dim=(2,3)) + mask_anno.sum(dim=(2,3)) - intersection
            iou[:, :, c-1] = (intersection + 1e-5) / (union + 1e-5)
        return {'val': iou.mean(), 'N': B*Q*C}

    def _seg_to_segmap(self, segmentation, classes):
        """
        Args:
            segmentation (LongTensor (B N H W)):
            classes      (LongTensor (B N C):
        Returns:
            FloatTensor (B, N, C, H W)
        """
        B, N, C = classes.size()
        seg_lst = [(segmentation == c) for c in range(0, C+1)]
        segmap = torch.stack(seg_lst, dim=2).float()
        return segmap
    
    def forward(self, data, state=None, visualize=False, epoch=0):
        """We denote sizes with single capital letters. B is batch size, S is support set size,
        Q is query set size, C is the number of classes (often 1), H height, and W width.
        Args:
            data (dict)
                support_images        (None or Tensor(B, S, 3, H, W))
                support_segmentations (None or LongTensor(B, S, H, W))
                support_classes       (None or LongTensor(B, N, C))
                query_images          (None or Tensor(B, Q, 3, H, W))
                query_segmentations   (None or LongTensor(B, Q, 3, H, W))
                query_classes         (None or LongTensor(B, N, C))
        """
        GLOBALS['data identifier'] = data['identifier']
        #GLOBALS['number of forward calls'] = GLOBALS['number of forward calls'] + 1
        if state is None:
            state = {}
        if data.get('support_images') is not None:
            if self.support_set_sampler is None:
                state['distributions'] = self.learner.learn(
                    data['support_images'],
                    self._seg_to_segmap(data['support_segmentations'], data['support_classes']),
                    data['support_classes'])
            else:
                state['distributions'] = self.learner.learn(*self.support_set_sampler(
                    data['support_images'],
                    self._seg_to_segmap(data['support_segmentations'], data['support_classes']),
                    data['support_classes']))

        if data.get('query_images') is not None:
            if data.get('query_segmentations') is not None:
                query_segmentations = self._seg_to_segmap(data['query_segmentations'], data['query_classes'])
            else:
                query_segmentations = None
            output_segscores = self.learner(
                data['query_images'],
                state['distributions'],
                query_segmentations)
            output_segs = output_segscores.argmax(dim=2)
            
        if self.calculate_losses:
            losses = self.get_losses(
                output_segscores, output_segs, data['query_segmentations'], data['query_classes'])
        else:
            losses = None

        output = {
            'query_segmentations': output_segs,
            'query_segscores': output_segscores
        }

        # Ugly debugging code
        self.debug = GLOBALS

        return output, state, losses

