import os
import sys
import argparse
import datetime
import time
import math
import random

parser = argparse.ArgumentParser(description="Experiment runfile, you run experiments from this file")
parser.add_argument("--train", action="store_true", default=False)
parser.add_argument("--test", action="store_true", default=False)
parser.add_argument("--dataset", type=str,required=True)
parser.add_argument("--fold", type=int,required=True)
parser.add_argument("--test_num_support", type=int, default=1)
parser.add_argument("--add_packages_to_path",action="store_true",default=False)
parser.add_argument("--seed", type=int,default=0)
parser.add_argument("--checkpoint")
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("-d", "--device", dest="device", help="Device to run on, the cpu or gpu.",
                    type=str, default="cuda:0")
parser.add_argument("--restart", action="store_true", default=False)
args = parser.parse_args()
#Following is needed to enable running the code when not using a package solution
if args.add_packages_to_path:
    env_path = os.path.join(os.path.dirname(__file__), '..') #This may or may not work hehe
    sys.path.append(env_path)
args.fstart_id = '{dt}_{filename}'.format(dt=datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S'), filename=os.path.splitext(os.path.basename(__file__))[0])

import numpy as np
import torch
torch.set_printoptions(edgeitems=4, linewidth=117)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import torchvision as tv

from local_config import config
import fss.models.dgp as dgp
from fss.models.kernels import RBF,NormalizedRBF,CosineSim,SoftPlusCosineSim,Trivial,LinearKernel
import fss.datasets.pascal as pascal
import fss.trainers.fss_trainer as fss_trainer
import fss.evaluation.fss_evaluator as fss_evaluator
from fss.datasets.transform import Compose, Normalize, RandomRotate90, Resize, ToTensor, RandomHorizontalFlip

def train(model, device, dataset, fold, restart, seed):
    params_bb = ({param for param in model.learner.image_encoder.parameters()}
                - {param for param in model.learner.image_encoder.terminal_module_dict.parameters()})
    params_new = {param for param in model.parameters()} - params_bb
    parameters = [{'params': [param for param in params_new if param.requires_grad]},
                  {'params': [param for param in params_bb if param.requires_grad], 'lr': 1e-6}]
    optimizer = optim.AdamW(parameters, lr=5e-5, weight_decay=1e-3)
    lr_sched = optim.lr_scheduler.LambdaLR(optimizer, (lambda n:
                                                       1.0 if n <= 10 else
                                                       0.1))
    num_epochs = 20

    train_transform = Compose([RandomHorizontalFlip(),
                                Resize((384, 384)),
                                ToTensor(),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_transform = Compose([Resize((384, 384)),
                                ToTensor(),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    training_data = pascal.DatasetPASCAL(
        datapath = config['pascal_path'],
        fold  = fold,
        transform = train_transform,
        split      = 'train',
        shot   = 1,
        mode   = "training",
        data_list_path = os.path.join(config['workspace_path'], 'data_splits', 'pascal')
        )
    validation_data = pascal.DatasetPASCAL(
        datapath = config['pascal_path'],
        fold  = fold,
        transform = val_transform,
        split      = 'val',
        shot   = 1,
        mode   = "training",
        data_list_path = os.path.join(config['workspace_path'], 'data_splits', 'pascal')
        )
    print("Loaded training set with", len(training_data), "samples")
    print("Loaded validation set with", len(validation_data), "samples")
    training_sampler = torch.utils.data.RandomSampler(training_data, num_samples=8000, replacement=True)
    validation_sampler = torch.utils.data.RandomSampler(validation_data, num_samples=2000, replacement=True)
    training_loader = DataLoader(training_data, sampler=training_sampler, batch_size=8, num_workers=16)
    validation_loader = DataLoader(validation_data, sampler=validation_sampler, batch_size=20, num_workers=16)

    trainer = fss_trainer.FSSTrainer(
        model                = model,
        optimizer            = optimizer,
        lr_sched             = lr_sched,
        train_loader         = training_loader,
        val_loaders          = [validation_loader],
        checkpoint_path      = config['checkpoint_path'],
        visualization_path   = os.path.join(config['visualization_path']),
        save_name            = f"{os.path.splitext(os.path.basename(__file__))[0]}_{dataset}_{fold}_{seed}",
        device               = device,
        checkpoint_epochs    = [num_epochs],
        print_interval       = 100,
        visualization_epochs = [2, num_epochs],
    )
    if not restart:
        trainer.load_checkpoint()
    trainer.train(num_epochs)

def test(model, device, dataset, fold, num_support, seed):
    test_transform = Compose([Resize((448,448)),
                              ToTensor(),
                              Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    data = pascal.DatasetPASCAL(
        datapath = config['pascal_path'],
        fold  = fold,
        transform = test_transform,
        split      = 'val',
        shot   = num_support,
        mode   = "evaluation",
        data_list_path = os.path.join(config['workspace_path'], 'data_splits', 'pascal')
    )
    sampler = pascal.SequentialSampler(data, 5000)
    data = DataLoader(data, sampler=sampler, batch_size=20, num_workers=11)
        
    evaluator = fss_evaluator.FSSEvaluator(
        visualization_path     = os.path.join(
            config['visualization_path'],
            f"{os.path.splitext(os.path.basename(__file__))[0]}_{dataset}_{fold}_{seed}"),
        device                 = device,
    )
    evaluator.evaluate(model, data)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def torch_init(to_device):
    cuda_avail = torch.cuda.is_available()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cpu")
    if cuda_avail and 'cuda' in to_device:
        device = torch.device(to_device)
        torch.cuda.set_device(device)

    return cuda_avail, device

def main(args):
    print("Started script: {}, with pytorch {}".format(os.path.basename(__file__), torch.__version__))
    print(args)
    print(f"Seeding with seed: {args.seed}")
    seed_all(args.seed)
    cuda_avail, device = torch_init(args.device)
    print("pytorch using device", device)
    os.makedirs(config['plot_path'],exist_ok=True)
    os.makedirs(config['visualization_path'],exist_ok=True)
    if args.debug:
        os.makedirs(os.path.join(config['visualization_path'],"debugging"),exist_ok=True)#hacky way to make sure debugging folder exist later
    os.makedirs(config['checkpoint_path'],exist_ok=True)

    resnet = tv.models.resnet50(pretrained=True)
    covar_size = 5
    covariance_output_mode = 'concatenate variance'
    freeze_bn = True
    
    model = dgp.DGP(
        learner = dgp.FSSLearner(
            image_encoder = dgp.ImageEncoder(
                resnet               = resnet,
                terminal_module_dict = nn.ModuleDict({
                    's32': nn.Conv2d(2048, 512, 1, 1, 0),
                    's16': nn.Conv2d(1024, 512, 1, 1, 0),
                }),
                freeze_bn            = freeze_bn
            ),
            anno_encoder  = dgp.AnnoEncoder(
                input_dim         = 1,
                init_dim          = 16,
                res_dims          = [32, 64, 64],
                out_dims          = [32, 64, 64],
                use_bn            = True,
                use_terminal_bn   = True
            ),
            model_dict    = nn.ModuleDict({
                's32': dgp.DGPModel(
                    kernel                 = RBF(length=1/(512**.25)),
                    covariance_output_mode = covariance_output_mode,
                    covar_size             = covar_size,
                    stride                 = 1,
                ),
                's16': dgp.DGPModel(
                    kernel                 = RBF(length=1/(512**.25)),
                    covariance_output_mode = covariance_output_mode,
                    covar_size             = covar_size,
                    stride                 = 2,
                ),
            }),
            upsampler     = dgp.DFN(
                internal_dim       = 256,
                feat_input_modules = nn.ModuleDict({
                    's8': nn.Identity(),
                    's4': nn.Identity(),
                }),
                pred_input_modules = nn.ModuleDict({
                    's32': nn.Identity(),
                    's16': nn.Identity(),
                }),
                rrb_d_dict = nn.ModuleDict({
                    's32': dgp.RRB(64 + covar_size**2, 256),
                    's16': dgp.RRB(64 + covar_size**2, 256),
                    's8': dgp.RRB(512, 256),
                    's4': dgp.RRB(256, 256),
                }),
                cab_dict = nn.ModuleDict({
                    's32': dgp.CAB(2 * 256, 256),
                    's16': dgp.CAB(2 * 256, 256),
                    's8': dgp.CAB(2 * 256, 256),
                    's4': dgp.CAB(2 * 256, 256),
                }),
                rrb_u_dict = nn.ModuleDict({
                    's32': dgp.RRB(256, 256),
                    's16': dgp.RRB(256, 256),
                    's8': dgp.RRB(256, 256),
                    's4': dgp.RRB(256, 256),
                }),
                terminal_module = nn.Conv2d(256, 2, 1, 1, 0)
            ),            
        ),
        debugging = args.debug,
        loss = nn.CrossEntropyLoss(weight=torch.Tensor([1.0, 4.0]), ignore_index=255),
    )
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['net'])
    model.to(device)
    if args.train:
        train(model, device, args.dataset, args.fold, args.restart, args.seed)
    if args.test:
        test(model, device, args.dataset, args.fold, args.test_num_support, args.seed)

#Run main
main(args)
