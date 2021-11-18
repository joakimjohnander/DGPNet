"""Dataset code adapted from https://github.com/juhongm999/hsnet.git
"""

import os

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np


PASCAL_CLASSNAMES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
    'train', 'tvmonitor'
]
PASCAL_CLASSNAMES_NOT_IN_COCO_TRAIN = [
    ['aeroplane', 'boat', 'chair', 'diningtable', 'dog', 'person'],
    ['horse', 'sofa', 'bicycle', 'bus'],
    ['bird', 'car', 'pottedplant', 'sheep', 'train', 'tvmonitor'],
    ['bottle', 'cow', 'cat', 'motorbike'],
]

class SequentialSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, num_samples):
        self.data_source = data_source
        self.datalen = len(data_source)
        self.num_samples = num_samples
    def __iter__(self):
        return iter([idx % self.datalen for idx in range(self.num_samples)])
    def __len__(self):
        return self.num_samples


class DatasetPASCAL(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, mode,data_list_path):
        if split in ['val', 'test']:
            self.split = 'val'
        elif split == 'not_in_coco_train':
            self.split = 'not_in_coco_train'
        else:
            self.split = 'train'
        self.data_list_path = data_list_path
        self.fold = fold
        self.nfolds = 4
        self.nclass = 20
        self.benchmark = 'pascal'
        self.shot = shot
        if mode == "evaluation":
            self.use_original_imgsize = True
        else:
            self.use_original_imgsize = True # xd /Johan
        self.mode = mode
        self.num_query_per_episode = 1
        self.img_path = os.path.join(datapath, 'JPEGImages/')
        self.ann_path = os.path.join(datapath, 'SegmentationClassAug/')
        self.transform = transform

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()

    def __len__(self):
        return len(self.img_metadata) if self.split == 'train' else 1000

    def __getitem__(self, idx):
        idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_cmask, support_imgs, support_cmasks, org_qry_imsize = self.load_frame(query_name, support_names)
        
        query_mask = self.extract_ignore_idx(query_cmask, class_sample)
        original_query_mask = query_mask
        query_img, query_mask = self.transform(query_img, torch.stack(query_mask))
        
        support_masks = self.extract_ignore_idx(support_cmasks, class_sample)
        support_set = [self.transform(s_i,s_cm) for s_i,s_cm in zip(support_imgs,support_masks)]
        support_imgs, support_masks = torch.stack([i for i,_ in support_set]),torch.stack([m for _,m in support_set])
        
        classes = torch.tensor([[class_sample]], dtype=torch.int64).reshape(1, 1)        
        
        output = {'query_images': query_img[None],'query_segmentations': query_mask,
                  'support_images': support_imgs, 'support_segmentations': support_masks,
                  'query_classes': classes, 'support_classes': classes,
                  'identifier': f"class {classes.item()}, query {idx}"}
        
        if self.mode == 'evaluation':
            orig_query_segs = torch.full((self.num_query_per_episode, 500, 500), 255, dtype=torch.int64)
            for idx, sample in enumerate(original_query_mask):
                H, W = sample.size()
                orig_query_segs[idx, :H, :W] = sample
            output['original_query_segmentations'] = orig_query_segs
            output['original_query_sizes'] = torch.tensor(org_qry_imsize[::-1])[None] # ::-1 since imsize comes from PIL w,h format
        return output

    def extract_ignore_idx(self, cmasks, class_id):
        if not isinstance(cmasks,list):
            cmasks = [cmasks]
        masks = []
        for cmask in cmasks:
            mask = cmask.long()#it might be byte, torch doesnt like that :'(
            ignore = (mask==255)
            mask[mask != class_id + 1] = 0
            mask[mask == class_id + 1] = 1
            mask[ignore] = 255
            masks.append(mask)
        return masks

    def get_classes(self):
        return self.class_ids

    def load_frame(self, query_name, support_names):
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        support_imgs = [self.read_img(name) for name in support_names]
        support_masks = [self.read_mask(name) for name in support_names]

        org_qry_imsize = query_img.size

        return query_img, query_mask, support_imgs, support_masks, org_qry_imsize

    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = torch.tensor(np.array(Image.open(os.path.join(self.ann_path, img_name) + '.png')))
        return mask

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + '.jpg')

    def sample_episode(self, idx):
        query_name, class_sample = self.img_metadata[idx]

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        return query_name, support_names, class_sample

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]

        if self.split == 'train':
            return class_ids_trn
        elif self.split == 'not_in_coco_train':
            class_ids = [PASCAL_CLASSNAMES.index(classname)
                         for classname in PASCAL_CLASSNAMES_NOT_IN_COCO_TRAIN[self.fold]]
            return class_ids
        else:
            return class_ids_val

    def build_img_metadata(self):

        def read_metadata(split, fold_id):
            fold_n_metadata = f'{self.data_list_path}/{split}/fold{fold_id}.txt'
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        if self.split == 'train':  # For training, read image-metadata of "the other" folds
            for fold_id in range(self.nfolds):
                if fold_id == self.fold:  # Skip validation fold
                    continue
                img_metadata += read_metadata(self.split, fold_id)
        elif self.split == 'not_in_coco_train':
            for fold_id in range(self.nfolds):
                img_metadata += read_metadata('val', fold_id)
                img_metadata = [(image_fname, class_idx)
                                for image_fname, class_idx in img_metadata
                                if class_idx in self.class_ids]
        elif self.split == 'val':  # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata(self.split, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.split)

        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))

        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]
        print(f"Loaded (class_idx, num_samples): {[(c, len(lst)) for c, lst in img_metadata_classwise.items()]}")
        return img_metadata_classwise


