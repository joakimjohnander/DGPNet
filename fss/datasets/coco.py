"""Dataset code adapted from https://github.com/juhongm999/hsnet.git
"""

import os
import pickle

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import PIL.Image as Image
import numpy as np


class DatasetCOCO(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, mode, data_list_path, class_balance=True):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 80
        self.benchmark = 'coco'
        self.shot = shot
        self.base_path = datapath
        self.transform = transform
        self.data_list_path = data_list_path
        self.mode = mode
        self.num_query_per_episode = 1

        self.class_balance = class_balance

        self.class_ids = self.build_class_ids()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.img_metadata = self.build_img_metadata()

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        query_image, query_mask, support_images, support_masks, query_name, support_names, class_sample, org_qry_imsize = self.load_frame(idx)

        query_image = query_image
        query_mask = query_mask.long()
        original_query_mask = query_mask.clone()[None]
        query_image, query_mask = self.transform(query_image, query_mask)
        query_image = query_image[None]
        query_mask = query_mask[None]
        

        support_set = [self.transform(s_i,s_cm) for s_i,s_cm in zip(support_images,support_masks)]
        support_images, support_masks = torch.stack([i for i,_ in support_set]),torch.stack([m for _,m in support_set])
        support_masks = support_masks.long()
        
        classes = torch.tensor([[class_sample]], dtype=torch.int64).reshape(1, 1)        

        output = {'query_images': query_image,'query_segmentations': query_mask,
                  'support_images': support_images, 'support_segmentations': support_masks,
                  'query_classes': classes, 'support_classes': classes,
                  'identifier': f"class {classes.item()}, query {idx}"}
        
        if self.mode == 'evaluation':
            orig_query_segs = torch.full((self.num_query_per_episode, 640, 640), 255, dtype=torch.int64)
            for idx, sample in enumerate(original_query_mask):
                H, W = sample.size()
                orig_query_segs[idx, :H, :W] = sample
            output['original_query_segmentations'] = orig_query_segs
            output['original_query_sizes'] = torch.tensor(org_qry_imsize[::-1])[None] # ::-1 since imsize comes from PIL w,h format
        
        return output

    def get_classes(self):
        return self.class_ids

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold + self.nfolds * v for v in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        class_ids = class_ids_trn if self.split == 'trn' else class_ids_val

        return class_ids

    def build_img_metadata_classwise(self):
        print(f'{self.data_list_path}/{self.split}/fold{self.fold}.pkl')
        with open(f'{self.data_list_path}/{self.split}/fold{self.fold}.pkl', 'rb') as f:
            img_metadata_classwise = pickle.load(f)
        return img_metadata_classwise

    def build_img_metadata(self):
        img_metadata = [(c, img)
                        for c, img_list in self.img_metadata_classwise.items()
                        for img in img_list]
        return img_metadata

    def read_mask(self, name):
        mask_path = os.path.join(self.base_path, 'annotations', name)
        mask = torch.tensor(np.array(Image.open(mask_path[:mask_path.index('.jpg')] + '.png')))
        return mask

    def load_frame(self, idx):
        if self.class_balance:
            class_sample = np.random.choice(self.class_ids, 1, replace=False)[0]
            query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        else:
            class_sample, query_name  = self.img_metadata[idx]
            
        query_img = Image.open(os.path.join(self.base_path, query_name)).convert('RGB')
        query_mask = self.read_mask(query_name)

        org_qry_imsize = query_img.size

        query_mask[query_mask != class_sample + 1] = 0
        query_mask[query_mask == class_sample + 1] = 1

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        support_imgs = []
        support_masks = []
        for support_name in support_names:
            support_imgs.append(Image.open(os.path.join(self.base_path, support_name)).convert('RGB'))
            support_mask = self.read_mask(support_name)
            support_mask[support_mask != class_sample + 1] = 0
            support_mask[support_mask == class_sample + 1] = 1
            support_masks.append(support_mask)

        return query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize

