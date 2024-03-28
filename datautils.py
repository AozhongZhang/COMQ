import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import random
import os
import scipy.io as sio
import glob
import re
import pickle

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

def parse_imagenet_val_labels(data_dir):
    meta_path = os.path.join(data_dir, 'meta.mat')
    meta = sio.loadmat(meta_path, squeeze_me=True)['synsets']
    nums_children = list(zip(*meta))[4]
    meta = [meta[idx] for idx, num_children in enumerate(nums_children)
                if num_children == 0]
    idcs, wnids = list(zip(*meta))[:2]
    idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}

    val_path = os.path.join(data_dir, 'ILSVRC2012_validation_ground_truth.txt')
    val_idcs = np.loadtxt(val_path) 
    val_wnids = [idx_to_wnid[idx] for idx in val_idcs]
   
    label_path = os.path.join(data_dir, 'wnid_to_label.pickle')  
    with open(label_path, 'rb') as f:
        wnid_to_label = pickle.load(f)
    
    val_labels = [wnid_to_label[wnid] for wnid in val_wnids]
    return np.array(val_labels)

class Imagenet(Dataset):
    """
    Validation dataset of Imagenet
    """
    def __init__(self, data_dir, transform):
        # we can maybe pput this into diff files.
        self.Y = torch.from_numpy(parse_imagenet_val_labels(data_dir)).long()
        self.X_path = sorted(glob.glob(os.path.join(data_dir, 'ILSVRC2012_img_val/*.JPEG')), 
            key=lambda x: re.search('%s(.*)%s' % ('ILSVRC2012_img_val/', '.JPEG'), x).group(1))
        self.transform = transform

    def __len__(self):
        return len(self.X_path)
    
    def __getitem__(self, idx):
        img = Image.open(self.X_path[idx]).convert('RGB')
        y = self.Y[idx] 
        if self.transform:
            x = self.transform(img)
        return x, y


def data_loader(ds_path, batch_size, train_transform, test_transform, num_workers=8): 

    # data_dir = '../data/ILSVRC2012'
    data_dir = ds_path

    if not os.path.isdir(data_dir):
        raise Exception('Please download Imagenet2012 dataset!')
        
    train_ds = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'ILSVRC2012_img_train'),
                                                transform=train_transform)
        
    if not os.path.isfile(os.path.join(data_dir, 'wnid_to_label.pickle')):
        with open(os.path.join(data_dir, 'wnid_to_label.pickle'), 'wb') as f:
            pickle.dump(train_ds.class_to_idx, f)  

    # if not os.path.isfile('../data/ILSVRC2012/wnid_to_label.pickle'):
    #     with open('../data/ILSVRC2012/wnid_to_label.pickle', 'wb') as f:
    #         pickle.dump(train_ds.class_to_idx, f)         

    test_ds = Imagenet(data_dir, test_transform) 
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers,
                            worker_init_fn=seed_worker, generator=g)
    test_dl = DataLoader(test_ds, min(batch_size, 1024), shuffle=False,
                            num_workers=num_workers) 
         
    return train_dl, test_dl 
