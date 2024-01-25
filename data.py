import os
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, TensorDataset, Dataset
import numpy as np
import pytorch_lightning as pl
#import lightning.pytorch as pl
import kornia

from LRHR_PKL_dataset import LRHR_PKLDataset
# from pytorch_lightning.cli import DATAMODULE_REGISTRY

def resize(x, size):
    #TODO: reimplement, move to utils?
    return kornia.geometry.transform.resize(x, size, antialias=True, align_corners=False)

def resize_uint8(x, size):
    #TODO: reimplement, move to utils?
    return kornia.geometry.transform.resize(x*1.0, size, antialias=True, align_corners=False).clamp(0, 255).type(torch.uint8)


class Resize(nn.Module):
    """
    resize float data and quantize them 
    """
    def __init__(self, size, quantize):
        super().__init__()
        self.quantize = quantize
        self.resize = kornia.geometry.transform.Resize(size=size, align_corners=False, antialias=True)
    
    def forward(self, x):
        # resize and add noise 
        y = self.resize(x)
        if self.quantize:
            y = torch.clamp(y, 0, 255).type(torch.uint8).type(torch.float)
        return y

class DummyDataset(Dataset):
    """
    A dataset that return dummy data (for lightning compatibility)
    """
    def __init__(self, size=20000):
        super().__init__()
        self.size = size

    def __getitem__(self, index):
        return torch.rand(1)

    def __len__(self):
        return self.size

# @DATAMODULE_REGISTRY
class FFHQDataModule(pl.LightningDataModule):
    def __init__(self, data_path : str = "data/dummy.npy", batch_size : int = 4, num_workers : int = 2):
        """
        data module containing FFHQ samples
        """
        super().__init__()
        # path to .npy file
        self.path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_hyperparameters()
    
    def prepare_data(self):
        self.X = torch.from_numpy(np.load(self.path, mmap_mode='r+')).permute(0, 3, 1, 2).contiguous().type(torch.float32)

    def setup(self, stage=None):
        # we use the same split as in vdvae original implementation, since we reuse their model
        np.random.seed(5)
        tr_va_split_indices = np.random.permutation(self.X.shape[0])
        train = self.X[tr_va_split_indices[:-7000]]
        valid = self.X[tr_va_split_indices[-7000:]]
        self.train_dataset = TensorDataset(train)
        self.val_dataset = TensorDataset(valid)
        self.test_dataset = self.val_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("DataModule")
        parser.add_argument("--data_path", type=str, default="data/ffhq-256.npy")
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--num_workers", type=int, default=4)
        return parent_parser
        

def load_ffhq256_validation_set(path):
    """
    return the validation set vdvae was trained on 
    """
    X = np.load(path, mmap_mode='r+')
    np.random.seed(5)
    tr_va_split_indices = np.random.permutation(X.shape[0])
    valid = X[tr_va_split_indices[-7000:]]
    valid = torch.tensor(valid).permute(0, 3, 1, 2).contiguous().type(torch.float32)
    return valid

class ImageFolderDateset(Dataset):
    """
    A dataset that return dummy data (for lightning compatibility)
    """
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        self.image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)]

    def __getitem__(self, index):
        imname = os.path.splitext(os.path.split(self.image_paths[index])[1])[0]
        return (imname, torchvision.io.read_image(self.image_paths[index]))

    def __len__(self):
        return len(self.image_paths)
        
# @DATAMODULE_REGISTRY
class LRHRDataModule(pl.LightningDataModule):
    """
    Lightning datamodule wrapper around LRHR_PKL_dataset class from https://github.com/xinntao/BasicSR
    """
    def __init__(
        self, path_train_HR : str,
        path_train_LR : str, 
        path_val_HR : str,
        path_val_LR : str,
        use_flip : bool = True,
        use_rot : bool = False,
        use_crop : bool = False,
        workers_per_loader : int = 4,
        batch_size : int = 4
        ):
        super().__init__()
        self.opt_train = {
            "dataroot_GT" : path_train_HR,
            "dataroot_LQ" : path_train_LR,
            "use_flip" : use_flip,
            "use_rot" : use_rot,
            "use_crop" : use_crop
        }
        self.opt_val = {
            "dataroot_GT" : path_val_HR,
            "dataroot_LQ" : path_val_LR,
            "use_flip" : use_flip,
            "use_rot" : use_rot,
            "use_crop" : use_crop
        }
        self.workers_per_loader = workers_per_loader
        self.batch_size = batch_size

    #def prepare_data(self):

    def setup(self, stage: str = None):
        self.val_dataset = LRHR_PKLDataset(self.opt_val)
        self.train_dataset = LRHR_PKLDataset(self.opt_train)
        self.test_dataset = self.val_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.workers_per_loader, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.workers_per_loader, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.workers_per_loader, pin_memory=True)




if __name__ == "__main__":
#    print('initializing module')
#    dmod = FFHQDataModule("data/dummy.npy", batch_size=2, lr_size=32, std=3, num_workers=2)
#    print('preparing data...')
#    dmod.prepare_data()
#    print('setup...')
#    dmod.setup()
#    print('loader...')
#    loader = dmod.val_dataloader()
#    batch = next(iter(loader))
#    print([x.shape for x in batch])
    opt_val = {
            "dataroot_GT" :"/beegfs/jprost/data/FFHQ256/FFHQ256_CEMBic_va.pklv4",
            "dataroot_LQ" : "/beegfs/jprost/data/FFHQ256/FFHQ256_CEMBic_va_X16.pklv4",
            "use_flip" : False,
            "use_rot" : False,
            "use_crop" : False
        }
    ds = LRHR_PKLDataset(opt_val)
    loader = DataLoader(ds, batch_size=4)
    batch = next(iter(loader))
    print(len(batch))
    print(batch[0].shape)
    print(batch[1].shape)
