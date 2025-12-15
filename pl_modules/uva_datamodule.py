import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch
from data.uva_dataset import UVAData

class EmptyDataset(Dataset):
    """Empty dataset to skip validation"""
    def __len__(self):
        return 0
    
    def __getitem__(self, idx):
        raise IndexError("EmptyDataset has no items")

class UVADataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.training.batch_size
        self.num_workers = cfg.data.num_workers

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = UVAData(self.cfg, mode='train')
            self.val_dataset = UVAData(self.cfg, mode='val')

        if stage == 'test' or stage is None:
            self.test_dataset = UVAData(self.cfg, mode='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        # Check if validation should be skipped
        skip_validation = getattr(self.cfg.validation, 'skip', False)
        if skip_validation:
            # Return an empty DataLoader to skip validation
            empty_dataset = EmptyDataset()
            return DataLoader(empty_dataset, batch_size=1, num_workers=0)
        
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)
