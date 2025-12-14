import pytorch_lightning as pl
from torch.utils.data import DataLoader
# from data.citywalk_dataset import CityWalkDataset
from data.teleop_dataset import TeleopDataset

class TeleopDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.training.batch_size
        self.num_workers = cfg.data.num_workers

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = TeleopDataset(self.cfg, mode='train')
            self.val_dataset = TeleopDataset(self.cfg, mode='val')

        if stage == 'test' or stage is None:
            self.test_dataset = TeleopDataset(self.cfg, mode='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)
