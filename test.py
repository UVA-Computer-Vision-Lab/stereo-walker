# test.py

import pytorch_lightning as pl
import argparse
import yaml
import os
from pl_modules.citywalk_datamodule import CityWalkDataModule
from pl_modules.citywalk_feat_datamodule import CityWalkFeatDataModule
from pl_modules.stereowalk_datamodule import StereoWalkDataModule
from pl_modules.monowalk_datamodule import MonoWalkDataModule
from pl_modules.teleop_datamodule import TeleopDataModule
from pl_modules.uva_datamodule import UVADataModule
from pl_modules.stereowalk_labeled_datamodule import StereoWalkLabeledDataModule
from pl_modules.citywalker_module import CityWalkerModule
from pl_modules.citywalker_feat_module import CityWalkerFeatModule
from pl_modules.socialwalker_module import SocialWalkerModule
import torch
import glob

torch.set_float32_matmul_precision('medium')
pl.seed_everything(42, workers=True)


class DictNamespace(argparse.Namespace):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, DictNamespace(**value))
            else:
                setattr(self, key, value)


def parse_args():
    parser = argparse.ArgumentParser(description='Test UrbanNav model')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint. If not provided, the latest checkpoint will be used.')
    parser.add_argument('--save_predictions', action='store_true', help='Whether to save predictions')
    args = parser.parse_args()
    return args


def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = DictNamespace(**cfg_dict)
    return cfg


def find_latest_checkpoint(checkpoint_dir):
    """
    Finds the latest checkpoint in the given directory based on modification time.
    
    Args:
        checkpoint_dir (str): Path to the directory containing checkpoints.
    
    Returns:
        str: Path to the latest checkpoint file.
    
    Raises:
        FileNotFoundError: If no checkpoint files are found in the directory.
    """
    checkpoint_pattern = os.path.join(checkpoint_dir, '*.ckpt')
    checkpoint_files = glob.glob(checkpoint_pattern)
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in directory: {checkpoint_dir}")
    
    # Sort checkpoints by modification time (latest first)
    checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_checkpoint = checkpoint_files[0]
    return latest_checkpoint


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Create a directory for test results
    test_dir = os.path.join(cfg.project.result_dir, cfg.project.run_name, 'test')
    os.makedirs(test_dir, exist_ok=True)

    # Initialize the DataModule
    if cfg.data.type == 'citywalk':
        datamodule = CityWalkDataModule(cfg)
    elif cfg.data.type == 'citywalk_feat':
        datamodule = CityWalkFeatDataModule(cfg)
    elif cfg.data.type == 'stereowalk':
        datamodule = StereoWalkDataModule(cfg)
    elif cfg.data.type == 'monowalk':
        datamodule = MonoWalkDataModule(cfg)
    elif cfg.data.type == 'teleop':
        datamodule = TeleopDataModule(cfg)
    elif cfg.data.type == 'uva':
        datamodule = UVADataModule(cfg)
    elif cfg.data.type == 'stereowalk_labeled':
        datamodule = StereoWalkLabeledDataModule(cfg)
    else:
        raise ValueError(f"Invalid dataset: {cfg.data.type}")

    # Determine the checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    else:
        # Automatically find the latest checkpoint
        checkpoint_dir = os.path.join(cfg.project.result_dir, cfg.project.run_name, 'checkpoints')
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
        checkpoint_path = find_latest_checkpoint(checkpoint_dir)
        print(f"No checkpoint specified. Using the latest checkpoint: {checkpoint_path}")

    # Load the model from the checkpoint
    if cfg.model.type == 'citywalker':
        model = CityWalkerModule.load_from_checkpoint(checkpoint_path, cfg=cfg)
    elif cfg.model.type == 'citywalker_feat':
        model = CityWalkerFeatModule.load_from_checkpoint(checkpoint_path, cfg=cfg)
    else:
        model = SocialWalkerModule.load_from_checkpoint(checkpoint_path, cfg=cfg, strict=False)
    
    # 强制更新 datatype 和相关配置（重要！）
    model.datatype = cfg.data.type
    if cfg.data.type in ["teleop", "stereowalk_labeled"]:
        if not hasattr(model, 'test_catetories'):
            model.test_catetories = ['crowd', 'person_close_by', 'turn', 'action_target_mismatch', 'crossing', 'other']
            model.num_categories = len(model.test_catetories)
    
    model.result_dir = test_dir
    print(f"Loaded model from checkpoint: {checkpoint_path}")
    print(f"Model datatype set to: {model.datatype}")

    # Initialize Trainer
    trainer = pl.Trainer(
        default_root_dir=test_dir,
        devices=cfg.training.gpus,
        precision='16-mixed' if cfg.training.amp else 32,
        accelerator='ddp' if cfg.training.gpus > 1 else 'gpu',
        logger=False
        # You can add more Trainer arguments if needed
    )

    # Run testing
    pl.seed_everything(42, workers=True)
    trainer.test(model, datamodule=datamodule, verbose=True)


if __name__ == '__main__':
    main()
