#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stereo Sequence Dataset - matches stereowalk_dataset.py format
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from scipy.spatial.transform import Rotation as R
from PIL import Image
from pathlib import Path
import random


class UVAData(Dataset):
    """
    Dataset for stereo sequence data compatible with stereowalk_dataset format
    """
    
    def __init__(self, cfg, mode, sequence_file='dataset/offline_filtered_v2/sequence_1s.json'):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.sequence_file = cfg.data.sequence_file
        self.context_size = cfg.model.obs_encoder.context_size
        self.wp_length = cfg.model.decoder.len_traj_pred
        self.target_fps = cfg.data.target_fps
        self.search_window = cfg.data.search_window
        self.arrived_threshold = cfg.data.arrived_threshold
        self.arrived_prob = cfg.data.arrived_prob
        self.input_noise = cfg.data.input_noise
        
        # Load sequence data from multiple subdirectories
        # Check if the sequence_file points to a directory structure with subdirectories
        base_path = Path(self.sequence_file).parent
        
        # Try to load from subdirectories (0, 1, 2, ... or 0_0, 0_1, ...)
        # Accept both pure digits and digit_digit format (for segmented datasets)
        subdirs = sorted([d for d in base_path.iterdir() 
                         if d.is_dir() and (d.name.isdigit() or 
                                           (d.name.replace('_', '').isdigit() and '_' in d.name))])
        
        # Store sequences separately for each subdirectory
        self.sequences = []  # List of sequences, one per subdirectory
        self.subdir_names = []  # Name of each subdirectory
        
        if subdirs:
            # Load from multiple subdirectories (keep them separate)
            print(f"Found {len(subdirs)} subdirectories in {base_path}")
            test_names = ['0_2', '0_3', '0_4', '1', '2_0', '3']
            mode_dirs = [d for d in subdirs if d.name in test_names]
            for subdir in mode_dirs:
                # Try to load sequence with categories first
                subdir_sequence_file = subdir / 'sequence_1s_with_categories.json'
                if not subdir_sequence_file.exists():
                    # Fall back to original file
                    subdir_sequence_file = subdir / 'sequence_1s.json'
                if subdir_sequence_file.exists():
                    print(f"Loading sequence from: {subdir_sequence_file}")
                    with open(subdir_sequence_file, 'r') as f:
                        data = json.load(f)
                    
                    # Store metadata from first file
                    if not hasattr(self, 'metadata'):
                        self.metadata = data['metadata']
                    
                    self.sequences.append(data['sequence'])
                    self.subdir_names.append(subdir.name)
                    print(f"  Loaded {len(data['sequence'])} samples from {subdir.name}")
            
            self.base_dir = base_path
            total_samples = sum(len(seq) for seq in self.sequences)
            print(f"Total samples from all subdirectories: {total_samples}")
        else:
            # Fallback to single file loading (original behavior)
            print(f"Loading sequence from: {sequence_file}")
            with open(sequence_file, 'r') as f:
                data = json.load(f)
            
            self.metadata = data['metadata']
            self.sequences = [data['sequence']]
            self.subdir_names = ['']
            self.base_dir = Path(sequence_file).parent
            
            print(f"Total samples in sequence: {len(self.sequences[0])}")
        
        print(f"Base directory: {self.base_dir}")
        
        # Convert angle unit if needed
        self.angle_scale = 1.0
        if self.metadata['angle_unit'] == 'degrees':
            self.angle_scale = np.pi / 180.0  # Convert degrees to radians
        
        # Extract poses from each sequence separately and convert to quaternion format
        # Match stereowalk format: [x, y, z, qx, qy, qz, qw]
        self.poses_list = []  # List of pose arrays, one per subdirectory
        self.image_names_list = []  # List of image name lists, one per subdirectory
        
        # Also extract categories if available
        self.categories_list = []  # List of category arrays, one per subdirectory
        
        for seq_idx, sequence in enumerate(self.sequences):
            poses = []
            image_names = []
            categories = []
            
            for item in sequence:
                pose = item['pose']
                # Convert Euler angles to quaternion
                rx_rad = pose['rx'] * self.angle_scale
                ry_rad = pose['ry'] * self.angle_scale
                rz_rad = pose['rz'] * self.angle_scale
                
                rotation = R.from_rotvec([rx_rad, ry_rad, rz_rad])
                quat = rotation.as_quat()  # Returns [qx, qy, qz, qw]
                
                # Create pose array: [x, y, z, qx, qy, qz, qw]
                pose_array = np.array([
                    pose['tx'],
                    pose['ty'],
                    pose['tz'],
                    quat[0],  # qx
                    quat[1],  # qy
                    quat[2],  # qz
                    quat[3]   # qw
                ])
                poses.append(pose_array)
                
                # Store left and right image names with subdirectory info
                image_names.append({
                    'left': item['left_image'],
                    'right': item['right_image'],
                    'subdir': self.subdir_names[seq_idx]
                })
                
                # Extract categories if available
                if 'categories' in item:
                    categories.append(item['categories'])
                else:
                    # Default: all zeros except 'other' category
                    categories.append([0, 0, 0, 0, 0, 1])
            
            self.poses_list.append(np.array(poses))
            self.image_names_list.append(image_names)
            self.categories_list.append(np.array(categories, dtype=np.int32))
        
        # Calculate step scale - use [x, y] plane coordinates (ground robot)
        # Calculate from all sequences combined
        all_step_distances = []
        for poses in self.poses_list:
            step_distances = np.linalg.norm(np.diff(poses[:, [0, 1]], axis=0), axis=1)
            all_step_distances.extend(step_distances)
        self.step_scale = np.mean(all_step_distances)
        print(f"Average step scale: {self.step_scale:.4f}m (using XY plane)")
        
        # Build lookup table - keep sequences separate
        # LUT contains tuples of (sequence_index, pose_start_in_sequence)
        self.lut = []
        interval = self.context_size if self.mode == 'train' else 1
        
        # Check if we should skip 'others' category in training
        skip_others = getattr(self.cfg.data, 'skip_others_in_train', False) and self.mode == 'train'
        
        for seq_idx, poses in enumerate(self.poses_list):
            total_samples = len(poses)
            usable = total_samples - self.context_size - max(self.arrived_threshold*2, self.wp_length)
            
            skipped_count = 0
            added_count = 0
            
            for pose_start in range(0, usable, interval):
                # If skipping others, check the category at the current pose
                if skip_others:
                    # Get the category at pose_start + context_size - 1 (the last input frame)
                    category_idx = pose_start + self.context_size - 1
                    if (category_idx < len(self.categories_list[seq_idx]) and 
                        len(self.categories_list[seq_idx]) > 0):
                        categories = self.categories_list[seq_idx][category_idx]
                        # categories[5] is 'other' category, skip if it's 1
                        if len(categories) > 5 and categories[5] == 1:
                            skipped_count += 1
                            continue
                
                self.lut.append((seq_idx, pose_start))
                added_count += 1
            
            if skip_others:
                print(f"Sequence {self.subdir_names[seq_idx]}: {total_samples} samples, {usable} usable, {added_count} added, {skipped_count} skipped (others)")
            else:
                print(f"Sequence {self.subdir_names[seq_idx]}: {total_samples} samples, {usable} usable")
        
        print(f"Dataset mode: {mode}")
        print(f"Total samples in LUT: {len(self.lut)}")
    
    def __len__(self):
        return len(self.lut)
    
    def __getitem__(self, index):
        seq_idx, pose_start = self.lut[index]
        pose = self.poses_list[seq_idx]
        image_names = self.image_names_list[seq_idx]
        
        # Get input and future poses
        input_poses, future_poses = self.get_input_and_future_poses(pose, pose_start)
        original_input_poses = np.copy(input_poses)
        
        # Select target pose
        target_pose, arrived = self.select_target_pose(future_poses)
        
        # Extract waypoints
        waypoint_poses = self.extract_waypoints(pose, pose_start)
        
        # Transform poses
        current_pose = input_poses[-1]
        
        if self.cfg.model.cord_embedding.type == 'polar':
            transformed_input_positions = self.input2target(input_poses, target_pose)
        elif self.cfg.model.cord_embedding.type == 'input_target':
            transformed_input_positions = np.concatenate([
                self.transform_poses(input_poses, current_pose)[:, [0, 1]], 
                self.transform_target_pose(target_pose, current_pose)[np.newaxis, [0, 1]]
            ], axis=0)
        else:
            raise NotImplementedError(f"Coordinate embedding type {self.cfg.model.cord_embedding.type} not implemented")
        
        waypoints_transformed = self.transform_waypoints(waypoint_poses, current_pose)
        
        # Load stereo frames
        input_image_names = image_names[pose_start: pose_start + self.context_size]
        if self.cfg.data.use_stereo:
            input_frames = self.load_stereo_frames(input_image_names)
        else:
            input_frames = self.load_mono_frames(input_image_names)
        
        # Convert data to tensors
        input_positions = torch.tensor(transformed_input_positions, dtype=torch.float32)
        waypoints_transformed = torch.tensor(waypoints_transformed[:, [0, 1]], dtype=torch.float32)
        step_scale = torch.tensor(self.step_scale, dtype=torch.float32)
        step_scale = torch.clamp(step_scale, min=1e-2)
        input_positions_scaled = input_positions / step_scale
        waypoints_scaled = waypoints_transformed / step_scale
        
        # Add noise to input positions (except last one)
        input_positions_scaled[:self.context_size-1] += torch.randn(self.context_size-1, 2) * self.input_noise
        
        arrived = torch.tensor(arrived, dtype=torch.float32)
        
        sample = {
            'video_frames': input_frames,
            'input_positions': input_positions_scaled,
            'waypoints': waypoints_scaled,
            'arrived': arrived,
            'step_scale': step_scale
        }
        
        # For visualization during validation
        if self.mode in ['val', 'test']:
            transformed_original_input_positions = self.transform_poses(original_input_poses, current_pose)
            target_transformed = self.transform_target_pose(target_pose, current_pose)
            
            original_input_positions = torch.tensor(transformed_original_input_positions[:, [0, 1]], dtype=torch.float32)
            noisy_input_positions = input_positions_scaled[:-1] * step_scale
            target_transformed_position = torch.tensor(target_transformed[[0, 1]], dtype=torch.float32)
            sample['original_input_positions'] = original_input_positions
            sample['noisy_input_positions'] = noisy_input_positions
            sample['gt_waypoints'] = waypoints_transformed
            sample['target_transformed'] = target_transformed_position
            
            # Add categories for test mode
            if self.mode == 'test':
                categories = self.categories_list[seq_idx][pose_start + self.context_size - 1]
                sample['categories'] = torch.tensor(categories, dtype=torch.float32)
        
        return sample
    
    def get_input_and_future_poses(self, pose, pose_start):
        """Get input and future poses"""
        input_poses = pose[pose_start: pose_start + self.context_size]
        search_end = min(pose_start + self.context_size + self.search_window, pose.shape[0])
        future_poses = pose[pose_start + self.context_size: search_end]
        if future_poses.shape[0] == 0:
            raise IndexError(f"No future poses available for index {pose_start}.")
        return input_poses, future_poses
    
    def input2target(self, input_poses, target_pose):
        """Transform input positions relative to target - using XY plane"""
        input_positions = input_poses[:, :3]
        target_position = target_pose[:3]
        transformed_input_positions = (input_positions - target_position)[:, [0, 1]]
        
        if self.mode == 'train':
            rand_angle = np.random.uniform(-np.pi, np.pi)
            rot_matrix = np.array([[np.cos(rand_angle), -np.sin(rand_angle)], 
                                  [np.sin(rand_angle), np.cos(rand_angle)]])
            transformed_input_positions = transformed_input_positions @ rot_matrix.T
        
        return transformed_input_positions
    
    def select_target_pose(self, future_poses):
        """Select target waypoint"""
        arrived = np.random.rand() < self.arrived_prob
        if arrived:
            target_idx = random.randint(self.wp_length, self.wp_length + self.arrived_threshold)
        else:
            target_idx = random.randint(self.wp_length + self.arrived_threshold, future_poses.shape[0] - 1)
        target_pose = future_poses[target_idx]
        return target_pose, arrived
    
    def extract_waypoints(self, pose, pose_start):
        """Extract waypoint poses"""
        waypoint_start = pose_start + self.context_size
        waypoint_end = waypoint_start + self.wp_length
        waypoint_poses = pose[waypoint_start: waypoint_end]
        return waypoint_poses
    
    def transform_poses(self, poses, current_pose_array):
        """Transform poses to current pose frame"""
        current_pose_matrix = self.pose_to_matrix(current_pose_array)
        current_pose_inv = np.linalg.inv(current_pose_matrix)
        pose_matrices = self.poses_to_matrices(poses)
        transformed_matrices = np.matmul(current_pose_inv[np.newaxis, :, :], pose_matrices)
        positions = transformed_matrices[:, :3, 3]
        positions[:, [0, 1]] = positions[:, [1, 0]]
        positions[:, 0] *= -1
        return positions
    
    def transform_waypoints(self, waypoint_poses, current_pose_array):
        """Transform waypoints to current pose frame"""
        current_pose_matrix = self.pose_to_matrix(current_pose_array)
        current_pose_inv = np.linalg.inv(current_pose_matrix)
        waypoint_matrices = self.poses_to_matrices(waypoint_poses)
        transformed_waypoint_matrices = np.matmul(current_pose_inv[np.newaxis, :, :], waypoint_matrices)
        waypoints_positions = transformed_waypoint_matrices[:, :3, 3]
        waypoints_positions[:, [0, 1]] = waypoints_positions[:, [1, 0]]
        waypoints_positions[:, 0] *= -1
        return waypoints_positions
    
    def transform_target_pose(self, target_pose, current_pose_array):
        """Transform target pose to current pose frame"""
        current_pose_matrix = self.pose_to_matrix(current_pose_array)
        current_pose_inv = np.linalg.inv(current_pose_matrix)
        target_pose_matrix = self.pose_to_matrix(target_pose)
        transformed_target_matrix = np.matmul(current_pose_inv, target_pose_matrix)
        target_position = transformed_target_matrix[:3, 3]
        target_position[[0, 1]] = target_position[[1, 0]]
        target_position[0] *= -1
        return target_position
    
    def pose_to_matrix(self, pose):
        """Convert pose [x, y, z, qx, qy, qz, qw] to 4x4 transformation matrix"""
        position = pose[:3]
        rotation = R.from_quat(pose[3:])
        matrix = np.eye(4)
        matrix[:3, :3] = rotation.as_matrix()
        matrix[:3, 3] = position
        return matrix
    
    def poses_to_matrices(self, poses):
        """Convert multiple poses to transformation matrices"""
        positions = poses[:, :3]
        quats = poses[:, 3:]
        rotations = R.from_quat(quats)
        matrices = np.tile(np.eye(4), (poses.shape[0], 1, 1))
        matrices[:, :3, :3] = rotations.as_matrix()
        matrices[:, :3, 3] = positions
        return matrices
    
    def load_stereo_frames(self, image_names):
        """
        Load stereo image frames in the format of stereowalk_dataset
        Returns: (T, 2, C, H, W) where T=time steps, 2=left/right eyes
        """
        left_frames = []
        right_frames = []
        
        for img_name in image_names:
            # Build path with subdirectory
            if img_name['subdir']:
                subdir_path = self.base_dir / img_name['subdir']
                left_path = subdir_path / self.metadata['left_dir'] / img_name['left']
                right_path = subdir_path / self.metadata['right_dir'] / img_name['right']
            else:
                # Fallback for single directory structure
                left_path = self.base_dir / self.metadata['left_dir'] / img_name['left']
                right_path = self.base_dir / self.metadata['right_dir'] / img_name['right']
            
            # Load left image
            left_img = Image.open(left_path).convert('RGB')
            left_tensor = TF.to_tensor(left_img)
            left_frames.append(left_tensor)
            
            # Load right image
            right_img = Image.open(right_path).convert('RGB')
            right_tensor = TF.to_tensor(right_img)
            right_frames.append(right_tensor)
        
        # Stack to (T, C, H, W)
        left_frames = torch.stack(left_frames)
        right_frames = torch.stack(right_frames)
        
        # Combine to (T, 2, C, H, W) - same as stereowalk_dataset
        stereo_frames = torch.stack([left_frames, right_frames], dim=1)
        
        return stereo_frames
    
    def load_mono_frames(self, image_names):
        """
        Load mono image frames in the format of stereowalk_dataset
        Returns: (T, C, H, W) where T=time steps
        """
        frames = []
        for img_name in image_names:
            # Build path with subdirectory
            if img_name['subdir']:
                subdir_path = self.base_dir / img_name['subdir']
                image_path = subdir_path / self.metadata['left_dir'] / img_name['left']
            else:
                # Fallback for single directory structure
                image_path = self.base_dir / self.metadata['left_dir'] / img_name['left']
            
            image = Image.open(image_path).convert('RGB')
            image = TF.to_tensor(image)
            frames.append(image)
        
        frames = torch.stack(frames)
        return frames