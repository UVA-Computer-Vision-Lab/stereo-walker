import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import random
from PIL import Image

class TeleopDataset(Dataset):
    def __init__(self, cfg, mode):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.pose_dir = cfg.data.pose_dir
        self.image_root_dir = cfg.data.image_root_dir
        self.context_size = cfg.model.obs_encoder.context_size
        self.wp_length = cfg.model.decoder.len_traj_pred
        self.target_fps = cfg.data.target_fps
        self.num_workers = cfg.data.num_workers
        self.search_window = cfg.data.search_window
        self.arrived_threshold = cfg.data.arrived_threshold
        self.arrived_prob = cfg.data.arrived_prob

        pose_files = [
            "pose_traj_01.txt",
            "pose_traj_02.txt",
            "pose_traj_03.txt",
            "pose_traj_04.txt",
            "pose_traj_05.txt",
            "pose_traj_06.txt",
            "pose_traj_07.txt",
            "pose_traj_08.txt",
            "pose_traj_09.txt",
            "pose_traj_10.txt",
            "pose_traj_11.txt",
            "pose_traj_12.txt",
            "pose_traj_13.txt",
            "pose_traj_14.txt",
            "pose_traj_15.txt",
            "pose_traj_16.txt",
            "pose_traj_17.txt",
            "pose_traj_18.txt",
        ]
        self.pose_path = [os.path.join(self.pose_dir, f) for f in pose_files]

        if mode == 'train':
            self.pose_path = self.pose_path[:cfg.data.num_train]
        elif mode == 'val':
            self.pose_path = self.pose_path[cfg.data.num_train: cfg.data.num_train + cfg.data.num_val]
        elif mode == 'test':
            self.pose_path = self.pose_path[cfg.data.num_train + cfg.data.num_val: cfg.data.num_train + cfg.data.num_val + cfg.data.num_test]
        else:
            raise ValueError(f"Invalid mode {mode}")

        # Initialize storage
        self.gps_positions = []
        self.poses = []
        self.image_names = []
        self.count = []
        self.image_folders = []
        self.categories = []

        for f in tqdm(self.pose_path, desc="Loading data"):
            seq_idx = ''.join(filter(str.isdigit, os.path.basename(f)))
            image_folder = os.path.join(self.image_root_dir, f'traj_nav_{seq_idx}')
            if not os.path.exists(image_folder):
                raise FileNotFoundError(f"Image folder {image_folder} does not exist.")
            self.image_folders.append(image_folder)

            with open(f, 'r') as file:
                lines = file.readlines()

            gps_positions = []
            poses = []
            images = []
            categories = []

            # Reference GPS point for local coordinate conversion
            ref_lat, ref_lon, ref_alt = None, None, None

            for i in range(0, len(lines), 3):
                gps_line = lines[i].strip()
                pose_line = lines[i+1].strip()
                category_line = lines[i+2].strip()

                # Parse GPS data
                gps_tokens = gps_line.split(',')
                latitude = float(gps_tokens[1])
                longitude = float(gps_tokens[2])
                # accuracy = float(gps_tokens[3])
                altitude = float(gps_tokens[4])

                if ref_lat is None:
                    ref_lat = latitude
                    ref_lon = longitude
                    ref_alt = altitude

                # Convert GPS to local ENU coordinates
                x, y = self.latlon_to_local(latitude, longitude, ref_lat, ref_lon)
                z = altitude - ref_alt
                gps_position = np.array([x, y, z])
                gps_positions.append(gps_position)

                # Parse pose data
                pose_tokens = pose_line.split(',')
                tx = float(pose_tokens[1])
                ty = float(pose_tokens[2])
                tz = float(pose_tokens[3])
                rx = float(pose_tokens[4])
                ry = float(pose_tokens[5])
                rz = float(pose_tokens[6])
                image_name = f"forward_{int(pose_tokens[7]):04d}.jpg"
                pose = [tx, ty, tz, rx, ry, rz]
                poses.append(pose)
                images.append(image_name)

                # Parse category
                categories.append(category_line.split(','))

            gps_positions = np.array(gps_positions)
            poses = np.array(poses)
            if poses.shape[0] == 0 or gps_positions.shape[0] == 0:
                continue
            usable = poses.shape[0] - self.context_size - max(self.arrived_threshold*2, self.wp_length)
            categories = np.array(categories, dtype=np.int32)
            print(f"Sequence {seq_idx}: {usable} usable samples.")
            self.count.append(max(usable, 0))
            self.gps_positions.append(gps_positions)
            self.poses.append(poses)
            self.image_names.append(images)
            self.categories.append(categories)

        valid_indices = [i for i, c in enumerate(self.count) if c > 0]
        self.gps_positions = [self.gps_positions[i] for i in valid_indices]
        self.poses = [self.poses[i] for i in valid_indices]
        self.image_names = [self.image_names[i] for i in valid_indices]
        self.image_folders = [self.image_folders[i] for i in valid_indices]
        self.count = [self.count[i] for i in valid_indices]
        self.step_scale = []
        for pose in self.poses:
            step_scale = np.linalg.norm(np.diff(pose[:, [0, 1]], axis=0), axis=1).mean()
            self.step_scale.append(step_scale)

        self.lut = []
        self.sequence_ranges = []
        idx_counter = 0
        filtered_count = 0
        for seq_idx, count in enumerate(self.count):
            start_idx = idx_counter
            interval = self.context_size if self.mode == 'train' else 1
            # interval = 10
            for pose_start in range(0, count, interval):
                # Skip samples where the last category is 1 in training mode
                if self.mode == 'train':
                    category_idx = pose_start + self.context_size - 1
                    if self.categories[seq_idx][category_idx][-1] == 1:
                        filtered_count += 1
                        continue
                self.lut.append((seq_idx, pose_start))
                idx_counter += 1
            end_idx = idx_counter
            self.sequence_ranges.append((start_idx, end_idx))
        if self.mode == 'train' and filtered_count > 0:
            print(f"Filtered out {filtered_count} samples with last category = 1 in training mode.")
        assert len(self.lut) > 0, "No usable samples found."

    def __len__(self):
        return len(self.lut)

    def __getitem__(self, index):
        sequence_idx, pose_start = self.lut[index]
        gps_positions = self.gps_positions[sequence_idx]
        poses = self.poses[sequence_idx]
        images = self.image_names[sequence_idx]
        image_folder = self.image_folders[sequence_idx]

        # Get input GPS positions
        input_gps_positions = gps_positions[pose_start: pose_start + self.context_size]
        future_waypoints = poses[pose_start + self.context_size: pose_start + self.context_size + self.search_window]
        if future_waypoints.shape[0] == 0:
            raise IndexError(f"No future waypoints available for index {pose_start}.")

        # Select target waypoint
        target_idx, arrived = self.select_target_index(future_waypoints)

        # Get waypoints from poses
        # For input frames (history positions)
        input_poses = poses[pose_start: pose_start + self.context_size]
        # For future frames (gt waypoints)
        waypoint_start = pose_start + self.context_size
        waypoint_end = waypoint_start + self.wp_length
        gt_waypoint_poses = poses[waypoint_start: waypoint_end]

        # Transform waypoints to the coordinate frame of the current pose
        current_pose = input_poses[-1]
        history_positions = self.transform_poses(input_poses, current_pose)
        gt_waypoints = self.transform_poses(gt_waypoint_poses, current_pose)

        # Select target pose for visualization
        target_pose = poses[pose_start + self.context_size + target_idx]
        target_transformed = self.transform_pose(target_pose, current_pose)

        # Transform input GPS positions by subtracting target waypoint position
        if self.cfg.model.cord_embedding.type == 'polar':
            input_positions = self.input2target(input_gps_positions, target_transformed[:2])
            # Apply random rotation if in training mode
            if self.mode == 'train':
                rand_angle = np.random.uniform(-np.pi, np.pi)
                rot_matrix = np.array([[np.cos(rand_angle), -np.sin(rand_angle)],
                                       [np.sin(rand_angle), np.cos(rand_angle)]])
                input_positions = input_positions @ rot_matrix.T
        elif self.cfg.model.cord_embedding.type == 'input_target':
            input_positions = self.transform_input(input_gps_positions)
            input_positions = np.concatenate([input_positions, target_transformed[np.newaxis, :2]], axis=0)
        else:
            raise NotImplementedError(f"Coordinate embedding type {self.cfg.model.cord_embedding.type} not implemented")

        input_positions = torch.tensor(input_positions, dtype=torch.float32)
        arrived = torch.tensor(arrived, dtype=torch.float32)

        # Load frames
        input_image_names = images[pose_start: pose_start + self.context_size]
        frames = self.load_frames(image_folder, input_image_names)

        # Convert to tensors
        waypoints_transformed = torch.tensor(gt_waypoints[:, [0, 1]], dtype=torch.float32)
        step_scale = torch.tensor(self.step_scale[sequence_idx], dtype=torch.float32)
        step_scale = torch.clamp(step_scale, min=1e-2)
        waypoints_scaled = waypoints_transformed / step_scale
        input_positions_scaled = input_positions / step_scale

        sample = {
            'video_frames': frames,
            'input_positions': input_positions_scaled,
            'waypoints': waypoints_scaled,
            'arrived': arrived,
            'step_scale': step_scale
        }

        if self.mode in ['val', 'test']:
            # For visualization
            history_positions = torch.tensor(history_positions[:, [0, 1]], dtype=torch.float32)
            if self.cfg.model.cord_embedding.type == 'polar':
                target_transformed_position = torch.tensor(target_transformed[[0, 1]], dtype=torch.float32)
                sample['original_input_positions'] = history_positions
                sample['noisy_input_positions'] = history_positions
                sample['gt_waypoints'] = waypoints_transformed
                sample['target_transformed'] = target_transformed_position
            elif self.cfg.model.cord_embedding.type == 'input_target':
                sample['original_input_positions'] = history_positions
                sample['noisy_input_positions'] = input_positions[:-1, :]
                sample['gt_waypoints'] = waypoints_transformed
                sample['target_transformed'] = input_positions[-1, :]

        if self.mode == 'test':
            categories = self.categories[sequence_idx][pose_start + self.context_size - 1]
            # categories = self.categories[sequence_idx][pose_start]
            sample['categories'] = torch.tensor(categories, dtype=torch.float32)

        return sample

    def input2target(self, input_positions, target_position):
        transformed_input_positions = input_positions - target_position
        return transformed_input_positions
    
    def transform_input(self, input_positions):
        # Translate positions to current position
        current_position = input_positions[-1]
        translated_input = input_positions - current_position

        # Calculate angle to rotate second last input to negative y-axis
        second_last = translated_input[-2]
        angle = -np.pi / 2 - np.arctan2(second_last[1], second_last[0])

        # Create rotation matrix
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])

        # Apply rotation to input and target positions
        rotated_input = np.dot(translated_input[:, :2], rotation_matrix.T)

        return rotated_input

    def select_target_index(self, future_positions):
        arrived = np.random.rand() < self.arrived_prob
        max_idx = future_positions.shape[0] - 1
        if arrived:
            target_idx = random.randint(self.wp_length, min(self.wp_length + self.arrived_threshold, max_idx))
        else:
            target_idx = random.randint(self.wp_length + self.arrived_threshold, max_idx)
        return target_idx, arrived

    def transform_poses(self, poses, current_pose_array):
        current_pose_matrix = self.pose_to_matrix(current_pose_array)
        current_pose_inv = np.linalg.inv(current_pose_matrix)
        pose_matrices = self.poses_to_matrices(poses)
        transformed_matrices = np.matmul(current_pose_inv[np.newaxis, :, :], pose_matrices)
        positions = transformed_matrices[:, :3, 3]
        # Handel lidar extrinsic
        positions[:, [0, 1]] = positions[:, [1, 0]]
        positions[:, 1] *= -1
        return positions

    def transform_pose(self, pose, current_pose_array):
        current_pose_matrix = self.pose_to_matrix(current_pose_array)
        current_pose_inv = np.linalg.inv(current_pose_matrix)
        pose_matrix = self.pose_to_matrix(pose)
        transformed_matrix = np.matmul(current_pose_inv, pose_matrix)
        position = transformed_matrix[:3, 3]
        # Handel lidar extrinsic
        position[[0, 1]] = position[[1, 0]]
        position[1] *= -1
        return position

    def pose_to_matrix(self, pose):
        tx, ty, tz, rx, ry, rz = pose
        rotation = R.from_rotvec([rx, ry, rz])
        matrix = np.eye(4)
        matrix[:3, :3] = rotation.as_matrix()
        matrix[:3, 3] = [tx, ty, tz]
        return matrix

    def poses_to_matrices(self, poses):
        tx = poses[:, 0]
        ty = poses[:, 1]
        tz = poses[:, 2]
        rx = poses[:, 3]
        ry = poses[:, 4]
        rz = poses[:, 5]
        rotations = R.from_rotvec(np.stack([rx, ry, rz], axis=1))
        matrices = np.tile(np.eye(4), (poses.shape[0], 1, 1))
        matrices[:, :3, :3] = rotations.as_matrix()
        matrices[:, :3, 3] = np.stack([tx, ty, tz], axis=1)
        return matrices

    def load_frames(self, image_folder, image_names):
        frames = []
        for image_name in image_names:
            image_path = os.path.join(image_folder, image_name)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image {image_path} does not exist.")
            image = Image.open(image_path).convert('RGB')
            image = TF.to_tensor(image)
            frames.append(image)
        frames = torch.stack(frames)
        return frames

    def latlon_to_local(self, lat, lon, lat0, lon0):
        R_earth = 6378137  # Earth's radius in meters
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        lat0_rad = np.radians(lat0)
        lon0_rad = np.radians(lon0)
        dlat = lat_rad - lat0_rad
        dlon = lon_rad - lon0_rad
        x = dlon * np.cos((lat_rad + lat0_rad) / 2) * R_earth
        y = dlat * R_earth
        return x, y
