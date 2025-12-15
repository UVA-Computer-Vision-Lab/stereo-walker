import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
from model.stereowalker import StereoWalker
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
plt.style.use('seaborn-v0_8')
import os

class StereoWalkerModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = StereoWalker(cfg)
        self.save_hyperparameters(cfg)
        self.do_normalize = cfg.training.normalize_step_length
        self.datatype = cfg.data.type
        
        # Coordinate representation
        self.output_coordinate_repr = cfg.model.output_coordinate_repr
        if self.output_coordinate_repr not in ["euclidean"]:
            raise ValueError(f"Unsupported coordinate representation: {self.output_coordinate_repr}")
        
        self.decoder = cfg.model.decoder.type
        if self.decoder not in ["attention"]:
            raise ValueError(f"Unsupported decoder: {self.decoder}")
        
        # Direction loss weight (you can adjust this value in your cfg)
        self.direction_loss_weight = cfg.training.direction_loss_weight
        self.feature_loss_weight = cfg.training.feature_loss_weight
        self.arrived_loss_weight = cfg.training.arrived_loss_weight
        
        # Visualization settings
        self.val_num_visualize = cfg.validation.num_visualize
        self.test_num_visualize = cfg.testing.num_visualize
        self.vis_count = 0
        
        self.result_dir = cfg.project.result_dir
        self.batch_size = cfg.training.batch_size
        self.image_mean = np.array([0.485, 0.456, 0.406])
        self.image_std = np.array([0.229, 0.224, 0.225])

        if self.datatype in ["teleop", "stereowalk_labeled", "uva"]:
            self.test_catetories = ['crowd', 'person_close_by', 'turn', 'action_target_mismatch', 'crossing', 'other']
            self.num_categories = len(self.test_catetories)

    def forward(self, obs, cord, future_obs):
        return self.model(obs, cord, future_obs)
    
    def training_step(self, batch, batch_idx):
        obs = batch['video_frames']
        cord = batch['input_positions']
        if "future_video_frames" in batch:
            future_obs = batch['future_video_frames']
        else:
            future_obs = None
        
        wp_pred, arrive_pred, feature_pred, feature_gt = self(obs, cord, future_obs)
        losses = self.compute_loss(wp_pred, arrive_pred, feature_pred, feature_gt, batch)
        waypoints_loss = losses['waypoints_loss']
        arrived_loss = losses['arrived_loss']
        direction_loss = losses['direction_loss']
        feature_loss = losses['feature_loss']
        total_loss = waypoints_loss + self.arrived_loss_weight * arrived_loss + self.direction_loss_weight * direction_loss + feature_loss * self.feature_loss_weight

        # Common logs
        self.log('train/l_wp', waypoints_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train/l_arvd', arrived_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train/l_dir', direction_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train/l_feat', feature_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        obs = batch['video_frames']
        cord = batch['input_positions']
        if "future_video_frames" in batch:
            future_obs = batch['future_video_frames']
        else:
            future_obs = None
        wp_pred, arrive_pred, feature_pred, feature_gt = self(obs, cord, future_obs)
        losses = self.compute_loss(wp_pred, arrive_pred, feature_pred, feature_gt, batch)
        l1_loss = losses['waypoints_loss']
        direction_loss = losses['direction_loss']
        feature_loss = losses['feature_loss']
        self.log('val/l1_loss', l1_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Compute accuracy for "arrived" prediction
        arrived_target = batch['arrived']
        arrived_logits = arrive_pred.flatten()
        arrived_probs = torch.sigmoid(arrived_logits)
        arrived_pred_binary = (arrived_probs >= 0.5).float()
        correct = (arrived_pred_binary == arrived_target).float()
        accuracy = correct.sum() / correct.numel()
        
        # Log the metrics
        self.log('val/arrived_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/direction_loss', direction_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val/feature_loss', feature_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        
        # Handle visualization
        wp_pred_vis = wp_pred * batch['step_scale'].unsqueeze(-1).unsqueeze(-1)
        self.process_visualization(
            mode='val',
            batch=batch,
            obs=obs,
            wp_pred=wp_pred_vis,
            arrive_pred=arrive_pred,
            # categories=None
        )
        
        return direction_loss

    def test_step(self, batch, batch_idx):
        obs = batch['video_frames']
        cord = batch['input_positions']
        if "future_video_frames" in batch:
            future_obs = batch['future_video_frames']
        else:
            future_obs = None
        
        # Handle different input shapes
        if len(obs.shape) == 6:  # Stereo input: (B, T, 2, C, H, W)
            B, T = obs.shape[0], obs.shape[1]
        else:  # Mono input: (B, T, C, H, W)
            B, T = obs.shape[0], obs.shape[1]
        
        if self.datatype == "citywalk":
            wp_pred, arrive_pred, _, _ = self(obs, cord, future_obs)
            # Compute L1 loss for waypoints
            waypoints_target = batch['waypoints']
            l1_loss = F.l1_loss(wp_pred, waypoints_target, reduction='mean').item()
            
            # Compute accuracy for "arrived" prediction
            arrived_target = batch['arrived']
            arrived_logits = arrive_pred.flatten()
            arrived_probs = torch.sigmoid(arrived_logits)
            arrived_pred_binary = (arrived_probs >= 0.5).float()
            correct = (arrived_pred_binary == arrived_target).float()
            accuracy = correct.sum().item() / correct.numel()

            # wp_pred_last = wp_pred[:, -1, :]  # shape [batch_size, 2]
            # waypoints_target_last = waypoints_target[:, -1, :]  # shape [batch_size, 2]

            # Compute cosine similarity
            wp_pred_view = wp_pred.view(-1, 2)
            waypoints_target_view = waypoints_target.view(-1, 2)
            # dot_product = (wp_pred_view * waypoints_target_view).sum(dim=1)  # shape [batch_size]
            # norm_pred = wp_pred_view.norm(dim=1)  # shape [batch_size]
            # norm_target = waypoints_target_view.norm(dim=1)  # shape [batch_size]
            # cos_sim = dot_product / (norm_pred * norm_target + 1e-8)  # avoid division by zero
            cos_sim = F.cosine_similarity(wp_pred_view, waypoints_target_view, dim=1)
            
            # Compute angle in degrees
            angle = torch.acos(cos_sim) * 180 / torch.pi  # shape [batch_size]
            angle = angle.view(B, T)
            
            # Take mean angle
            mean_angle = angle.mean(dim=0).cpu().numpy()
            
            # Store the metrics
            if self.output_coordinate_repr == "euclidean":
                self.test_metrics['l1_loss'].append(l1_loss)
            self.test_metrics['arrived_accuracy'].append(accuracy)
            self.test_metrics['mean_angle'].append(mean_angle)
        elif self.datatype == "uva":
            # Check if we have categories
            has_categories = 'categories' in batch and hasattr(self, 'test_catetories')
            
            if has_categories:
                # UVA with categories (similar to teleop)
                category = batch['categories']
            
            wp_pred, arrive_pred, _, _ = self(obs, cord, future_obs)
            wp_pred *= batch['step_scale'].unsqueeze(-1).unsqueeze(-1)
            
            # Compute L1 loss for waypoints
            waypoints_target = batch['waypoints']
            waypoints_target *= batch['step_scale'].unsqueeze(-1).unsqueeze(-1)
            l1_loss = F.mse_loss(wp_pred, waypoints_target, reduction='none') ** 0.5
            
            # Compute accuracy for "arrived" prediction
            arrived_target = batch['arrived']
            arrived_probs = torch.sigmoid(arrive_pred)
            arrived_pred_binary = (arrived_probs >= 0.5).float().squeeze(-1)
            correct = (arrived_pred_binary == arrived_target).float()

            # Compute cosine similarity
            wp_pred_view = wp_pred.view(-1, 2)
            waypoints_target_view = waypoints_target.view(-1, 2)
            cos_sim = F.cosine_similarity(wp_pred_view, waypoints_target_view, dim=1)
            # Compute angle in degrees
            angle = torch.acos(cos_sim) * 180 / torch.pi
            angle = angle.view(B, -1)

            gt_wp_last_norm = waypoints_target[:, -1, :].norm(dim=1)

            if has_categories:
                # Record metrics by category
                for batch_idx in range(B):
                    for category_idx in range(self.num_categories):
                        if category[batch_idx, category_idx] == 1:
                            category_name = self.test_catetories[category_idx]
                            self.test_metrics[category_name]['l1_loss'].append(l1_loss[batch_idx].max().item())
                            self.test_metrics[category_name]['arrived_accuracy'].append(correct[batch_idx].item())
                            if gt_wp_last_norm[batch_idx] > 1:
                                self.test_metrics[category_name]['mean_angle'].append(angle[batch_idx].max().item())
                                self.test_metrics[category_name]['angle_step1'].append(angle[batch_idx, 0].item())
                                self.test_metrics[category_name]['angle_step2'].append(angle[batch_idx, 1].item())
                                self.test_metrics[category_name]['angle_step3'].append(angle[batch_idx, 2].item())
                                self.test_metrics[category_name]['angle_step4'].append(angle[batch_idx, 3].item())
                                self.test_metrics[category_name]['angle_step5'].append(angle[batch_idx, 4].item())
                        else:
                            continue
                    self.test_metrics['overall']['l1_loss'].append(l1_loss[batch_idx].max().item())
                    self.test_metrics['overall']['arrived_accuracy'].append(correct[batch_idx].item())
                    if gt_wp_last_norm[batch_idx] > 1:
                        self.test_metrics['overall']['mean_angle'].append(angle[batch_idx].max().item())
                        self.test_metrics['overall']['angle_step1'].append(angle[batch_idx, 0].item())
                        self.test_metrics['overall']['angle_step2'].append(angle[batch_idx, 1].item())
                        self.test_metrics['overall']['angle_step3'].append(angle[batch_idx, 2].item())
                        self.test_metrics['overall']['angle_step4'].append(angle[batch_idx, 3].item())
                        self.test_metrics['overall']['angle_step5'].append(angle[batch_idx, 4].item())
            else:
                # UVA without categories (original behavior)
                for batch_idx in range(B):
                    self.test_metrics['l1_loss'].append(l1_loss[batch_idx].max().item())
                    self.test_metrics['arrived_accuracy'].append(correct[batch_idx].item())
                    if gt_wp_last_norm[batch_idx] > 1:
                        self.test_metrics['mean_angle'].append(angle[batch_idx].max().item())
            
        elif self.datatype in ["teleop", "stereowalk_labeled"]:
            category = batch['categories']
            wp_pred, arrive_pred, _, _ = self(obs, cord, future_obs)
            wp_pred *= batch['step_scale'].unsqueeze(-1).unsqueeze(-1)
            
            # Compute L1 loss for waypoints
            waypoints_target = batch['waypoints']
            waypoints_target *= batch['step_scale'].unsqueeze(-1).unsqueeze(-1)
            # l1_loss = F.l1_loss(wp_pred, waypoints_target, reduction='none')
            l1_loss = F.mse_loss(wp_pred, waypoints_target, reduction='none') ** 0.5
            
            # Compute accuracy for "arrived" prediction
            arrived_target = batch['arrived']
            arrived_probs = torch.sigmoid(arrive_pred)
            arrived_pred_binary = (arrived_probs >= 0.5).float().squeeze(-1)
            correct = (arrived_pred_binary == arrived_target).float()

            # Compute cosine similarity
            wp_pred_view = wp_pred.view(-1, 2)
            waypoints_target_view = waypoints_target.view(-1, 2)
            # dot_product = (wp_pred_view * waypoints_target_view).sum(dim=1)  # shape [batch_size]
            # norm_pred = wp_pred_view.norm(dim=1)  # shape [batch_size]
            # norm_target = waypoints_target_view.norm(dim=1)  # shape [batch_size]
            # cos_sim = dot_product / (norm_pred * norm_target + 1e-8)  # avoid division by zero
            cos_sim = F.cosine_similarity(wp_pred_view, waypoints_target_view, dim=1)
            # Compute angle in degrees
            angle = torch.acos(cos_sim) * 180 / torch.pi  # shape [batch_size]
            angle = angle.view(B, T)

            gt_wp_last_norm = waypoints_target[:, -1, :].norm(dim=1)

            for batch_idx in range(B):
                for category_idx in range(self.num_categories):
                    if category[batch_idx, category_idx] == 1:
                        category_name = self.test_catetories[category_idx]
                        self.test_metrics[category_name]['l1_loss'].append(l1_loss[batch_idx].max().item())
                        self.test_metrics[category_name]['arrived_accuracy'].append(correct[batch_idx].item())
                        if gt_wp_last_norm[batch_idx] > 1:
                            self.test_metrics[category_name]['mean_angle'].append(angle[batch_idx].max().item())
                            self.test_metrics[category_name]['angle_step1'].append(angle[batch_idx, 0].item())
                            self.test_metrics[category_name]['angle_step2'].append(angle[batch_idx, 1].item())
                            self.test_metrics[category_name]['angle_step3'].append(angle[batch_idx, 2].item())
                            self.test_metrics[category_name]['angle_step4'].append(angle[batch_idx, 3].item())
                            self.test_metrics[category_name]['angle_step5'].append(angle[batch_idx, 4].item())
                    else:
                        continue
                self.test_metrics['overall']['l1_loss'].append(l1_loss[batch_idx].max().item())
                self.test_metrics['overall']['arrived_accuracy'].append(correct[batch_idx].item())
                if gt_wp_last_norm[batch_idx] > 1:
                    self.test_metrics['overall']['mean_angle'].append(angle[batch_idx].max().item())
                    self.test_metrics['overall']['angle_step1'].append(angle[batch_idx, 0].item())
                    self.test_metrics['overall']['angle_step2'].append(angle[batch_idx, 1].item())
                    self.test_metrics['overall']['angle_step3'].append(angle[batch_idx, 2].item())
                    self.test_metrics['overall']['angle_step4'].append(angle[batch_idx, 3].item())
                    self.test_metrics['overall']['angle_step5'].append(angle[batch_idx, 4].item())

        
        # Handle visualization
        if self.datatype == "citywalk":
            wp_pred *= batch['step_scale'].unsqueeze(-1).unsqueeze(-1)
        if self.output_coordinate_repr == "euclidean":
            self.process_visualization(
                mode='test',
                batch=batch,
                obs=obs,
                wp_pred=wp_pred,
                arrive_pred=arrive_pred
            )
        elif self.output_coordinate_repr == "polar":
            self.process_visualization(
                mode='test',
                batch=batch,
                obs=obs,
                wp_pred=wp_pred,
                arrive_pred=arrive_pred
            )

    def on_test_epoch_end(self):
        if self.datatype == "citywalk":
            import pandas as pd
            self.test_metrics['count'] = len(self.test_metrics['l1_loss'])
            for metric in self.test_metrics:
                if metric != 'count':
                    self.test_metrics[metric] = np.nanmean(np.array(self.test_metrics[metric]))

            df = pd.DataFrame([self.test_metrics])  
            save_path = os.path.join(self.result_dir, 'test_metrics.csv')
            df.to_csv(save_path, index=False)
        
        elif self.datatype == "uva":
            import pandas as pd
            # Check if we have categories
            has_categories = hasattr(self, 'test_catetories') and 'overall' in self.test_metrics
            
            if has_categories:
                # UVA with categories (similar to teleop)
                for category in self.test_catetories:
                    # Add a new 'count' metric for each category by counting 'l1_loss' entries
                    self.test_metrics[category]['count'] = len(self.test_metrics[category]['l1_loss'])
                self.test_metrics['overall']['count'] = sum(self.test_metrics[category]['count'] for category in self.test_catetories)
                self.test_metrics['mean']['count'] = 0

                for category in self.test_catetories:
                    for metric in self.test_metrics[category]:
                        if metric != 'count':
                            self.test_metrics[category][metric] = np.nanmean(np.array(self.test_metrics[category][metric]))
                for metric in self.test_metrics['overall']:
                    if metric != 'count':
                        self.test_metrics['overall'][metric] = np.nanmean(np.array(self.test_metrics['overall'][metric]))
                metrics = ['l1_loss', 'arrived_accuracy', 'angle_step1', 'angle_step2', 'angle_step3', 'angle_step4', 'angle_step5', 'mean_angle']
                for metric in metrics:
                    category_val = []
                    for category in self.test_catetories:
                        category_val.append(self.test_metrics[category][metric])
                    self.test_metrics['mean'][metric] = np.array(category_val).mean()
                    print(f"{metric}: Sample mean {self.test_metrics['overall'][metric]:.4f}, Category mean {self.test_metrics['mean'][metric]:.4f}")

                df = pd.DataFrame(self.test_metrics)
                df = df.reset_index().rename(columns={'index': 'Metrics'})
                save_path = os.path.join(self.result_dir, 'test_metrics.csv')
                df.to_csv(save_path, index=False)
            else:
                # UVA without categories (original behavior)
                self.test_metrics['count'] = len(self.test_metrics['l1_loss'])
                for metric in self.test_metrics:
                    if metric != 'count':
                        self.test_metrics[metric] = np.nanmean(np.array(self.test_metrics[metric]))

                df = pd.DataFrame([self.test_metrics])  
                save_path = os.path.join(self.result_dir, 'test_metrics.csv')
                df.to_csv(save_path, index=False)

        elif self.datatype in ["teleop", "stereowalk_labeled"]:
            import pandas as pd
            for category in self.test_catetories:
                # Add a new 'count' metric for each category by counting 'l1_loss' entries
                self.test_metrics[category]['count'] = len(self.test_metrics[category]['l1_loss'])
            self.test_metrics['overall']['count'] = sum(self.test_metrics[category]['count'] for category in self.test_catetories)
            self.test_metrics['mean']['count'] = 0

            for category in self.test_catetories:
                for metric in self.test_metrics[category]:
                    if metric != 'count':
                        # print(f"{category} {metric}: {self.test_metrics[category][metric]}")
                        self.test_metrics[category][metric] = np.nanmean(np.array(self.test_metrics[category][metric]))
            for metric in self.test_metrics['overall']:
                if metric != 'count':
                    self.test_metrics['overall'][metric] = np.nanmean(np.array(self.test_metrics['overall'][metric]))
            metrics = ['l1_loss', 'arrived_accuracy', 'angle_step1', 'angle_step2', 'angle_step3', 'angle_step4', 'angle_step5', 'mean_angle']
            for metric in metrics:
                category_val = []
                for category in self.test_catetories:
                    category_val.append(self.test_metrics[category][metric])
                self.test_metrics['mean'][metric] = np.array(category_val).mean()
                print(f"{metric}: Sample mean {self.test_metrics['overall'][metric]:.4f}, Category mean {self.test_metrics['mean'][metric]:.4f}")

            df = pd.DataFrame(self.test_metrics)
            df = df.reset_index().rename(columns={'index': 'Metrics'})
            save_path = os.path.join(self.result_dir, 'test_metrics.csv')
            df.to_csv(save_path, index=False)


    def on_validation_epoch_start(self):
        self.vis_count = 0

    def on_test_epoch_start(self):
        self.vis_count = 0
        if self.datatype == "citywalk":
            if self.output_coordinate_repr == "euclidean":
                self.test_metrics = {'l1_loss': [], 'arrived_accuracy': [], 'mean_angle': []}
            elif self.output_coordinate_repr == "polar":
                self.test_metrics = {'distance_loss': [], 'angle_loss': [], 'arrived_accuracy': [], 'mean_angle': []}
        elif self.datatype == "uva":
            # Check if we have categories support
            categories = self.test_catetories[:] if hasattr(self, 'test_catetories') else []
            if categories:
                # UVA with categories
                self.test_metrics = {}  # 初始化字典
                categories.extend(['mean', 'overall'])
                for category in categories:
                    if self.output_coordinate_repr == "euclidean":
                        self.test_metrics[category] = {
                            'l1_loss': [], 
                            'arrived_accuracy': [], 
                            'angle_step1': [],
                            'angle_step2': [],
                            'angle_step3': [],
                            'angle_step4': [],
                            'angle_step5': [],
                            'mean_angle': []
                        }
                    elif self.output_coordinate_repr == "polar":
                        raise ValueError("Polar representation is not supported for uva dataset with categories.")
            else:
                # UVA without categories (original behavior)
                if self.output_coordinate_repr == "euclidean":
                    self.test_metrics = {'l1_loss': [], 'arrived_accuracy': [], 'mean_angle': []}
                elif self.output_coordinate_repr == "polar":
                    self.test_metrics = {'distance_loss': [], 'angle_loss': [], 'arrived_accuracy': [], 'mean_angle': []}
        elif self.datatype in ["teleop", "stereowalk_labeled"]:
            self.test_metrics = {}
            categories = self.test_catetories[:]
            categories.extend(['mean', 'overall'])
            for category in categories:
                if self.output_coordinate_repr == "euclidean":
                    self.test_metrics[category] = {
                        'l1_loss': [], 
                        'arrived_accuracy': [], 
                        'angle_step1': [],
                        'angle_step2': [],
                        'angle_step3': [],
                        'angle_step4': [],
                        'angle_step5': [],
                        'mean_angle': []
                    }
                elif self.output_coordinate_repr == "polar":
                    raise ValueError(f"Polar representation is not supported for {self.datatype} dataset.")

    def compute_loss(self, wp_pred, arrive_pred, feature_pred, feature_gt, batch):
        waypoints_target = batch['waypoints']
        arrived_target = batch['arrived']
        if feature_pred is not None and feature_gt is not None:
            feature_loss = F.mse_loss(feature_pred, feature_gt)
        else:
            feature_loss = 0.0
        wp_loss = F.mse_loss(wp_pred, waypoints_target) ** 0.5
        arrived_loss = F.binary_cross_entropy_with_logits(arrive_pred.flatten(), arrived_target)

        # Compute direction loss
        wp_pred_view = wp_pred.view(-1, 2)
        wp_target_view = waypoints_target.view(-1, 2)

        # Compute cosine similarity
        dot_product = (wp_pred_view * wp_target_view).sum(dim=1)  # shape [batch_size]
        norm_pred = wp_pred_view.norm(dim=1)  # shape [batch_size]
        norm_target = wp_target_view.norm(dim=1)  # shape [batch_size]
        cos_sim = dot_product / (norm_pred * norm_target + 1e-8)  # avoid division by zero

        # Loss is 1 - cos_sim
        direction_loss = 1 - cos_sim.mean()

        return {'waypoints_loss': wp_loss, 'arrived_loss': arrived_loss, 'direction_loss': direction_loss, 'feature_loss': feature_loss}
    
    def compute_loss_polar(self, wp_pred_euclidean, distance_pred, angle_pred, arrive_pred, batch):
        waypoints_target = batch['waypoints']
        arrived_target = batch['arrived']
        
        # Compute distance and angle targets
        distance_target, angle_target = self.waypoints_to_polar(waypoints_target)
        
        # Compute L1 loss for distance and angle
        distance_loss = F.l1_loss(distance_pred, distance_target)
        angle_loss = F.l1_loss(angle_pred, angle_target)
        
        # Compute arrived loss
        arrived_loss = F.binary_cross_entropy_with_logits(arrive_pred.flatten(), arrived_target)
        
        # Compute direction loss using Euclidean waypoints
        wp_pred_last = wp_pred_euclidean[:, -1, :]  # shape [batch_size, 2]
        wp_target_last = waypoints_target[:, -1, :]  # shape [batch_size, 2]

        # Compute cosine similarity
        dot_product = (wp_pred_last * wp_target_last).sum(dim=1)  # shape [batch_size]
        norm_pred = wp_pred_last.norm(dim=1)  # shape [batch_size]
        norm_target = wp_target_last.norm(dim=1)  # shape [batch_size]
        cos_sim = dot_product / (norm_pred * norm_target + 1e-8)  # avoid division by zero

        # Loss is 1 - cos_sim
        direction_loss = (1 - cos_sim.mean()) ** 2

        return {'distance_loss': distance_loss, 'angle_loss': angle_loss, 'arrived_loss': arrived_loss, 'direction_loss': direction_loss}

    def compute_loss_diff_policy(self, wp_pred, noise_pred, arrived_pred, noise, batch):
        # Compute loss for noise prediction
        waypoints_target = batch['waypoints']
        noise_loss = F.mse_loss(noise_pred, noise)
        
        # Compute loss for arrived prediction
        arrived_target = batch['arrived']
        arrived_loss = F.binary_cross_entropy_with_logits(arrived_pred.flatten(), arrived_target)

        # Compute direction loss
        wp_pred_last = wp_pred[:, -1, :]  # shape [batch_size, 2]
        wp_target_last = waypoints_target[:, -1, :]  # shape [batch_size, 2]

        # Compute cosine similarity
        dot_product = (wp_pred_last * wp_target_last).sum(dim=1)  # shape [batch_size]
        norm_pred = wp_pred_last.norm(dim=1)  # shape [batch_size]
        norm_target = wp_target_last.norm(dim=1)  # shape [batch_size]
        cos_sim = dot_product / (norm_pred * norm_target + 1e-8)  # avoid division by zero

        # Loss is 1 - cos_sim
        direction_loss = (1 - cos_sim.mean()) ** 2
        
        return {'noise_loss': noise_loss, 'arrived_loss': arrived_loss, 'direction_loss': direction_loss}
    
    def configure_optimizers(self):
        optimizer_name = self.cfg.optimizer.name.lower()
        lr = float(self.cfg.optimizer.lr)
        scheduler_cfg = self.cfg.scheduler
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=self.cfg.optimizer.weight_decay)
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=self.cfg.optimizer.weight_decay)
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Scheduler
        if scheduler_cfg.name.lower() == 'step_lr':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_cfg.step_size, gamma=scheduler_cfg.gamma)
            return [optimizer], [scheduler]
        elif scheduler_cfg.name.lower() == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.training.max_epochs)
            return [optimizer], [scheduler]
        elif scheduler_cfg.name.lower() == 'onecycle':
            optimizer = torch.optim.AdamW(self.parameters(), lr=float(scheduler_cfg.max_lr) / scheduler_cfg.div_factor)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=float(scheduler_cfg.max_lr),
                epochs=self.cfg.training.max_epochs,
                steps_per_epoch=498,
                pct_start=0.3,
                anneal_strategy='cos',
                div_factor=scheduler_cfg.div_factor,
                final_div_factor=float(1e4)
            )
            return [optimizer], [scheduler]
        elif scheduler_cfg.name.lower() == 'none':
            return optimizer
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_cfg.name}")

    def process_visualization(self, mode, batch, obs, wp_pred, arrive_pred):
        """
        Handles visualization for both validation and testing.

        Args:
            mode (str): 'val' or 'test'
            batch (dict): Batch data
            obs (torch.Tensor): Observation frames
            wp_pred (torch.Tensor): Predicted waypoints
            arrive_pred (torch.Tensor): Predicted arrival logits
        """
        if mode == 'val':
            num_visualize = self.val_num_visualize
            vis_dir = os.path.join(self.result_dir, 'val_vis', f'epoch_{self.current_epoch}')
        elif mode == 'test':
            num_visualize = self.test_num_visualize
            vis_dir = os.path.join(self.result_dir, 'test_vis')
        else:
            raise ValueError("Mode should be either 'val' or 'test'.")

        os.makedirs(vis_dir, exist_ok=True)

        batch_size = obs.size(0)
        for idx in range(batch_size):
            if self.vis_count >= num_visualize:
                break

            # Extract necessary data
            arrived_target = batch['arrived'][idx].item()
            arrived_logits = arrive_pred[idx].flatten()
            arrived_probs = torch.sigmoid(arrived_logits).item()

            original_input_positions = batch['original_input_positions'][idx].cpu().numpy()
            noisy_input_positions = batch['noisy_input_positions'][idx].cpu().numpy()
            gt_waypoints = batch['gt_waypoints'][idx].cpu().numpy()
            pred_waypoints = wp_pred[idx].detach().cpu().numpy()
            target_transformed = batch['target_transformed'][idx].cpu().numpy()

            # if self.do_normalize:
            #     step_length = np.linalg.norm(gt_waypoints, axis=1).mean()
            #     original_input_positions = original_input_positions / step_length
            #     noisy_input_positions = noisy_input_positions / step_length
            #     gt_waypoints = gt_waypoints / step_length
            #     pred_waypoints = pred_waypoints / step_length
            #     target_transformed = target_transformed / step_length

            # Get the last frame from the sequence
            # Handle stereo format [B, N, 2, 3, H, W] or mono format [B, N, 3, H, W]
            if obs.dim() == 6:  # Stereo: use left camera (index 0)
                frame = obs[idx, -1, 0].permute(1, 2, 0).cpu().numpy()
            else:  # Mono
                frame = obs[idx, -1].permute(1, 2, 0).cpu().numpy()
            frame = (frame * 255).astype(np.uint8)  # Convert to uint8 for visualization

            # Visualization title
            arrive_title = f"Arrived GT: {'True' if arrived_target else 'False'}, Pred: {arrived_probs:.2f}"

                       # Plotting
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            plt.subplots_adjust(wspace=0.3)

            # Left axis: plot the current observation (frame) with arrived info in title
            ax1.imshow(frame)
            ax1.axis('off')
            ax1.set_title(arrive_title, fontsize=20)

            # Right axis: plot the coordinates
            ax2.axis('equal')
            ax2.plot(original_input_positions[:, 0], original_input_positions[:, 1],
                     'o-', label='Original Input Positions', color='#5771DB')
            ax2.plot(noisy_input_positions[:, 0], noisy_input_positions[:, 1],
                     'o-', label='Noisy Input Positions', color='#DBC257')
            ax2.plot(gt_waypoints[:, 0], gt_waypoints[:, 1],
                     'X-', label='GT Waypoints', color='#92DB58')
            ax2.plot(pred_waypoints[:, 0], pred_waypoints[:, 1],
                     's-', label='Predicted Waypoints', color='#DB6057')
            ax2.plot(target_transformed[0], target_transformed[1],
                     marker='*', markersize=15, label='Target Coordinate', color='#A157DB')
            # ax2.legend(fontsize=20)
            ax2.set_title('Coordinates', fontsize=20)
            ax2.set_xlabel('X (m)', fontsize=20)
            ax2.set_ylabel('Y (m)', fontsize=20)
            ax2.tick_params(axis='both', labelsize=18)
            ax2.grid(True)

            # Save the plot
            output_path = os.path.join(vis_dir, f'sample_{self.vis_count}.png')
            plt.savefig(output_path)
            plt.close(fig)

            data_to_save = {
                'arrived_target': arrived_target,
                'arrived_probs': arrived_probs,
                'original_input_positions': original_input_positions,
                'noisy_input_positions': noisy_input_positions,
                'gt_waypoints': gt_waypoints,
                'pred_waypoints': pred_waypoints,
                'target_transformed': target_transformed,
                'frame': frame,
            }
            if 'step_scale' in batch:
                data_to_save['step_scale'] = batch['step_scale'][idx].item() if batch['step_scale'].dim() == 1 else batch['step_scale'][idx].cpu().numpy()

            npz_path = os.path.join(vis_dir, f'sample_{self.vis_count}.npz')
            np.savez(npz_path, **data_to_save)

            self.vis_count += 1

    def waypoints_to_polar(self, waypoints):
        # Compute relative differences
        deltas = torch.diff(waypoints, dim=1, prepend=torch.zeros_like(waypoints[:, :1, :]))
        distance = torch.norm(deltas, dim=2)
        angle = torch.atan2(deltas[:, :, 1], deltas[:, :, 0])
        return distance, angle
