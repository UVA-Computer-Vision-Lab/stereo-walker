import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from efficientnet_pytorch import EfficientNet
from model.model_utils import PolarEmbedding, FeatTransformer, FeatPredictor, Tracktention
from torchvision import models
import pickle
import sys
import os
from argparse import Namespace

# Monster will be imported lazily in __init__ to avoid path issues


class TinyDepthEncoder(nn.Module):
    def __init__(self, in_ch=1, feature_dim=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 1, 1), nn.GELU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, 2, 1),   nn.GELU(), nn.BatchNorm2d(64),   # /2
            nn.Conv2d(64, 64, 3, 1, 1),   nn.GELU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, feature_dim, 3, 2, 1), nn.GELU(), nn.BatchNorm2d(feature_dim)  # /4
        )

    def forward(self, depth_map, Hp=25, Wp=25):
        feat = self.stem(depth_map)                       
        feat = F.adaptive_avg_pool2d(feat, (Hp, Wp))    
        feat = feat.permute(0, 2, 3, 1).contiguous()    
        return feat


class DeepWalker(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.context_size = cfg.model.obs_encoder.context_size
        self.obs_encoder_type = cfg.model.obs_encoder.type
        self.cord_embedding_type = cfg.model.cord_embedding.type
        self.decoder_type = cfg.model.decoder.type
        self.len_traj_pred = cfg.model.decoder.len_traj_pred
        self.do_rgb_normalize = cfg.model.do_rgb_normalize
        self.do_resize = cfg.model.do_resize
        self.output_coordinate_repr = cfg.model.output_coordinate_repr  # 'polar' or 'euclidean'

        # if self.obs_encoder_type.startswith("dinov2"):
        self.crop = cfg.model.obs_encoder.crop
        self.resize = cfg.model.obs_encoder.resize

        self.seq_len = ((self.resize[0] * self.resize[1] ) // 196 + 1) * self.context_size + 1

        if self.do_rgb_normalize:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Observation Encoder
        if self.obs_encoder_type.startswith("dinov2"):
            self.obs_encoder = torch.hub.load('facebookresearch/dinov2', self.obs_encoder_type)
            feature_dim = {
                "dinov2_vits14": 384,
                "dinov2_vitb14": 768,
                "dinov2_vitl14": 1024,
                "dinov2_vitg14": 1536,
            }
            if cfg.model.obs_encoder.freeze:
                for param in self.obs_encoder.parameters():
                    param.requires_grad = False
            self.num_obs_features = feature_dim[self.obs_encoder_type]
            self.num_total_features = self.num_obs_features + 64
            
            # Initialize Monster model for mono/stereo depth estimation
            # Map dinov2 encoder to Monster encoder naming
            encoder_map = {
                "dinov2_vits14": "vits",
                "dinov2_vitb14": "vitb", 
                "dinov2_vitl14": "vitl",
                "dinov2_vitg14": "vitg"
            }
            # monster_encoder = encoder_map.get(self.obs_encoder_type, "vits")
            monster_encoder = "vits"
            
            # Lazy import Monster to avoid path issues at module load time
            _project_root = '/bigtemp/tsx4zn/stereo-walker'
            _monster_path = os.path.join(_project_root, 'MonSter-plusplus/RT-MonSter++')
            _depth_anything_path = os.path.join(_project_root, 'MonSter-plusplus/RT-MonSter++/Depth-Anything-V2-list3')
            
            # Critical: Remove ALL conflicting depth_anything_v2 paths before importing Monster
            # This prevents Monster from importing the wrong version
            _original_sys_path = sys.path.copy()
            sys.path = [p for p in sys.path if 'Depth-Anything-V2' not in p or 'list3' in p]
            
            # Add correct paths at the beginning (highest priority)
            if _depth_anything_path not in sys.path:
                sys.path.insert(0, _depth_anything_path)
            if _monster_path not in sys.path:
                sys.path.insert(0, _monster_path)
            
            from core.monster import Monster
            
            # Restore other paths (but keep Monster paths at front)
            for p in _original_sys_path:
                if p not in sys.path and 'Depth-Anything-V2' not in p:
                    sys.path.append(p)
            
            # Create args for Monster model
            monster_args = Namespace(
                encoder=monster_encoder,
                hidden_dims=[32, 64, 96],
                n_gru_layers=3,
                max_disp=192,
                corr_radius=[2, 2, 4],
                mixed_precision=False,
                corr_implementation="reg",
                shared_backbone=False,
                n_downsample=2,
                slow_fast_gru=False
            )
            
            self.monster = Monster(monster_args)
            
            # Load Monster checkpoint
            monster_checkpoint_path = '/bigtemp/tsx4zn/stereo-walker/MonSter-plusplus/RT-MonSter++/checkpoint/sceneflow.pth'
            if os.path.exists(monster_checkpoint_path):
                try:
                    print(f"Loading Monster checkpoint from {monster_checkpoint_path}")
                    monster_state_dict = torch.load(monster_checkpoint_path, map_location='cpu')
                    # Handle DataParallel state dict
                    if 'model' in monster_state_dict:
                        monster_state_dict = monster_state_dict['model']
                    # Remove 'module.' prefix if exists
                    new_state_dict = {}
                    for k, v in monster_state_dict.items():
                        if k.startswith('module.'):
                            new_state_dict[k[7:]] = v
                        else:
                            new_state_dict[k] = v
                    missing_keys, unexpected_keys = self.monster.load_state_dict(new_state_dict, strict=False)
                    print(f"Monster weights loaded successfully")
                    if missing_keys:
                        print(f"Missing keys: {len(missing_keys)}")
                    if unexpected_keys:
                        print(f"Unexpected keys: {len(unexpected_keys)}")
                except Exception as e:
                    print(f"Warning: Could not load Monster checkpoint: {e}")
                    print("Initializing Monster with random weights (will use pretrained DepthAnything encoder)")
            
            # Freeze Monster if needed
            if cfg.model.obs_encoder.freeze:
                for param in self.monster.parameters():
                    param.requires_grad = False
            
            # Initialize DepthAnything V2 metric head for monocular mode
            # This provides absolute depth in meters (unlike Monster's relative depth)
            sys.path.insert(0, os.path.join(_project_root, 'Depth-Anything-V2/metric_depth'))
            from depth_anything_v2.dpt import DPTHead
            
            # Configuration based on encoder type
            depth_head_configs = {
                "dinov2_vits14": {"in_channels": 384, "features": 64, "out_channels": [48, 96, 192, 384]},
                "dinov2_vitb14": {"in_channels": 768, "features": 128, "out_channels": [96, 192, 384, 768]},
                "dinov2_vitl14": {"in_channels": 1024, "features": 256, "out_channels": [256, 512, 1024, 1024]},
            }
            
            head_config = depth_head_configs.get(self.obs_encoder_type, depth_head_configs["dinov2_vits14"])
            self.depth_head = DPTHead(
                in_channels=head_config["in_channels"],
                features=head_config["features"],
                use_bn=False,
                out_channels=head_config["out_channels"],
                use_clstoken=False
            )
            self.intermediate_layer_idx = [2, 5, 8, 11] if "vits" in self.obs_encoder_type or "vitb" in self.obs_encoder_type else [4, 11, 17, 23]
            
            # Load DepthAnything V2 metric weights
            # Map encoder type to checkpoint
            encoder_to_ckpt = {
                "dinov2_vits14": "depth_anything_v2_metric_vkitti_vits.pth",
                "dinov2_vitb14": "depth_anything_v2_metric_vkitti_vitb.pth",
                "dinov2_vitl14": "depth_anything_v2_metric_vkitti_vitl.pth",
            }
            
            ckpt_name = encoder_to_ckpt.get(self.obs_encoder_type, "depth_anything_v2_metric_vkitti_vitb.pth")
            depth_metric_ckpt = f'/bigtemp/tsx4zn/stereo-walker/Depth-Anything-V2/metric_depth/checkpoints/{ckpt_name}'
            
            if os.path.exists(depth_metric_ckpt):
                try:
                    print(f"Loading DepthAnything V2 metric checkpoint from {depth_metric_ckpt}")
                    depth_state = torch.load(depth_metric_ckpt, map_location='cpu', weights_only=False)
                    # Extract depth_head weights
                    depth_head_state = {}
                    for k, v in depth_state.items():
                        if k.startswith('depth_head.'):
                            depth_head_state[k.replace('depth_head.', '')] = v
                    if depth_head_state:
                        missing, unexpected = self.depth_head.load_state_dict(depth_head_state, strict=False)
                        if len(missing) == 0:
                            print(f"DepthAnything V2 metric head loaded successfully ✅")
                        else:
                            print(f"DepthAnything V2 metric head loaded with {len(missing)} missing keys")
                    else:
                        print(f"Warning: No depth_head weights found, trying full state dict")
                        self.depth_head.load_state_dict(depth_state, strict=False)
                except Exception as e:
                    print(f"Warning: Could not load depth metric checkpoint: {e}")
            else:
                print(f"Warning: Metric checkpoint not found: {depth_metric_ckpt}")
                print(f"Will use Monster mono depth instead for monocular mode")
                self.depth_head = None  # 标记没有metric权重，使用Monster
            
            # Freeze depth head if needed
            if cfg.model.obs_encoder.freeze and self.depth_head is not None:
                for param in self.depth_head.parameters():
                    param.requires_grad = False
                    
            # Camera intrinsics for depth <-> disparity conversion
            # Canon R5 with RF5.2mm F2.8 L DUAL FISHEYE lens parameters
            # Baseline: 60mm between the two fisheye lenses
            # 
            # CALIBRATED from 10 videos (test_absolute_depth_consistency.py):
            # - Monocular: DepthAnything V2 metric (consistent depth, 3.10% variability) ✅
            # - Stereo: Monster stereo matching (absolute disparity) ✅
            # - Measured: depth_metric × disparity ≈ 12.43 m·px (highly consistent!)
            # - Theoretical formula: baseline × focal = depth × disparity
            # - Calculated: focal = 12.43 / 0.06 = 207.25px
            # - Corresponding HFOV ≈ 102° (very reasonable for fisheye rectification!)
            #
            # Note: rectify.py's 60° HFOV (focal=443.41) is target crop FOV,
            #       but actual depth measurements suggest effective HFOV ≈ 102°
            self.baseline = 0.06  # meters (60mm lens separation - fixed)
            self.focal_length = 207.25  # pixels (CALIBRATED, effective HFOV≈102°, 3.10% variability)
            # Note: Use model.set_camera_intrinsics() to override if needed
            
        else:
            raise NotImplementedError(f"Observation encoder type {self.obs_encoder_type} not implemented")
        self.depth_encoder = TinyDepthEncoder(in_ch=1, feature_dim=64)
        self.tracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        # Freeze tracker parameters
        for param in self.tracker.parameters():
            param.requires_grad = False
        self.time_embedding = Tracktention(
            feature_dim=self.num_total_features, 
            d_k=64,  # Reduced from 256
            sigma=1.0, 
            use_rope=True,
            num_heads=4,  # Reduced from 8
            use_qk_norm=True,
            use_track_transformer=True,  
            track_transformer_heads=4,
            track_transformer_layers=2,  
            use_splatting=True,  # Disabled to reduce parameters
            splatting_heads=4,
            use_flash_attention=True  # Enable optimized attention
        )
        # Latent vector covariance needed
        self.latent = nn.Parameter(torch.randn(1, 625, self.num_total_features))
        
        # Coordinate Embedding
        if self.cord_embedding_type == 'input_target':
            self.cord_embedding = PolarEmbedding(cfg)
            self.compress_goal_enc = nn.Linear(self.cord_embedding.out_dim, self.num_total_features)
        else:
            raise NotImplementedError(f"Coordinate embedding type {self.cord_embedding_type} not implemented")

        # Decoder
        self.encoder = FeatTransformer(
            embed_dim=self.num_total_features,
            nhead=cfg.model.decoder.num_heads,
            num_layers=12,
            ff_dim_factor=cfg.model.decoder.ff_dim_factor,
        )
        # Second transformer for final processing
        self.decoder = FeatTransformer(
            embed_dim=self.num_total_features,
            nhead=cfg.model.decoder.num_heads,
            num_layers=4,
            ff_dim_factor=cfg.model.decoder.ff_dim_factor,
        )
        self.predictor_mlp = nn.Sequential(
            nn.Linear(self.num_total_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.wp_predictor = nn.Linear(32, self.len_traj_pred * 2)
        self.arrive_predictor = nn.Linear(32, 1)
    
    def set_camera_intrinsics(self, baseline, focal_length):
        """
        Set camera intrinsics for depth to disparity conversion
        
        Args:
            baseline: stereo baseline in meters
            focal_length: focal length in pixels
        """
        self.baseline = baseline
        self.focal_length = focal_length
        print(f"Camera intrinsics updated: baseline={baseline}m, focal_length={focal_length}px")
    
    def forward(self, obs, cord, future_obs=None):
        """
        Args:
            obs: Observation images in one of two formats:
                - Monocular: (B, N, 3, H, W) - from CityWalkFeatDataset
                - Stereo: (B, N, 2, 3, H, W) - from StereoWalkDataset (dim=2 for left/right)
            cord: (B, N+1, 2) tensor - input positions + target position
            future_obs: NOT USED (kept for backward compatibility with pl_modules)
        
        Returns:
            wp_pred: (B, len_traj_pred, 2) - predicted waypoints
            arrive_pred: (B, 1) - arrival prediction
            feature_pred: (B, N, HW, feature_dim) - predicted features
            None: placeholder (future_obs_enc removed)
        """
        # future_obs is ignored (not used)
        # Auto-detect stereo vs monocular based on input shape
        if obs.dim() == 6:  # Stereo: (B, N, 2, 3, H, W)
            is_stereo = True
            B, N, _, _, H, W = obs.shape
            # Split left and right images
            obs_left = obs[:, :, 0, :, :, :]  # (B, N, 3, H, W)
            obs_right = obs[:, :, 1, :, :, :]  # (B, N, 3, H, W)
            obs = obs_left.view(B * N, 3, H, W)  # Process left images
            stereo_right = obs_right.view(B * N, 3, H, W)
        elif obs.dim() == 5:  # Monocular: (B, N, 3, H, W)
            is_stereo = False
            B, N, _, H, W = obs.shape
            obs = obs.view(B * N, 3, H, W)
            stereo_right = None
        else:
            raise ValueError(f"Unexpected obs shape: {obs.shape}. "
                           f"Expected (B, N, 3, H, W) for mono or (B, N, 2, 3, H, W) for stereo.")
        
        # Save original obs for Monster (before resize/normalize)
        if is_stereo:
            obs_original = TF.resize(obs, [352, 352])
            stereo_right_original = TF.resize(stereo_right, [352, 352])
            
        # Normalize images (for DINOv2 and other components)
        if self.do_rgb_normalize:
            obs = (obs - self.mean) / self.std
            if is_stereo and stereo_right is not None:
                stereo_right = (stereo_right - self.mean) / self.std
                
        # Resize images (for DINOv2 and other components)
        if self.do_resize:
            obs = TF.center_crop(obs, self.crop)
            obs = TF.resize(obs, self.resize) 
            if is_stereo and stereo_right is not None:
                stereo_right = TF.center_crop(stereo_right, self.crop)
                stereo_right = TF.resize(stereo_right, self.resize)
        # Reshape obs back to (B, N, C, H, W) for co-tracker
        obs_for_tracker = obs.view(B, N, 3, obs.shape[-2], obs.shape[-1])
        pred_tracks, _ = self.tracker(obs_for_tracker * 255, grid_size = 10) # [B, 5, 100, 2]
        
        # Reshape obs for DINOv2 encoder: (B*N, C, H, W)
        obs_reshaped = obs.view(B * N, 3, obs.shape[-2], obs.shape[-1])
        
        # Get DINOv2 features for main encoder
        depth_intermediate_features = self.obs_encoder.get_intermediate_layers(
            obs_reshaped, 
            [2, 5, 8, 11] if self.obs_encoder_type == "dinov2_vits14" else [2, 5, 8, 11], 
            return_class_token=True
        )
        obs_enc = depth_intermediate_features[0][0].view(B, N, -1, self.num_obs_features)
        
        # Get depth/disparity using Monster
        # Use original obs (before resize/normalize) for Monster
        # obs_original is already (B*N, 3, H, W) shape, no need to reshape
        if is_stereo and stereo_right_original is not None:
            # Stereo mode: use Monster's full forward to get disparity
            obs_monster = obs_original  # Already in 0-255 range, original size
            stereo_right_monster = stereo_right_original  # Already in 0-255 range, original size
            # Process frames in mini-batches for better GPU utilization across all devices
            batch_size_monster = B  
            disparity_maps = []
            for i in range(0, B * N, batch_size_monster):
                end_idx = min(i + batch_size_monster, B * N)
                # Monster expects [B, C, H, W], returns disparity
                disp = self.monster(
                    obs_monster[i:end_idx], 
                    stereo_right_monster[i:end_idx],
                    iters=4,
                    test_mode=True
                )
                disparity_maps.append(disp)
            disparity_map = torch.cat(disparity_maps, dim=0)  # [B*N, 1, H, W]
            disparity_map = F.interpolate(disparity_map, size=(obs.shape[-2], obs.shape[-1]), 
                                        mode='bilinear', align_corners=False)
            disparity_map = (self.baseline * self.focal_length) / (disparity_map + 1e-6)
        else:
            # Monocular mode
            # Use DepthAnything V2 metric (absolute depth, more consistent)
            # Reuse depth_intermediate_features already computed above (no redundant computation!)
            
            # Compute metric depth using depth_head for all frames at once
            patch_h, patch_w = obs_reshaped.shape[-2] // 14, obs_reshaped.shape[-1] // 14
            disparity_map = self.depth_head(depth_intermediate_features, patch_h, patch_w)
            # depth_map shape: [B*N, 1, patch_h*14, patch_w*14]
            
            # Convert metric depth to disparity
            # disparity_map = (self.baseline * self.focal_length) / (disparity_map + 1e-6)
            
        # Get target feature map size from obs_enc
        # obs_enc shape: [B, N, num_patches, features]
        # num_patches = (H/14) * (W/14) for ViT with patch_size=14
        # For 350x350 image: num_patches = 625, Hp=Wp=25
        target_h = obs.shape[-2] // 14  # H / 14
        target_w = obs.shape[-1] // 14  # W / 14
        
        # Use TinyDepthEncoder to encode disparity/depth to match obs_enc size
        depth_enc = self.depth_encoder(disparity_map, Hp=target_h, Wp=target_w).view(B, N, -1, 64)
        
        image_size = (obs.shape[-2], obs.shape[-1])  # (350, 350)
        obs_enc = torch.cat([obs_enc, depth_enc], dim=-1)
        tracktention_result = self.time_embedding(obs_enc, pred_tracks, self.num_total_features, image_size)
        
        # Handle different return values based on splatting configuration
        if isinstance(tracktention_result, tuple) and len(tracktention_result) == 2:
            chrono, updated_obs_enc = tracktention_result
        else:
            chrono = tracktention_result
            updated_obs_enc = None

        # future_obs feature encoding removed (not used)
        
        # Coordinate Encoding
        cord_enc = self.cord_embedding(cord) # cord.size() = torch.Size([B, 6, 2])
        cord_enc = self.compress_goal_enc(cord_enc) # torch.Size([B, 6, 768])
        
        # Use updated feature map if available from splatting, otherwise use original
        feature_map_to_use = updated_obs_enc if updated_obs_enc is not None else obs_enc
        
        obs_and_cord = torch.cat([cord_enc[:, :-1, :].unsqueeze(2), feature_map_to_use], dim=2)
        
        # Add latent vector to tokens
        latent_batch = self.latent.expand(B, -1, -1)  # [B, 625, 768]
        tokens = torch.cat([obs_and_cord.view(B, -1, self.num_total_features), latent_batch], dim=1)
        # First transformer
        feature_pred = self.encoder(tokens) # [B, seq_len+625+1, 768]
        
        # Add cord_enc[:, -1:, :] and pass through second transformer
        final_tokens = torch.cat([feature_pred, cord_enc[:, -1:, :]], dim=1)
        final_output = self.decoder(final_tokens) # [B, seq_len+625+2, 768]

        # Take the cord_enc[:, -1, :] part (last token) for predictor_mlp
        dec_out = self.predictor_mlp(final_output[:, -1]) # [B, 32]
        
        # Calculate the correct dimensions for feature_pred reshaping
        # feature_pred shape: [B, seq_len, D_f] where seq_len includes coordinate tokens + feature tokens + latent tokens
        # We want to extract the feature part for each time step
        hw_size = feature_map_to_use.shape[2]  # HW dimension from the feature map
        
        # The sequence structure is: [coord_tokens, feature_tokens, latent_tokens, goal_token]
        # coord_tokens: (N-1) tokens (one per time step except last)
        # feature_tokens: N * hw_size tokens (feature map for each time step)
        # latent_tokens: 625 tokens
        # goal_token: 1 token
        
        # Extract the feature tokens part (excluding coord, latent, and goal tokens)
        coord_tokens_count = N - 1  # Number of coordinate tokens
        latent_tokens_count = 625   # Number of latent tokens
        goal_tokens_count = 1       # Number of goal tokens
        
        # Calculate the start and end indices for feature tokens
        # We need N time steps, not N-1
        feature_start_idx = coord_tokens_count
        feature_end_idx = feature_start_idx + N * hw_size
        
        # Extract feature tokens and reshape
        feature_tokens = feature_pred[:, feature_start_idx:feature_end_idx, :]
        feature_pred_reshaped = feature_tokens.view(B, N, hw_size, self.num_total_features) # [B, N, HW, 768]
        wp_pred = self.wp_predictor(dec_out).view(B, self.len_traj_pred, 2) 
        arrive_pred = self.arrive_predictor(dec_out).view(B, 1) 
        # Waypoint Prediction Processing
        if self.output_coordinate_repr == 'euclidean':
            # Predict deltas and compute cumulative sum
            wp_pred = torch.cumsum(wp_pred, dim=1)
            return wp_pred, arrive_pred, feature_pred_reshaped, None  # future_obs_enc removed
        else:
            raise NotImplementedError(f"Output coordinate representation {self.output_coordinate_repr} not implemented")
