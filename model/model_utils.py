import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PolarEmbedding(nn.Module):
    def __init__(self, cfg):
        super(PolarEmbedding, self).__init__()
        self.num_freqs = cfg.model.cord_embedding.num_freqs
        self.include_input = cfg.model.cord_embedding.include_input
        # Register freq_bands as a buffer to ensure it's moved to the correct device
        freq_bands = 2.0 ** torch.linspace(0, self.num_freqs - 1, self.num_freqs)
        self.register_buffer('freq_bands', freq_bands)
        self.out_dim = 2 + 4 * self.num_freqs if self.include_input else 4 * self.num_freqs

    def forward(self, coords):
        """
        Args:
            coords: Tensor of shape (B, N, 2)
        
        Returns:
            Tensor of shape (B, N, D) where D = 2 (if include_input) + 4 * num_freqs
        """
        # Ensure coords has the correct shape
        if coords.dim() != 3 or coords.size(-1) != 2:
            raise ValueError(f"Expected coords of shape (B, N, 2), but got {coords.shape}")
        
        x, y = coords[..., 0], coords[..., 1]  # Shape: (B, N)
        r = torch.sqrt(x**2 + y**2).unsqueeze(-1)            # Shape: (B, N)
        theta = torch.atan2(y, x).unsqueeze(-1)              # Shape: (B, N)
        
        enc = [r, theta] if self.include_input else []
        
        # Expand freq_bands to (1, 1, num_freqs) for broadcasting
        freq_bands = self.freq_bands.view(1, 1, -1)  # Shape: (1, 1, num_freqs)
        
        # Compute sin and cos for theta and r with frequency bands
        enc.append(torch.sin(theta * freq_bands))  # Shape: (B, N, num_freqs)
        enc.append(torch.cos(theta * freq_bands))  # Shape: (B, N, num_freqs)
        enc.append(torch.sin(r * freq_bands))      # Shape: (B, N, num_freqs)
        enc.append(torch.cos(r * freq_bands))      # Shape: (B, N, num_freqs)
        
        # Concatenate all encodings along the last dimension
        enc = torch.cat(enc, dim=-1)  # Shape: (B, N, D)
        
        return enc

class PositionalEncoding(nn.Module):
    """
    Borrowed from https://github.com/robodhruv/visualnav-transformer/blob/main/train/vint_train/models/vint/self_attention.py
    """
    def __init__(self, d_model, max_seq_len=6):
        super().__init__()

        # Compute the positional encoding once
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0)

        # Register the positional encoding as a buffer to avoid it being
        # considered a parameter when saving the model
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        # Add the positional encoding to the input
        x = x + self.pos_enc[:, :x.size(1), :]
        return x
    
class DynamicPE(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
    
    def forward(self, x):
        # x: (B, T, D)
        max_seq_len = x.shape[1]
        device, dtype = x.device, x.dtype
        pos = torch.arange(max_seq_len, device=device, dtype=dtype).unsqueeze(1)      
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device, dtype=dtype)
                             * (-math.log(10000.0) / self.d_model))    
        pe = torch.zeros((max_seq_len, self.d_model), device=device, dtype=dtype)  
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return x + pe.unsqueeze(0)  

class MultiLayerDecoder(nn.Module):
    """
    Borrowed from https://github.com/robodhruv/visualnav-transformer/blob/main/train/vint_train/models/vint/self_attention.py
    """
    def __init__(self, embed_dim=512, seq_len=6, output_layers=[256, 128, 64], nhead=8, num_layers=8, ff_dim_factor=4):
        super(MultiLayerDecoder, self).__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len=seq_len)
        self.sa_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim_factor*embed_dim, activation="gelu", batch_first=True, norm_first=True)
        self.sa_decoder = nn.TransformerEncoder(self.sa_layer, num_layers=num_layers)
        self.output_layers = nn.ModuleList([nn.Linear(seq_len*embed_dim, embed_dim)])
        self.output_layers.append(nn.Linear(embed_dim, output_layers[0]))
        for i in range(len(output_layers)-1):
            self.output_layers.append(nn.Linear(output_layers[i], output_layers[i+1]))

    def forward(self, x):
        if self.positional_encoding: x = self.positional_encoding(x)
        x = self.sa_decoder(x)
        # currently, x is [batch_size, seq_len, embed_dim]
        x = x.reshape(x.shape[0], -1)
        for i in range(len(self.output_layers)):
            x = self.output_layers[i](x)
            x = F.relu(x)
        return x
    
class FeatPredictor(nn.Module):
    def __init__(self, embed_dim=512, seq_len=6, nhead=8, num_layers=8, ff_dim_factor=4):
        super().__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len=seq_len)
        self.sa_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim_factor*embed_dim, activation="gelu", batch_first=True, norm_first=True)
        self.sa_decoder = nn.TransformerEncoder(self.sa_layer, num_layers=num_layers)
    
    def forward(self, x):
        if self.positional_encoding: x = self.positional_encoding(x)
        x = self.sa_decoder(x)
        return x
    
class FeatTransformer(nn.Module):
    def __init__(self, embed_dim=512, nhead=8, num_layers=8, ff_dim_factor=4):
        super().__init__()
        self.positional_encoding = DynamicPE(embed_dim)
        self.sa_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim_factor*embed_dim, activation="gelu", batch_first=True, norm_first=True)
        self.sa_decoder = nn.TransformerEncoder(self.sa_layer, num_layers=num_layers)

    def forward(self, x):
        if self.positional_encoding: x = self.positional_encoding(x)
        x = self.sa_decoder(x)
        return x

class FrameTransformer(nn.Module):
    def __init__(self, batch_size, embed_dim=512, nhead=8, num_layers=8, ff_dim_factor=4, context_size=5):
        super().__init__()
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.context_size = context_size
        self.positional_encoding = DynamicPE(embed_dim)
        self.sa_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim_factor*embed_dim, activation="gelu", batch_first=True, norm_first=True)
        self.sa_decoder = nn.TransformerEncoder(self.sa_layer, num_layers=num_layers)

    def forward(self, x):
        target = x[:, -1:, :]
        x = x[:, :-1, :].view(self.batch_size*self.context_size, -1, self.embed_dim)
        if self.positional_encoding: x = self.positional_encoding(x)
        x = self.sa_decoder(x)
        return torch.cat([x.view(self.batch_size, -1, self.embed_dim), target], dim=1)

class GlobalTransformer(nn.Module):
    def __init__(self, batch_size, embed_dim=512, nhead=8, num_layers=8, ff_dim_factor=4, context_size=5):
        super().__init__()
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.context_size = context_size
        self.positional_encoding = DynamicPE(embed_dim)
        self.sa_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim_factor*embed_dim, activation="gelu", batch_first=True, norm_first=True)
        self.sa_decoder = nn.TransformerEncoder(self.sa_layer, num_layers=num_layers)

    def forward(self, x):
        if self.positional_encoding: x = self.positional_encoding(x)
        x = self.sa_decoder(x)
        return x

class RoPE2D(nn.Module):
    """
    Rotational Position Encodings (RoPE) for 2D spatial locations.
    
    Implements RoPE as specified in the paper:
    - First d/2 channels encode x position
    - Second d/2 channels encode y position
    - Uses the formula: RoPE(f)[2i] = f[2i] * cos(θ_i * x) - f[2i+1] * sin(θ_i * x)
    - Where θ_i = 100^(-2(i-1)/(d/2))
    """
    
    def __init__(self, feature_dim: int, base_freq: float = 100.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.base_freq = base_freq
        
        # Ensure feature dimension is divisible by 4 (d/2 for x, d/2 for y)
        assert feature_dim % 4 == 0, f"Feature dimension {feature_dim} must be divisible by 4"
        
        # Compute frequency components for x and y dimensions
        # For x: use first d/2 channels, for y: use second d/2 channels
        d_half = feature_dim // 2
        d_quarter = feature_dim // 4
        
        # Compute θ_i = 100^(-2(i-1)/(d/2)) for i = 1, ..., d/4
        i_range = torch.arange(1, d_quarter + 1, dtype=torch.float32)
        theta = self.base_freq ** (-2 * (i_range - 1) / d_half)
        
        # Register as buffer so it's moved to the correct device
        self.register_buffer('theta', theta)
    
    def apply_rope_1d(self, features: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to features along one dimension (x or y).
        
        Args:
            features: (..., d/2) feature tensor for one dimension
            positions: (...,) position tensor
            
        Returns:
            Features with RoPE applied
        """
        d_half = self.feature_dim // 2
        d_quarter = self.feature_dim // 4
        
        # Reshape to (..., d/4, 2) for complex number representation
        feat_reshaped = features.reshape(*features.shape[:-1], d_quarter, 2)  # (..., d/4, 2)
        
        # Apply RoPE formula
        # For each i = 1, ..., d/4:
        # RoPE(f)[2i] = f[2i] * cos(θ_i * pos) - f[2i+1] * sin(θ_i * pos)
        # RoPE(f)[2i+1] = f[2i] * sin(θ_i * pos) + f[2i+1] * cos(θ_i * pos)
        
        # Expand positions to match theta dimensions
        pos_expanded = positions.unsqueeze(-1)  # (..., 1)
        theta_expanded = self.theta.unsqueeze(0)  # (1, d/4)
        
        # Compute angles: θ_i * pos
        angles = pos_expanded * theta_expanded  # (..., d/4)
        
        # Compute cos and sin
        cos_vals = torch.cos(angles)  # (..., d/4)
        sin_vals = torch.sin(angles)  # (..., d/4)
        
        # Apply rotation
        f_even = feat_reshaped[..., 0]  # f[2i]
        f_odd = feat_reshaped[..., 1]   # f[2i+1]
        
        # RoPE(f)[2i] = f[2i] * cos(θ_i * pos) - f[2i+1] * sin(θ_i * pos)
        rope_even = f_even * cos_vals - f_odd * sin_vals
        
        # RoPE(f)[2i+1] = f[2i] * sin(θ_i * pos) + f[2i+1] * cos(θ_i * pos)
        rope_odd = f_even * sin_vals + f_odd * cos_vals
        
        # Combine back to (..., d/2)
        rope_result = torch.stack([rope_even, rope_odd], dim=-1)  # (..., d/4, 2)
        rope_result = rope_result.reshape(*rope_result.shape[:-2], d_half)  # (..., d/2)
        
        return rope_result
    
    def forward(self, features: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Apply 2D RoPE to features based on spatial positions.
        
        Args:
            features: (..., d) feature tensor
            positions: (..., 2) position tensor with (x, y) coordinates
            
        Returns:
            Features with 2D RoPE applied
        """
        d_half = self.feature_dim // 2
        
        # Split features into x and y components
        x_features = features[..., :d_half]  # First d/2 channels for x
        y_features = features[..., d_half:]  # Second d/2 channels for y
        
        # Extract x and y positions
        x_pos = positions[..., 0]  # (...,)
        y_pos = positions[..., 1]  # (...,)
        
        # Apply RoPE to x and y components
        x_rope = self.apply_rope_1d(x_features, x_pos)
        y_rope = self.apply_rope_1d(y_features, y_pos)
        
        # Combine results
        rope_features = torch.cat([x_rope, y_rope], dim=-1)
        
        return rope_features


class QKNormalization(nn.Module):
    """
    QK-normalization for stabilizing training by normalizing query and key scales.
    
    This helps prevent gradient instability caused by differences in scale between
    track token embeddings and feature map embeddings.
    """
    
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, Q, K):
        """
        Apply QK-normalization to queries and keys.
        
        Args:
            Q: (..., seq_len_q, d_k) query tensor
            K: (..., seq_len_k, d_k) key tensor
            
        Returns:
            Tuple of (normalized_Q, normalized_K)
        """
        # Compute L2 norms along the last dimension
        Q_norm = torch.norm(Q, dim=-1, keepdim=True)  # (..., seq_len_q, 1)
        K_norm = torch.norm(K, dim=-1, keepdim=True)  # (..., seq_len_k, 1)
        
        # Normalize with small epsilon to prevent division by zero
        Q_normalized = Q / (Q_norm + self.eps)
        K_normalized = K / (K_norm + self.eps)
        
        return Q_normalized, K_normalized


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention implementation with QK-normalization and optimized attention.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_k: Key/Query dimension per head
        use_qk_norm: Whether to use QK-normalization
        use_flash_attention: Whether to use optimized attention (PyTorch 2.0+)
    """
    
    def __init__(self, d_model, num_heads, d_k=None, use_qk_norm=True, use_flash_attention=True, eps=1e-6):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k if d_k is not None else d_model // num_heads
        self.use_qk_norm = use_qk_norm
        self.use_flash_attention = use_flash_attention
        
        # Ensure d_model is divisible by num_heads
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        
        # Output projection
        self.W_o = nn.Linear(num_heads * self.d_k, d_model, bias=False)
        
        # QK-normalization
        if self.use_qk_norm:
            self.qk_norm = QKNormalization(eps=eps)
        
        # Scale factor for attention
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        # Check if optimized attention is available
        if self.use_flash_attention:
            try:
                from torch.nn.functional import scaled_dot_product_attention
                self.scaled_dot_product_attention = scaled_dot_product_attention
            except ImportError:
                print("Warning: scaled_dot_product_attention not available, falling back to standard attention")
                self.use_flash_attention = False
    
    def forward(self, Q_input, K_input, V_input, bias=None, return_attention=False):
        """
        Multi-head attention forward pass with optimized attention.
        
        Args:
            Q_input: (..., seq_len_q, d_model) query input
            K_input: (..., seq_len_k, d_model) key input  
            V_input: (..., seq_len_v, d_model) value input
            bias: (..., num_heads, seq_len_q, seq_len_k) optional bias
            return_attention: Whether to return attention weights
            
        Returns:
            (..., seq_len_q, d_model) attended output
            If return_attention=True, also returns attention weights
        """
        batch_size = Q_input.shape[0]
        seq_len_q = Q_input.shape[-2]
        seq_len_k = K_input.shape[-2]
        seq_len_v = V_input.shape[-2]
        
        # Linear projections
        Q = self.W_q(Q_input)  # (..., seq_len_q, num_heads * d_k)
        K = self.W_k(K_input)  # (..., seq_len_k, num_heads * d_k)
        V = self.W_v(V_input)  # (..., seq_len_v, num_heads * d_k)
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)  # (..., num_heads, seq_len_q, d_k)
        K = K.reshape(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)  # (..., num_heads, seq_len_k, d_k)
        V = V.reshape(batch_size, seq_len_v, self.num_heads, self.d_k).transpose(1, 2)  # (..., num_heads, seq_len_v, d_k)
        
        # Apply QK-normalization if enabled
        if self.use_qk_norm:
            Q, K = self.qk_norm(Q, K)
        
        attention_weights = None
        
        if self.use_flash_attention and bias is None and not return_attention:
            # Use optimized attention when no bias is provided and attention weights not needed
            # Reshape for scaled_dot_product_attention: (batch, num_heads, seq_len, head_dim)
            attended = self.scaled_dot_product_attention(
                Q, K, V, 
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scale
            )  # (batch, num_heads, seq_len_q, d_k)
        else:
            # Fallback to standard attention (when bias is provided, flash attention unavailable, or attention weights needed)
            # Compute attention scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (..., num_heads, seq_len_q, seq_len_k)
            
            # Add bias if provided
            if bias is not None:
                scores = scores + bias
            
            # Apply softmax
            attention_weights = F.softmax(scores, dim=-1)  # (..., num_heads, seq_len_q, seq_len_k)
            
            # Apply attention to values
            attended = torch.matmul(attention_weights, V)  # (..., num_heads, seq_len_q, d_k)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().reshape(batch_size, seq_len_q, self.num_heads * self.d_k)  # (..., seq_len_q, num_heads * d_k)
        
        # Output projection
        output = self.W_o(attended)  # (..., seq_len_q, d_model)
        
        if return_attention:
            return output, attention_weights
        else:
            return output


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal sequences.
    
    Provides the model with a sense of temporal order by encoding
    position information using sine and cosine functions.
    """
    
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute div_term for sinusoidal encoding
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer so it's moved to the correct device
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """
        Add positional encoding to input.
        
        Args:
            x: (batch_size, seq_len, d_model) input tensor
            
        Returns:
            Input with positional encoding added
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer with multi-head self-attention and feed-forward network.
    """
    
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        
        # Multi-head self-attention
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            use_qk_norm=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, self.d_ff),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass of transformer encoder layer.
        
        Args:
            x: (batch_size, seq_len, d_model) input tensor
            
        Returns:
            (batch_size, seq_len, d_model) output tensor
        """
        # Multi-head self-attention with residual connection
        attn_output = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward network with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x
    

class TrackTransformer(nn.Module):
    """
    Track Transformer for processing track tokens along temporal dimension.
    
    Processes track tokens S of shape (T, M, D_f) by:
    1. Swapping dimensions to (M, T, D_f) 
    2. Applying 2-layer transformer encoder along temporal dimension
    3. Swapping back to (T, M, D_f)
    
    This allows each track to attend to all time steps within that track,
    making tracks more temporally consistent.
    """
    
    def __init__(self, d_model, num_heads=8, num_layers=2, d_ff=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Sinusoidal positional encoding for temporal order
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, track_tokens):
        """
        Process track tokens along temporal dimension.
        
        Args:
            track_tokens: (T, M, D_f) track tokens from sampling
            
        Returns:
            (T, M, D_f) updated track tokens
        """
        T, M, D_f = track_tokens.shape
        
        # Swap dimensions: (T, M, D_f) -> (M, T, D_f)
        # This treats each track as a separate sequence
        track_tokens_swapped = track_tokens.transpose(0, 1)  # (M, T, D_f)
        
        # Add positional encoding for temporal order
        track_tokens_with_pos = self.pos_encoding(track_tokens_swapped)
        
        # Apply transformer encoder layers
        x = track_tokens_with_pos
        for layer in self.layers:
            x = layer(x)
        
        # Final layer normalization
        x = self.norm(x)
        
        # Swap dimensions back: (M, T, D_f) -> (T, M, D_f)
        updated_track_tokens = x.transpose(0, 1)
        
        return updated_track_tokens


class AttentionalSplatting(nn.Module):
    """
    Attentional Splatting module for mapping updated track tokens back to feature maps.
    
    Reverses the roles of track tokens and feature tokens:
    - Queries: derived from grid coordinates of output feature map
    - Keys/Values: generated from track tokens
    - Uses transposed bias term B_t^T
    - Applies same QK-normalization and RoPE encodings
    """
    
    def __init__(self, feature_dim, d_k, num_heads=8, use_qk_norm=True, use_rope=True, use_flash_attention=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_k = d_k
        self.num_heads = num_heads
        self.use_qk_norm = use_qk_norm
        self.use_rope = use_rope
        
        # Multi-head attention for splatting (reversed roles)
        self.attention = MultiHeadAttention(
            d_model=feature_dim,
            num_heads=num_heads,
            d_k=d_k // num_heads,
            use_qk_norm=use_qk_norm,
            use_flash_attention=use_flash_attention
        )
        
        # Output projection
        self.W_out = nn.Linear(feature_dim, feature_dim)
        
        # RoPE for feature map coordinates (if enabled)
        if self.use_rope:
            # Ensure feature dimension is divisible by 4 for RoPE
            if feature_dim % 4 != 0:
                self.feature_dim_padded = ((feature_dim + 3) // 4) * 4
                self.feature_padding = nn.Linear(feature_dim, self.feature_dim_padded)
                self.feature_unpadding = nn.Linear(self.feature_dim_padded, feature_dim)
            else:
                self.feature_dim_padded = feature_dim
                self.feature_padding = None
                self.feature_unpadding = None
            
            self.rope = RoPE2D(self.feature_dim_padded, base_freq=100.0)
    
    def forward(self, track_tokens, feature_map, feature_positions, spatial_bias):
        """
        Map updated track tokens back to feature maps.
        
        Args:
            track_tokens: (T, M, D_f) updated track tokens
            feature_map: (T, HW, D_f) original feature map
            feature_positions: (T, HW, 2) feature map positions
            spatial_bias: (T, M, HW) spatial bias (will be transposed)
            
        Returns:
            (T, HW, D_f) updated feature map
        """
        T, M, D_f = track_tokens.shape
        _, HW, _ = feature_map.shape
        
        # Step 1: Prepare queries from feature map coordinates
        # Create positional embeddings for feature map coordinates
        feature_coords_embedding = nn.Linear(2, D_f).to(track_tokens.device)
        Q_input = feature_coords_embedding(feature_positions)  # (T, HW, D_f)
        
        # Apply RoPE to queries if enabled
        if self.use_rope:
            if self.feature_padding is not None:
                Q_padded = self.feature_padding(Q_input)
            else:
                Q_padded = Q_input
            
            Q_rope = self.rope(Q_padded, feature_positions)
            
            if self.feature_unpadding is not None:
                Q_input = self.feature_unpadding(Q_rope)
            else:
                Q_input = Q_rope
        
        # Step 2: Prepare keys and values from track tokens
        K_input = track_tokens  # (T, M, D_f)
        V_input = track_tokens  # (T, M, D_f)
        
        # Step 3: Transpose spatial bias: (T, M, HW) -> (T, HW, M)
        spatial_bias_transposed = spatial_bias.transpose(-2, -1)  # (T, HW, M)
        
        # Step 4: Expand spatial bias for multi-head attention
        # spatial_bias_transposed: (T, HW, M) -> (T, num_heads, HW, M)
        spatial_bias_expanded = spatial_bias_transposed.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        # Step 5: Apply multi-head attention (reversed roles)
        # Q: feature map coordinates, K/V: track tokens
        splatted_features = self.attention(
            Q_input, K_input, V_input, spatial_bias_expanded, return_attention=False
        )
        
        # Step 6: Apply output projection
        output_features = self.W_out(splatted_features)  # (T, HW, D_f)
        
        return output_features


class Tracktention(nn.Module):
    """
    Tracktention mechanism for sampling features along tracks using attention.
    
    Implements the tracktention mechanism described in the paper:
    - F ∈ R^(T×HW×D_f): input feature map
    - P ∈ R^(T×M×2): set of M tracks, each a sequence of 2D points
    - T ∈ R^(T×M×D_f): track tokens after positional embedding
    """

    def __init__(self, feature_dim=768, d_k=256, sigma=1.0, use_rope=True, num_heads=8, use_qk_norm=True, 
                 use_track_transformer=True, track_transformer_heads=8, track_transformer_layers=2,
                 use_splatting=True, splatting_heads=8, use_flash_attention=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_k = d_k
        self.sigma = sigma
        self.use_rope = use_rope
        self.num_heads = num_heads
        self.use_qk_norm = use_qk_norm
        self.use_track_transformer = use_track_transformer
        self.use_splatting = use_splatting
        
        # Multi-head attention with QK-normalization for sampling
        self.attention = MultiHeadAttention(
            d_model=feature_dim,
            num_heads=num_heads,
            d_k=d_k // num_heads,  # d_k per head
            use_qk_norm=use_qk_norm,
            use_flash_attention=use_flash_attention
        )
        
        # Positional embedding for 2D coordinates
        self.pos_embedding = nn.Linear(2, feature_dim)
        
        # RoPE for keys and values from feature map
        if self.use_rope:
            # Ensure feature dimension is divisible by 4 for RoPE
            if feature_dim % 4 != 0:
                # Pad feature dimension to make it divisible by 4
                self.feature_dim_padded = ((feature_dim + 3) // 4) * 4
                self.feature_padding = nn.Linear(feature_dim, self.feature_dim_padded)
                self.feature_unpadding = nn.Linear(self.feature_dim_padded, feature_dim)
            else:
                self.feature_dim_padded = feature_dim
                self.feature_padding = None
                self.feature_unpadding = None
            
            self.rope = RoPE2D(self.feature_dim_padded, base_freq=100.0)
        
        # Track Transformer for temporal processing
        if self.use_track_transformer:
            self.track_transformer = TrackTransformer(
                d_model=feature_dim,
                num_heads=track_transformer_heads,
                num_layers=track_transformer_layers
            )
        
        # Attentional Splatting for mapping track tokens back to feature maps
        if self.use_splatting:
            self.splatting = AttentionalSplatting(
                feature_dim=feature_dim,
                d_k=d_k,
                num_heads=splatting_heads,
                use_qk_norm=use_qk_norm,
                use_rope=use_rope,
                use_flash_attention=use_flash_attention
            )

    def positional_embedding_2d(self, coords):
        """
        Apply positional embedding to 2D coordinates.
        
        Args:
            coords: (T, M, 2) tensor of 2D coordinates
            
        Returns:
            (T, M, D_f) tensor of embedded coordinates
        """
        return self.pos_embedding(coords)

    def compute_spatial_bias(self, track_positions, feature_positions):
        """
        Compute spatial bias B_t that encourages attention to align with tracks.
        
        B_{tij} = -||P_{ti} - pos_{F_t}(j)||^2 / (2σ^2)
        
        Args:
            track_positions: (T, M, 2) tensor of track positions
            feature_positions: (T, HW, 2) tensor of feature map positions
            
        Returns:
            (T, M, HW) tensor of spatial bias values
        """
        T, M, _ = track_positions.shape
        _, HW, _ = feature_positions.shape
        
        # Expand dimensions for broadcasting: (T, M, 1, 2) and (T, 1, HW, 2)
        track_pos = track_positions.unsqueeze(2)  # (T, M, 1, 2)
        feat_pos = feature_positions.unsqueeze(1)  # (T, 1, HW, 2)
        
        # Compute squared distances: (T, M, HW)
        distances_sq = torch.sum((track_pos - feat_pos) ** 2, dim=-1)
        
        # Apply bias formula
        bias = -distances_sq / (2 * self.sigma ** 2)
        
        return bias

    def convert_track_to_feature_coords(self, track_positions, image_size, feature_grid_size):
        """
        Convert track positions from pixel coordinates to feature grid coordinates.
        
        Args:
            track_positions: (T, M, 2) tensor of pixel coordinates
            image_size: (height, width) of the image in pixels
            feature_grid_size: (height, width) of the feature grid
            
        Returns:
            (T, M, 2) tensor of feature grid coordinates
        """
        T, M, _ = track_positions.shape
        device = track_positions.device
        dtype = track_positions.dtype
        
        # Convert image_size and feature_grid_size to tensors
        if isinstance(image_size, (int, float)):
            image_size = (image_size, image_size)
        if isinstance(feature_grid_size, (int, float)):
            feature_grid_size = (feature_grid_size, feature_grid_size)
            
        image_h, image_w = image_size
        feat_h, feat_w = feature_grid_size
        
        # Clamp to ensure coordinates are within image bounds
        track_positions_clamped = torch.clamp(track_positions, 0, max(image_h, image_w))
        
        # Scale directly to feature grid coordinates
        # This ensures that max pixel coordinate maps to max feature coordinate
        feature_x = track_positions_clamped[..., 0] * (feat_w - 1) / (image_w - 1)  # (T, M)
        feature_y = track_positions_clamped[..., 1] * (feat_h - 1) / (image_h - 1)  # (T, M)
        
        # Stack back to (T, M, 2)
        feature_coords = torch.stack([feature_x, feature_y], dim=-1)
        
        return feature_coords

    def get_feature_positions(self, feature_map):
        """
        Generate spatial positions for feature map tokens.
        
        Args:
            feature_map: (T, HW, D_f) tensor
            
        Returns:
            (T, HW, 2) tensor of spatial positions
        """
        T, HW, _ = feature_map.shape
        
        # Assume feature map is arranged in a grid
        # For now, we'll create a simple grid layout
        # In practice, this should match the actual spatial arrangement
        h = int(math.sqrt(HW))
        w = HW // h
        
        # Create grid coordinates
        y_coords = torch.arange(h, device=feature_map.device, dtype=feature_map.dtype)
        x_coords = torch.arange(w, device=feature_map.device, dtype=feature_map.dtype)
        
        # Create meshgrid
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Flatten and stack
        positions = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=-1)  # (HW, 2)
        
        # Expand to all time steps
        positions = positions.unsqueeze(0).expand(T, -1, -1)  # (T, HW, 2)
        
        return positions


    def sample(self, feature_map, tracks, image_size=(350, 350)):
        """
        Sample features along tracks using tracktention mechanism.
        
        Args:
            feature_map: (T, HW, D_f) input feature map
            tracks: (T, M, 2) track positions in pixel coordinates
            image_size: (height, width) of the image in pixels
            
        Returns:
            (T, M, D_f) sampled track features
        """
        T, HW, D_f = feature_map.shape
        _, M, _ = tracks.shape
        
        # Step 1: Convert track positions from pixel coordinates to feature grid coordinates
        feature_grid_size = int(math.sqrt(HW))  # Assuming square grid
        tracks_feature_coords = self.convert_track_to_feature_coords(
            tracks, image_size, feature_grid_size
        )  # (T, M, 2)
        
        # Step 2: Get spatial positions for bias computation and RoPE
        feature_positions = self.get_feature_positions(feature_map)  # (T, HW, 2)
        
        # Step 3: Apply positional embedding to converted track coordinates
        track_tokens = self.positional_embedding_2d(tracks_feature_coords)  # (T, M, D_f)
        
        # Step 4: Prepare inputs for multi-head attention
        Q_input = track_tokens  # (T, M, D_f) - track tokens as queries
        
        # Apply RoPE to feature map for keys and values
        if self.use_rope:
            # Prepare feature map for RoPE
            if self.feature_padding is not None:
                feature_map_padded = self.feature_padding(feature_map)  # (T, HW, D_f_padded)
            else:
                feature_map_padded = feature_map
            
            # Apply RoPE to feature map based on spatial positions
            feature_map_rope = self.rope(feature_map_padded, feature_positions)  # (T, HW, D_f_padded)
            
            # Unpad if necessary for K and V inputs
            if self.feature_unpadding is not None:
                K_input = self.feature_unpadding(feature_map_rope)  # (T, HW, D_f)
                V_input = K_input  # Use same for keys and values
            else:
                K_input = feature_map_rope  # (T, HW, D_f)
                V_input = K_input  # Use same for keys and values
        else:
            # Original implementation without RoPE
            K_input = feature_map  # (T, HW, D_f)
            V_input = feature_map  # (T, HW, D_f) - no projection to preserve features
        
        # Step 5: Compute spatial bias using converted coordinates
        spatial_bias = self.compute_spatial_bias(tracks_feature_coords, feature_positions)  # (T, M, HW)
        
        # Expand spatial bias for multi-head attention
        # spatial_bias: (T, M, HW) -> (T, num_heads, M, HW)
        spatial_bias_expanded = spatial_bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        # Step 6: Apply multi-head attention with QK-normalization
        sampled_features, attention_weights = self.attention(
            Q_input, K_input, V_input, spatial_bias_expanded, return_attention=True
        )
        
        # Step 7: Apply Track Transformer for temporal processing
        if self.use_track_transformer:
            sampled_features = self.track_transformer(sampled_features)
        
        # Step 8: Apply Attentional Splatting to map track tokens back to feature maps
        if self.use_splatting:
            updated_feature_map = self.splatting(
                sampled_features, feature_map, feature_positions, spatial_bias
            )
            return sampled_features, attention_weights, updated_feature_map
        else:
            return sampled_features, attention_weights, None

    def forward(self, feature_map, tracks, feature_dim=None, image_size=(350, 350)):
        """
        Forward pass of tracktention mechanism.
        
        Args:
            feature_map: (B, T, HW, D_f) or (T, HW, D_f) feature map
            tracks: (B, T, M, 2) or (T, M, 2) track positions in pixel coordinates
            feature_dim: feature dimension (for compatibility)
            image_size: (height, width) of the image in pixels
            
        Returns:
            (B, T, M, D_f) or (T, M, D_f) sampled track features
        """
        # Handle batch dimension
        if feature_map.dim() == 4:  # (B, T, HW, D_f)
            B, T, HW, D_f = feature_map.shape
            _, _, M, _ = tracks.shape
            
            # Reshape to process all batches together
            feature_map = feature_map.reshape(B * T, HW, D_f)
            tracks = tracks.reshape(B * T, M, 2)
            
            # Sample features
            result = self.sample(feature_map, tracks, image_size)
            if len(result) == 3:
                sampled_features, attention_weights, updated_feature_map = result
            else:
                sampled_features, attention_weights = result
                updated_feature_map = None
            
            # Reshape back to batch format
            sampled_features = sampled_features.reshape(B, T, M, D_f)
            # attention_weights shape: (B*T, num_heads, M, HW) -> (B, T, num_heads, M, HW)
            attention_weights = attention_weights.reshape(B, T, self.num_heads, M, HW)
            
            if updated_feature_map is not None:
                updated_feature_map = updated_feature_map.reshape(B, T, HW, D_f)
            
        else:  # (T, HW, D_f)
            result = self.sample(feature_map, tracks, image_size)
            if len(result) == 3:
                sampled_features, attention_weights, updated_feature_map = result
            else:
                sampled_features, attention_weights = result
                updated_feature_map = None
            
        if updated_feature_map is not None:
            return sampled_features, updated_feature_map
        else:
            return sampled_features 