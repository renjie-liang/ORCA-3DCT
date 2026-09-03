"""
3D Perceiver / Spatial Pooling Module for CT-CLIP

Based on M3D's SpatialPoolingProjector, adapted for CTViT output format.
This module aggregates 3D patch tokens from CTViT into a smaller set of tokens
suitable for text alignment, while preserving 3D spatial structure.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Perceiver3D(nn.Module):
    """
    3D Perceiver module that aggregates CTViT patch tokens.
    
    Input: (B, T, H, W, D) - 3D structured tokens from CTViT
    Output: (B, T', H', W', D') - Aggregated tokens with reduced spatial dimensions
    
    The module:
    1. Optionally applies 3D spatial pooling to reduce dimensions
    2. Projects tokens through MLP/Linear layers
    3. Maintains 3D structure for better spatial understanding
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_tokens_pre: tuple,  # (T, H, W) - original token dimensions
        pooling_type: str = 'spatial',  # 'spatial' or 'none'
        pooling_size: int = 2,
        layer_type: str = 'mlp',  # 'mlp' or 'linear'
        layer_num: int = 2,
    ):
        """
        Args:
            in_dim: Input token dimension (from CTViT)
            out_dim: Output token dimension (for CLIP projection)
            num_tokens_pre: (T, H, W) tuple of original token dimensions
            pooling_type: 'spatial' for 3D avg pooling, 'none' for no pooling
            pooling_size: Pooling kernel size (applied to all 3D dimensions)
            layer_type: 'mlp' or 'linear' for projector
            layer_num: Number of projection layers
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_tokens_pre = num_tokens_pre
        self.pooling_type = pooling_type
        self.pooling_size = pooling_size
        
        # Calculate post-pooling dimensions
        if pooling_type == 'spatial':
            self.num_tokens_post = tuple(
                num // pooling_size for num in num_tokens_pre
            )
        else:
            self.num_tokens_post = num_tokens_pre
        
        # Build projector
        if layer_type == 'linear':
            modules = []
            modules.append(nn.Linear(in_dim, out_dim))
            for _ in range(1, layer_num):
                modules.append(nn.Linear(out_dim, out_dim))
            self.projector = nn.Sequential(*modules)
        elif layer_type == 'mlp':
            modules = []
            modules.append(nn.Linear(in_dim, out_dim))
            for _ in range(1, layer_num):
                modules.append(nn.GELU())
                modules.append(nn.Linear(out_dim, out_dim))
            self.projector = nn.Sequential(*modules)
        else:
            raise ValueError(f"Unknown layer_type: {layer_type}. Must be 'linear' or 'mlp'")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tokens (B, T, H, W, D) from CTViT
            
        Returns:
            Aggregated tokens (B, T', H', W', D') or (B, T', H', W', D') if pooling
        """
        B = x.shape[0]
        
        # Handle different input shapes dynamically
        # CTViT may output (B, T, H, W, D) or (B, D, T, H, W) depending on implementation
        if x.ndim == 5:
            # Check if shape is (B, D, T, H, W) or (B, T, H, W, D)
            # If second dimension matches feature dim, it's likely (B, D, T, H, W)
            if x.shape[1] == self.in_dim and x.shape[4] != self.in_dim:
                # Input is (B, D, T, H, W), convert to (B, T, H, W, D)
                x = rearrange(x, 'b d t h w -> b t h w d')
        
        # Get actual dimensions from input
        _, T_actual, H_actual, W_actual, _ = x.shape
        
        # Apply spatial pooling if enabled
        if self.pooling_type == 'spatial':
            # Rearrange to (B, D, T, H, W) for 3D pooling
            x = rearrange(x, 'b t h w d -> b d t h w')
            # Apply 3D average pooling
            x = F.avg_pool3d(x, kernel_size=self.pooling_size, stride=self.pooling_size)
            # Get actual post-pooling dimensions
            _, _, T_post_actual, H_post_actual, W_post_actual = x.shape
            # Rearrange back to (B, T', H', W', D)
            x = rearrange(x, 'b d t h w -> b t h w d', t=T_post_actual, h=H_post_actual, w=W_post_actual)
            # Use actual dimensions for final reshape
            T_final, H_final, W_final = T_post_actual, H_post_actual, W_post_actual
        else:
            # No pooling, use actual input dimensions
            T_final, H_final, W_final = T_actual, H_actual, W_actual
        
        # Flatten spatial dimensions for projection: (B, T*H*W, D)
        x = rearrange(x, 'b t h w d -> b (t h w) d')
        
        # Apply projector: (B, T*H*W, D) -> (B, T*H*W, out_dim)
        x = self.projector(x)
        
        # Reshape back to 3D: (B, T', H', W', out_dim)
        # Use actual dimensions (not initialization assumptions)
        x = rearrange(x, 'b (t h w) d -> b t h w d', 
                     t=T_final, h=H_final, w=W_final)
        
        return x
    
    @property
    def output_num_tokens(self) -> int:
        """Total number of output tokens (T' * H' * W')"""
        return self.num_tokens_post[0] * self.num_tokens_post[1] * self.num_tokens_post[2]
    
    @property
    def output_shape(self) -> tuple:
        """Output shape: (T', H', W')"""
        return self.num_tokens_post

