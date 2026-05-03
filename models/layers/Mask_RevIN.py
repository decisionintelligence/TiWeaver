import torch
import torch.nn as nn

class MaskedRevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False, 
                 min_valid_points=3, global_fallback=True):
        """
        Masked Reversible Instance Normalization (RevIN) with support for missing values
        
        :param num_features: number of features/channels
        :param eps: small value for numerical stability
        :param affine: if True, use learnable affine parameters
        :param subtract_last: if True, subtract last value instead of mean
        :param min_valid_points: minimum valid points required for statistics calculation
        :param global_fallback: use global fallback statistics when insufficient valid points
        """
        super(MaskedRevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.min_valid_points = min_valid_points
        self.global_fallback = global_fallback
        
        # Initialize global statistics as buffers
        self.register_buffer('global_mean', torch.zeros(num_features))
        self.register_buffer('global_stdev', torch.ones(num_features))
        self.register_buffer('global_count', torch.zeros(1))
        
        if self.affine:
            self._init_params()

    def _init_params(self):
        """Initialize affine parameters"""
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mask, mode: str):
        """
        Forward pass for Masked RevIN
        
        :param x: input tensor of shape (batch_size, seq_len, num_features)
        :param mask: binary mask tensor of same shape as x (1=valid, 0=invalid/missing)
        :param mode: 'norm' for normalization, 'denorm' for denormalization
        """
        if mode == "norm":
            # Get statistics using mask
            self._get_statistics(x, mask)
            x = self._normalize(x, mask)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError(f"Unknown mode: {mode}")
        return x

    def _get_statistics(self, x, mask):
        """
        Compute masked statistics (mean and stdev) for each feature
        
        :param x: input tensor (batch_size, seq_len, num_features)
        :param mask: binary mask tensor (batch_size, seq_len, num_features)
        """
        # Ensure mask has the same shape as x
        if mask.shape != x.shape:
            raise ValueError(f"Mask shape {mask.shape} must match input shape {x.shape}")
        
        # For subtract_last mode, store the last valid values
        if self.subtract_last:
            # Find last valid index for each feature
            batch_size, seq_len, num_features = x.shape
            device = x.device
            
            # Create index tensor
            indices = torch.arange(seq_len, device=device).repeat(batch_size, num_features, 1).permute(0, 2, 1)
            
            # Apply mask to indices (set invalid to -1)
            masked_indices = torch.where(mask, indices, -1)
            
            # Find last valid index for each feature in each batch
            last_indices = masked_indices.max(dim=1)[0]
            
            # Gather last valid values
            self.last = torch.zeros(batch_size, num_features, device=device)
            for b in range(batch_size):
                for f in range(num_features):
                    idx = last_indices[b, f].item()
                    if idx >= 0:  # Valid index found
                        self.last[b, f] = x[b, idx, f]
                    else:
                        # If no valid points, use zero (will be handled in normalization)
                        self.last[b, f] = torch.tensor(0.0, device=device)
            self.last = self.last.unsqueeze(1)  # Shape (batch_size, 1, num_features)
            return
        
        # Fully vectorized computation for masked mean and standard deviation
        batch_size, seq_len, num_features = x.shape
        
        # Count valid points per batch and feature: (batch_size, num_features)
        n_valid_per_batch_feature = mask.sum(dim=1)  # (batch_size, num_features)
        
        # Compute sum and sum of squares: (batch_size, num_features)
        sum_x = (x * mask).sum(dim=1)  # (batch_size, num_features)
        sum_x2 = (x * x * mask).sum(dim=1)  # (batch_size, num_features)
        
        # Compute mean and variance: (batch_size, num_features)
        mean_vals = sum_x / (n_valid_per_batch_feature + self.eps)
        var_vals = (sum_x2 / (n_valid_per_batch_feature + self.eps)) - (mean_vals * mean_vals)
        std_vals = torch.sqrt(torch.clamp(var_vals, min=0) + self.eps)
        
        # Check which features have sufficient valid points: (batch_size, num_features)
        valid_features = n_valid_per_batch_feature >= self.min_valid_points
        
        # Initialize output tensors
        self.mean = torch.zeros(batch_size, num_features, device=x.device)
        self.stdev = torch.ones(batch_size, num_features, device=x.device)
        
        # Assign computed statistics for valid features
        self.mean[valid_features] = mean_vals[valid_features]
        self.stdev[valid_features] = std_vals[valid_features]
        
        '''
        # Handle features with insufficient valid points
        invalid_features = ~valid_features
        if invalid_features.any():
            if self.global_fallback:
                # Use global statistics for invalid features
                self.mean[invalid_features] = self.global_mean.repeat(batch_size, 1)[invalid_features]
                self.stdev[invalid_features] = self.global_stdev.repeat(batch_size, 1)[invalid_features]
            else:
                # Use zero mean and unit std for insufficient points
                self.mean[invalid_features] = torch.zeros_like(self.mean[invalid_features])
                self.stdev[invalid_features] = torch.ones_like(self.stdev[invalid_features])
        
        
        # Update global statistics for valid features during training
        if self.global_fallback and self.training:
            # Get valid features that need global stats update
            valid_for_global = valid_features & (n_valid_per_batch_feature > 0)
            if valid_for_global.any():
                # Get indices of valid features
                valid_indices = torch.where(valid_for_global)
                batch_indices, feature_indices = valid_indices
                
                # Update global statistics for each valid feature
                for i in range(len(batch_indices)):
                    b_idx = batch_indices[i].item()
                    f_idx = feature_indices[i].item()
                    n_valid = n_valid_per_batch_feature[b_idx, f_idx].item()
                    self._update_global_stats(mean_vals[b_idx, f_idx], std_vals[b_idx, f_idx], n_valid, f_idx)
        '''
        
        # Reshape for broadcasting
        self.mean = self.mean.view(batch_size, 1, num_features)
        self.stdev = self.stdev.view(batch_size, 1, num_features)

    def _update_global_stats(self, mean_val, std_val, n_valid, feature_idx):
        """Update global statistics with exponential moving average"""
        alpha = 0.05  # Learning rate for global stats
        
        if self.global_count[0] == 0:
            # First update
            self.global_mean[feature_idx] = mean_val
            self.global_stdev[feature_idx] = std_val
            self.global_count[0] = n_valid
        else:
            # Exponential moving average
            weight = n_valid / (n_valid + self.global_count[0])
            self.global_mean[feature_idx] = (1 - alpha) * self.global_mean[feature_idx] + alpha * mean_val
            self.global_stdev[feature_idx] = (1 - alpha) * self.global_stdev[feature_idx] + alpha * std_val
            self.global_count[0] += n_valid

    def _normalize(self, x, mask):
        """Normalize input using masked statistics"""
        if self.subtract_last:
            # Create a mask for valid subtraction
            valid_sub = mask & (self.last != 0)  # Only subtract where last is valid
            x_norm = torch.where(valid_sub, x - self.last, x)
        else:
            # Only subtract mean from valid points
            x_norm = torch.where(mask.bool(), x - self.mean, x)
        
        # Only divide by stdev for valid points
        x_norm = torch.where(mask.bool(), x_norm / self.stdev, x_norm)
        
        if self.affine:
            # Apply affine transformation only to valid points
            x_norm = torch.where(mask.bool(), x_norm * self.affine_weight + self.affine_bias, x_norm)
        
        return x_norm

    def _denormalize(self, x):
        """Denormalize input using stored statistics"""
        if self.affine:
            # Reverse affine transformation
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        
        if self.subtract_last:
            x = x * self.stdev + self.last
        else:
            x = x * self.stdev + self.mean
        
        return x