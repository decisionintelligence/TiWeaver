import torch
import torch.nn as nn

class MaskedRevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False, 
                 min_valid_points=3, global_fallback=True):

        super(MaskedRevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.min_valid_points = min_valid_points
        self.global_fallback = global_fallback
        
        self.register_buffer('global_mean', torch.zeros(num_features))
        self.register_buffer('global_stdev', torch.ones(num_features))
        self.register_buffer('global_count', torch.zeros(1))
        
        if self.affine:
            self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mask, mode: str):

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

        if mask.shape != x.shape:
            raise ValueError(f"Mask shape {mask.shape} must match input shape {x.shape}")
        
        if self.subtract_last:
            batch_size, seq_len, num_features = x.shape
            device = x.device
            
            indices = torch.arange(seq_len, device=device).repeat(batch_size, num_features, 1).permute(0, 2, 1)
            
            masked_indices = torch.where(mask, indices, -1)
            
            last_indices = masked_indices.max(dim=1)[0]
            
            self.last = torch.zeros(batch_size, num_features, device=device)
            for b in range(batch_size):
                for f in range(num_features):
                    idx = last_indices[b, f].item()
                    if idx >= 0:  # Valid index found
                        self.last[b, f] = x[b, idx, f]
                    else:
                        self.last[b, f] = torch.tensor(0.0, device=device)
            self.last = self.last.unsqueeze(1)  # Shape (batch_size, 1, num_features)
            return
        
        batch_size, seq_len, num_features = x.shape
        
        n_valid_per_batch_feature = mask.sum(dim=1)  # (batch_size, num_features)
        
        sum_x = (x * mask).sum(dim=1)  # (batch_size, num_features)
        sum_x2 = (x * x * mask).sum(dim=1)  # (batch_size, num_features)
        
        mean_vals = sum_x / (n_valid_per_batch_feature + self.eps)
        var_vals = (sum_x2 / (n_valid_per_batch_feature + self.eps)) - (mean_vals * mean_vals)
        std_vals = torch.sqrt(torch.clamp(var_vals, min=0) + self.eps)
        
        valid_features = n_valid_per_batch_feature >= self.min_valid_points
        
        self.mean = torch.zeros(batch_size, num_features, device=x.device)
        self.stdev = torch.ones(batch_size, num_features, device=x.device)
        
        self.mean[valid_features] = mean_vals[valid_features]
        self.stdev[valid_features] = std_vals[valid_features]
        
        self.mean = self.mean.view(batch_size, 1, num_features)
        self.stdev = self.stdev.view(batch_size, 1, num_features)

    def _update_global_stats(self, mean_val, std_val, n_valid, feature_idx):
        alpha = 0.05  # Learning rate for global stats
        
        if self.global_count[0] == 0:
            self.global_mean[feature_idx] = mean_val
            self.global_stdev[feature_idx] = std_val
            self.global_count[0] = n_valid
        else:
            weight = n_valid / (n_valid + self.global_count[0])
            self.global_mean[feature_idx] = (1 - alpha) * self.global_mean[feature_idx] + alpha * mean_val
            self.global_stdev[feature_idx] = (1 - alpha) * self.global_stdev[feature_idx] + alpha * std_val
            self.global_count[0] += n_valid

    def _normalize(self, x, mask):
        if self.subtract_last:
            valid_sub = mask & (self.last != 0)  # Only subtract where last is valid
            x_norm = torch.where(valid_sub, x - self.last, x)
        else:
            x_norm = torch.where(mask.bool(), x - self.mean, x)
        
        x_norm = torch.where(mask.bool(), x_norm / self.stdev, x_norm)
        
        if self.affine:
            x_norm = torch.where(mask.bool(), x_norm * self.affine_weight + self.affine_bias, x_norm)
        
        return x_norm

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        
        if self.subtract_last:
            x = x * self.stdev + self.last
        else:
            x = x * self.stdev + self.mean
        
        return x