import torch
import torch.nn as nn
import math

class VariableAttentionDecoder(nn.Module):
    def __init__(self, 
                 patch_size, 
                 patch_num, 
                 d_model, 
                 n_heads, 
                 d_ff=2048, 
                 dropout=0.1,
                 num_layers=1):
        super().__init__()
        
        self.patch_size = patch_size
        self.patch_num = patch_num
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Create decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        
        # Create transformer decoder
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Cross attention QKV projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, memory=None):
        """
        Args:
            x: Input tensor of shape [batch_size, num_variables, patch_num, patch_size]
            memory: Memory tensor for cross-attention, if None, will use x as memory
        """
        batch_size, num_variables, patch_num, patch_size = x.shape
        
        # Reshape for transformer
        x = x.view(batch_size, num_variables, patch_num, -1)  # [batch, num_vars, patch_num, d_model]
        
        # Process each variable
        outputs = []
        for var_idx in range(num_variables):
            # Get current variable's patches
            current_var = x[:, var_idx]  # [batch, patch_num, d_model]
            
            # Add positional encoding
            current_var = self.pos_encoder(current_var)
            
            # Prepare memory (other variables' patches)
            if memory is None:
                # Use other variables as memory
                memory_patches = []
                for other_idx in range(num_variables):
                    if other_idx != var_idx:
                        memory_patches.append(x[:, other_idx])
                if memory_patches:
                    memory = torch.cat(memory_patches, dim=1)  # [batch, (num_vars-1)*patch_num, d_model]
                    memory = self.pos_encoder(memory)
                else:
                    memory = current_var
            
            # Project QKV for cross attention
            # Query comes from current variable
            q = self.q_proj(current_var)  # [batch, patch_num, d_model]
            
            # Key and Value come from memory (other variables)
            k = self.k_proj(memory)  # [batch, (num_vars-1)*patch_num, d_model]
            v = self.v_proj(memory)  # [batch, (num_vars-1)*patch_num, d_model]
            
            # Create target mask for causal attention
            tgt_mask = self.generate_square_subsequent_mask(patch_num).to(x.device)
            
            # Apply transformer decoder with custom QKV
            output = self.transformer_decoder(
                tgt=q,  # Use projected query
                memory=memory,  # Original memory for internal cross-attention
                tgt_mask=tgt_mask
            )
            
            outputs.append(output)
        
        # Stack all variable outputs
        return torch.stack(outputs, dim=1)  # [batch, num_vars, patch_num, d_model]
    
    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

# Example usage
if __name__ == "__main__":
    # Example parameters
    batch_size = 32
    num_variables = 3
    patch_num = 4
    patch_size = 16
    d_model = 64
    n_heads = 4
    
    # Create random input
    x = torch.randn(batch_size, num_variables, patch_num, patch_size)
    
    # Initialize decoder
    decoder = VariableAttentionDecoder(
        patch_size=patch_size,
        patch_num=patch_num,
        d_model=d_model,
        n_heads=n_heads
    )
    
    # Forward pass
    output = decoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}") 