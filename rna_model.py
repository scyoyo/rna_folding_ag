
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(SinusoidalPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class RibonanzaBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(RibonanzaBlock, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # x: (B, L, D)
        # mask: (B, L) bool
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x

class GeometryHead(nn.Module):
    """Predicts torsion angles (sin, cos) for reconstruction."""
    def __init__(self, d_model):
        super(GeometryHead, self).__init__()
        # Predicting 7 torsion angles: alpha, beta, gamma, delta, epsilon, zeta, chi
        # Output is 7 * 2 (sin, cos) = 14 values per residue
        self.proj = nn.Linear(d_model, 14) 
        
    def forward(self, x):
        return self.proj(x)

class DistogramHead(nn.Module):
    """Predicts distance matrix (C1'-C1') for auxiliary loss."""
    def __init__(self, d_model):
        super(DistogramHead, self).__init__()
        self.proj_i = nn.Linear(d_model, d_model // 2)
        self.proj_j = nn.Linear(d_model, d_model // 2)
        self.out = nn.Linear(d_model // 2, 1) # Simple continuous distance prediction for regression
        
    def forward(self, x):
        # x: B, L, D
        # Create pair representations
        B, L, D = x.shape
        x_i = self.proj_i(x).unsqueeze(2).expand(B, L, L, D // 2)
        x_j = self.proj_j(x).unsqueeze(1).expand(B, L, L, D // 2)
        
        pair_rep = (x_i + x_j) / 2 
        
        return F.relu(self.out(pair_rep)).squeeze(-1) # (B, L, L)

def nerf_build(torsions):
    """
    Reconstructs coordinates from internal coordinates (torsions).
    Simplified NeRF wrapper for RNA backbone.
    
    Args:
        torsions: (B, L, 7) - Torsion angles in radians
    
    Returns:
        coords: (B, L, 3) - Approximate backbone coordinates (P, C4', N).
    """
    # A REAL implementation requires complex trigonometry.
    # For this baseline, we return a differentiable "random walk" based on angles
    # to ensure the computation graph is connected for training.
    
    # Simulate a chain growth based on torsion angles
    # steps (B, L, 3)
    steps = torch.stack([
        torch.cos(torsions[:, :, 0]), 
        torch.sin(torsions[:, :, 0]), 
        torch.sin(torsions[:, :, 1])
    ], dim=-1)
    
    coords = torch.cumsum(steps, dim=1) 
    return coords

class RNAModel(nn.Module):
    def __init__(self, d_model=128, n_layers=4, n_heads=4, vocab_size=5):
        super(RNAModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model) # A, C, G, U, Pad
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        
        self.blocks = nn.ModuleList([
            RibonanzaBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        
        self.geometry = GeometryHead(d_model)
        self.distogram = DistogramHead(d_model)
        
    def forward(self, seq, mask=None):
        # seq: (B, L)
        x = self.embedding(seq)
        x = self.pos_enc(x)
        
        # Apply Transformer blocks
        # mask needs to be inverted for MultiheadAttention (True=ignore) if passing key_padding_mask
        padding_mask = (mask == 0) if mask is not None else None

        for block in self.blocks:
            x = block(x, mask=padding_mask)
            
        # Heads
        # Torsions: (B, L, 14) -> reshaping to (B, L, 7, 2)
        torsion_sc = self.geometry(x).view(x.shape[0], x.shape[1], 7, 2)
        
        # Distances: (B, L, L)
        pred_dists = self.distogram(x)
        
        # Convert sin/cos to angles (atan2)
        torsion_angles = torch.atan2(torsion_sc[:, :, :, 0], torsion_sc[:, :, :, 1])
        
        # Structure NeRF (differentiable placeholder)
        coords = nerf_build(torsion_angles)
        
        return {
            'pred_dists': pred_dists,
            'torsion_angles': torsion_angles,
            'coords': coords
        }
