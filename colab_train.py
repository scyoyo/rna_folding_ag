
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time

class RNADataset(Dataset):
    """
    Mock dataset for RNA 3D Folding.
    In a real scenario, this would load .tfrecord or .parquet files from the competition dataset.
    """
    def __init__(self, n_samples=100, seq_len=50):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.data = self._generate_mock_data()
        
    def _generate_mock_data(self):
        data = []
        for _ in range(self.n_samples):
            # Random sequence (0-3)
            seq = torch.randint(0, 4, (self.seq_len,))
            # Mock true distances (L, L)
            true_dists = torch.rand(self.seq_len, self.seq_len) * 20.0
            # Mock true torsions (L, 7)
            true_torsions = torch.rand(self.seq_len, 7) * 2 * np.pi - np.pi
            data.append({
                'seq': seq,
                'true_dists': true_dists,
                'true_torsions': true_torsions
            })
        return data
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[idx]

def loss_fn(pred_out, batch):
    """
    Combined loss for Distance and Torsion.
    pred_out: dict keys ['pred_dists', 'torsion_angles', 'coords']
    batch: dict keys ['true_dists', 'true_torsions']
    """
    true_dists = batch['true_dists'].to(pred_out['pred_dists'].device)
    true_torsions = batch['true_torsions'].to(pred_out['torsion_angles'].device)
    
    # Distance Loss (MSE)
    dist_loss = nn.MSELoss()(pred_out['pred_dists'], true_dists)
    
    # Torsion Loss (MSE on angles - beware of periodicity, using cos distance is better but MSE for now)
    # A better loss would be 1 - cos(pred - true)
    torsion_diff = pred_out['torsion_angles'] - true_torsions
    torsion_loss = torch.mean(1 - torch.cos(torsion_diff))
    
    return dist_loss + torsion_loss

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in loader:
        seq = batch['seq'].to(device)
        
        optimizer.zero_grad()
        output = model(seq)
        loss = loss_fn(output, batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            seq = batch['seq'].to(device)
            output = model(seq)
            loss = loss_fn(output, batch)
            total_loss += loss.item()
    return total_loss / len(loader)

def main_train_loop(model, epochs=5, batch_size=4, device='cpu'):
    print(f"Starting training on {device}...")
    dataset = RNADataset(n_samples=50, seq_len=30)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    start_time = time.time()
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, loader, optimizer, device)
        val_loss = validate(model, loader, device)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
    print(f"Training finished in {time.time() - start_time:.2f}s")
    
    # Return a sample prediction for inspection
    model.eval()
    sample_seq = dataset[0]['seq'].unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(sample_seq)
        
    return pred
