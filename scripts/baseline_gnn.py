#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Santhosh Shyamsundar, Prabhu S., and Studio Tyto
# SPDX-License-Identifier: MIT
"""
Simple GNN Baseline - No PyTorch Geometric Required
Basic Graph Neural Network using standard PyTorch operations

This module implements a lightweight Graph Neural Network using only standard PyTorch
operations, avoiding the dependency on PyTorch Geometric. The GNN models material
composition as graphs where nodes represent material components and edges represent
relationships between them.

Architecture:
- SimpleGNNLayer: Basic message-passing layer with learnable weights
- SimpleGNN: Multi-layer GNN with global pooling and MLP readout
- Training follows cross-dataset evaluation protocol (train on D1, test on D1-D4)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
import os
import time
from pathlib import Path

warnings.filterwarnings('ignore')

# --- PATHS ---
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / 'data'

DATASETS = {
    'D1 (UCI)': str(DATA_DIR / 'dataset_D1.csv'),
    'D2 (NDT)': str(DATA_DIR / 'dataset_D2.csv'),
    'D3 (SON)': str(DATA_DIR / 'dataset_D3.csv'),
    'D4 (RH)':  str(DATA_DIR / 'dataset_D4.csv'),
}

COMPONENT_MAPPING = {
    'cement': 0,
    'slag': 1,
    'fly_ash': 2,
    'water': 3,
    'superplasticizer': 4,
    'coarse_agg': 5,
    'fine_agg': 6,
}

class SimpleGNNLayer(nn.Module):
    """Simple Graph Neural Network Layer"""
    def __init__(self, in_features, out_features):
        super(SimpleGNNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()

    def forward(self, x, adj):
        """x: node features [batch_size, num_nodes, features]
           adj: adjacency matrix [batch_size, num_nodes, num_nodes]"""
        # Message passing: aggregate neighbor features
        messages = torch.bmm(adj, x)  # [batch, nodes, features]

        # Update node features
        updated = self.linear(messages + x)  # Residual connection
        return self.activation(updated)

class SimpleGNN(nn.Module):
    """Simple Graph Neural Network for material composition"""
    def __init__(self, node_features=3, hidden_dim=32, output_dim=1):
        super(SimpleGNN, self).__init__()
        self.conv1 = SimpleGNNLayer(node_features, hidden_dim)
        self.conv2 = SimpleGNNLayer(hidden_dim, hidden_dim)

        # Global pooling and prediction
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, node_features, adj_matrix):
        """
        node_features: [batch_size, num_nodes, node_features]
        adj_matrix: [batch_size, num_nodes, num_nodes]
        """
        batch_size, num_nodes, _ = node_features.shape

        # Graph convolutions
        x = self.conv1(node_features, adj_matrix)
        x = self.conv2(x, adj_matrix)

        # Global pooling across nodes
        x = x.transpose(1, 2)  # [batch, features, nodes]
        x = self.global_pool(x)  # [batch, features, 1]
        x = x.squeeze(2)  # [batch, features]

        # Final prediction
        return self.fc(x)

def create_simple_graph(row, dataset_id='D1'):
    """Create simple graph representation without PyG"""
    component_props = {
        'cement': [1.0, 0.95],        # [quantity_scale, reactivity]
        'slag': [0.8, 0.85],
        'fly_ash': [0.7, 0.75],
        'water': [0.0, 1.0],
        'superplasticizer': [0.0, 0.1],
        'coarse_agg': [0.0, 0.0],
        'fine_agg': [0.0, 0.0],
    }

    nodes = []
    node_features = []

    # Create nodes for each component
    for comp, idx in COMPONENT_MAPPING.items():
        quantity = row.get(comp, 0) or 0
        if quantity > 0:
            quantity_scale, reactivity = component_props[comp]
            features = [quantity * quantity_scale, idx/6.0, reactivity]
            node_features.append(features)
            nodes.append(idx)

    if not node_features:
        # Return empty graph if no components
        return None, None

    node_features = torch.tensor(node_features, dtype=torch.float)
    num_nodes = len(nodes)

    # Create simple adjacency matrix (fully connected for simplicity)
    adj_matrix = torch.ones(num_nodes, num_nodes, dtype=torch.float)

    # Add self-loops
    adj_matrix.fill_diagonal_(1.0)

    # Normalize adjacency matrix
    degree = torch.sum(adj_matrix, dim=1, keepdim=True)
    adj_matrix = adj_matrix / degree

    return node_features, adj_matrix

def collate_graph_batch(batch):
    """Collate graphs into batch tensors with proper handling"""
    if not batch:
        return None, None, None

    # Filter out None graphs and collect valid ones
    valid_batch = []
    valid_targets = []

    for item in batch:
        if item[0] is not None:  # item is (graph_data, target)
            valid_batch.append(item[0])  # graph_data is (node_features, adj_matrix)
            valid_targets.append(item[1])  # target

    if not valid_batch:
        return None, None, None

    # Process one graph at a time (simpler approach)
    # For each graph, we'll process it individually and collect results
    batch_results = []

    for (node_features, adj_matrix), target in zip(valid_batch, valid_targets):
        # Ensure tensors are on CPU and have correct types
        node_features = node_features.float()
        adj_matrix = adj_matrix.float()
        target = target.float()

        batch_results.append(((node_features, adj_matrix), target))

    # Return as list of individual graphs (will be handled in training loop)
    return batch_results

class GraphDataset(torch.utils.data.Dataset):
    """Dataset for graph data"""
    def __init__(self, df, dataset_id='D1'):
        self.df = df
        self.dataset_id = dataset_id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        node_features, adj_matrix = create_simple_graph(row, self.dataset_id)
        target = torch.tensor(row['strength'], dtype=torch.float)
        return (node_features, adj_matrix), target

def train_simple_gnn(model, train_loader, val_loader=None, epochs=200, device='cpu'):
    """Train the simple GNN with individual graph processing"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    patience = 40
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_count = 0

        for batch_items in train_loader:
            if batch_items is None:
                continue

            batch_loss = 0
            batch_count = 0

            for (node_features, adj_matrix), target in batch_items:
                node_features = node_features.to(device)
                adj_matrix = adj_matrix.to(device)
                target = target.to(device)

                optimizer.zero_grad()

                # Forward pass
                prediction = model(node_features.unsqueeze(0), adj_matrix.unsqueeze(0))
                loss = criterion(prediction.squeeze(), target)

                # Backward pass
                loss.backward()
                optimizer.step()

                batch_loss += loss.item()
                batch_count += 1

            if batch_count > 0:
                train_loss += batch_loss / batch_count
                train_count += 1

        if train_count > 0:
            train_loss /= train_count

        # Validation
        if val_loader and epoch % 5 == 0:
            model.eval()
            val_loss = 0
            val_count = 0

            with torch.no_grad():
                for batch_items in val_loader:
                    if batch_items is None:
                        continue

                    batch_val_loss = 0
                    batch_val_count = 0

                    for (node_features, adj_matrix), target in batch_items:
                        node_features = node_features.to(device)
                        adj_matrix = adj_matrix.to(device)
                        target = target.to(device)

                        prediction = model(node_features.unsqueeze(0), adj_matrix.unsqueeze(0))
                        loss = criterion(prediction.squeeze(), target)

                        batch_val_loss += loss.item()
                        batch_val_count += 1

                    if batch_val_count > 0:
                        val_loss += batch_val_loss / batch_val_count
                        val_count += 1

            if val_count > 0:
                val_loss /= val_count

                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), 'best_simple_gnn.pth')
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

    # Load best model
    if val_loader and os.path.exists('best_simple_gnn.pth'):
        model.load_state_dict(torch.load('best_simple_gnn.pth'))
        os.remove('best_simple_gnn.pth')

    return model

def predict_simple_gnn(model, loader, device='cpu'):
    """Make predictions with trained GNN"""
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch_items in loader:
            if batch_items is None:
                continue

            for (node_features, adj_matrix), target in batch_items:
                node_features = node_features.to(device)
                adj_matrix = adj_matrix.to(device)

                pred = model(node_features.unsqueeze(0), adj_matrix.unsqueeze(0))
                predictions.append(pred.cpu().numpy().flatten()[0])
                targets.append(target.cpu().numpy().flatten()[0])

    return np.array(predictions), np.array(targets)

def thermodynamic_admissibility(y_pred, X=None):
    """Same admissibility check"""
    accepted = 0
    total = len(y_pred)
    for val in y_pred:
        if val >= 0 and val <= 130:
            accepted += 1
    return (accepted / total) * 100.0

def run_simple_gnn_baseline():
    """Run simple GNN baseline on all datasets"""
    print("=" * 60)
    print("SIMPLE GNN BASELINE - Standard PyTorch Implementation")
    print("No PyTorch Geometric required")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    results = {}

    # Train on D1, evaluate on all
    if not os.path.exists(DATASETS['D1 (UCI)']):
        print("ERROR: D1 dataset not found!")
        return {}

    print("\n--- Loading and Processing Training Data (D1) ---")
    df_train_full = pd.read_csv(DATASETS['D1 (UCI)'])

    # Split for training/validation
    train_data, val_data = train_test_split(df_train_full, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = GraphDataset(train_data, 'D1')
    val_dataset = GraphDataset(val_data, 'D1')

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True, collate_fn=collate_graph_batch
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=16, shuffle=False, collate_fn=collate_graph_batch
    )

    print(f"Training graphs: {len(train_dataset)}")
    print(f"Validation graphs: {len(val_dataset)}")

    # Train GNN
    print("\n--- Training Simple GNN ---")
    model = SimpleGNN(node_features=3, hidden_dim=32)  # 3 features per node
    start_time = time.time()
    model = train_simple_gnn(model, train_loader, val_loader, epochs=200, device=device)
    training_time = time.time() - start_time
    print(".2f")

    # Evaluate on all datasets
    print("\n--- Cross-Dataset Evaluation ---")
    print(f"{'Dataset':<10} {'N':>6} {'GNN MAE':>10} {'Adm %':>8}")
    print("-" * 40)

    for d_name, path in DATASETS.items():
        if not os.path.exists(path):
            continue

        df_test = pd.read_csv(path)
        dataset_id = d_name.split(' ')[0]  # D1, D2, D3, D4

        # Create test dataset
        test_dataset = GraphDataset(df_test, dataset_id)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=16, shuffle=False, collate_fn=collate_graph_batch
        )

        # Predict
        y_pred, y_true = predict_simple_gnn(model, test_loader, device=device)

        if len(y_pred) == 0:
            print(f"    {d_name}: No valid graphs")
            continue

        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        adm = thermodynamic_admissibility(y_pred)

        results[dataset_id] = {
            'MAE': mae,
            'Admissibility': adm,
            'N': len(y_pred)
        }

        print("6")

    # Save results
    import json
    output_path = REPO_ROOT / 'results/canonical/raw/simple_gnn_baseline.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"Training time: {training_time:.2f} seconds")

    print("\n--- Simple GNN Performance Summary ---")
    print("Trained on D1 graph representations")
    print("Simple message-passing GNN without external dependencies")

    for did, result in results.items():
        print(".2f")

    return results

if __name__ == "__main__":
    results = run_simple_gnn_baseline()