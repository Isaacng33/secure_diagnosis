import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np
from itertools import product
import os
import datetime
import json

# Model definition
class LogisticRegression(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.linear(x)

# Data loading and preprocessing
df = pd.read_csv('./data/raw/dataset_1/newdataset.csv')
X = df.drop('diseases', axis=1)
y = df['diseases']
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split (no validation set)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Convert to tensors
X_train = torch.from_numpy(X_train.to_numpy().astype(np.float32))
X_test = torch.from_numpy(X_test.to_numpy().astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.int64))
y_test = torch.from_numpy(y_test.astype(np.int64))

train_dataset = TensorDataset(X_train, y_train)

# Create results directory if it doesn't exist
results_dir = './results'
os.makedirs(results_dir, exist_ok=True)

# Generate a timestamp for the output filename
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_filename = os.path.join(results_dir, f'dp_grid_search_results_{timestamp}.csv')
best_configs_filename = os.path.join(results_dir, f'dp_best_configs_{timestamp}.json')

# List to store all results
all_results = []

def train_and_evaluate(noise_multiplier, batch_size, epochs, max_grad_norm):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Learning rate scaling rule: LR = 0.1 * (batch_size / 64)
    learning_rate = 0.1 * (batch_size / 64)
    
    model = LogisticRegression(X_train.shape[1], len(le.classes_))
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    privacy_engine = PrivacyEngine()
    model, optimizer, _ = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm
    )
    
    # Training loop
    train_losses = []
    for epoch in range(epochs):
        epoch_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = nn.CrossEntropyLoss()(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(epoch_loss)
        
        # Optionally print progress to monitor training
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    
    # Evaluation on test set
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test).argmax(dim=1)
        accuracy = accuracy_score(y_test.numpy(), test_preds.numpy())
    
    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    
    return epsilon, accuracy, train_losses

# Parameter grid including 1.1 noise multiplier
param_grid = {
    'noise_multiplier': [0.5, 0.7, 1.0, 1.1],
    'batch_size': [64, 128, 256],
    'epochs': [10, 15],
    'max_grad_norm': [1.0]
}

# Track best configurations for ε=0-5
epsilon_targets = list(range(0, 6))
best_configs = {eps: {'accuracy': 0, 'params': None} for eps in epsilon_targets}

# Running counter for tracking progress
total_combinations = (len(param_grid['noise_multiplier']) * 
                      len(param_grid['batch_size']) * 
                      len(param_grid['epochs']) * 
                      len(param_grid['max_grad_norm']))
current_combination = 0

print(f"Starting grid search with {total_combinations} parameter combinations")

# Grid search
try:
    for nm, bs, ep, mgn in product(param_grid['noise_multiplier'],
                                  param_grid['batch_size'],
                                  param_grid['epochs'],
                                  param_grid['max_grad_norm']):
        current_combination += 1
        print(f"\n[{current_combination}/{total_combinations}] Testing: noise_mult={nm}, batch={bs}, epochs={ep}, max_grad_norm={mgn}")
        
        try:
            epsilon, acc, train_losses = train_and_evaluate(nm, bs, ep, mgn)
            
            # Store all information for this run
            result = {
                'noise_multiplier': nm,
                'batch_size': bs,
                'epochs': ep,
                'max_grad_norm': mgn,
                'learning_rate': 0.1 * (bs / 64),
                'epsilon': epsilon,
                'accuracy': acc,
                'final_loss': train_losses[-1] if train_losses else None,
                'loss_trajectory': train_losses
            }
            
            all_results.append(result)
            
            # Save results to file after each iteration in case of crash
            results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'loss_trajectory'} 
                                      for r in all_results])
            results_df.to_csv(results_filename, index=False)
            
            # Update best configurations
            if 0 <= epsilon <= 5:
                rounded_eps = int(round(epsilon))
                if acc > best_configs[rounded_eps]['accuracy']:
                    best_configs[rounded_eps] = {
                        'accuracy': float(acc),  # Convert numpy types to Python native for JSON
                        'params': {
                            'noise_multiplier': float(nm),
                            'batch_size': int(bs),
                            'epochs': int(ep),
                            'max_grad_norm': float(mgn),
                            'learning_rate': float(0.1 * (bs / 64))
                        }
                    }
                    
                    # Also save best configs after each update
                    with open(best_configs_filename, 'w') as f:
                        json.dump(best_configs, f, indent=2)
            
            print(f"ε={epsilon:.2f}, Accuracy={acc:.4f}")
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            # Add failed run to results
            all_results.append({
                'noise_multiplier': nm,
                'batch_size': bs,
                'epochs': ep,
                'max_grad_norm': mgn,
                'learning_rate': 0.1 * (bs / 64),
                'epsilon': None,
                'accuracy': None,
                'final_loss': None,
                'error': str(e)
            })
            
except KeyboardInterrupt:
    print("\nGrid search interrupted by user. Saving current results...")

# Final save to CSV
print(f"\nSaving final results to {results_filename}")
results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'loss_trajectory'} 
                          for r in all_results])
results_df.to_csv(results_filename, index=False)

# Save loss trajectories separately (too large for CSV)
loss_traj_filename = os.path.join(results_dir, f'dp_loss_trajectories_{timestamp}.json')
loss_trajectories = {
    f"run_{i}": {
        'params': {
            'noise_multiplier': float(r['noise_multiplier']),
            'batch_size': int(r['batch_size']),
            'epochs': int(r['epochs']),
            'max_grad_norm': float(r['max_grad_norm'])
        },
        'losses': r['loss_trajectory'] if 'loss_trajectory' in r and r['loss_trajectory'] else []
    }
    for i, r in enumerate(all_results)
}
with open(loss_traj_filename, 'w') as f:
    json.dump(loss_trajectories, f)

# Print final results
print("\n=== Best Configurations for ε=0-5 ===")
for eps in epsilon_targets:
    cfg = best_configs[eps]
    if cfg['params']:
        print(f"ε={eps}: Acc={cfg['accuracy']:.4f} | "
              f"Noise={cfg['params']['noise_multiplier']}, "
              f"Batch={cfg['params']['batch_size']}, "
              f"Epochs={cfg['params']['epochs']}, "
              f"GradNorm={cfg['params']['max_grad_norm']}, "
              f"LR={cfg['params']['learning_rate']:.2f}")
    else:
        print(f"ε={eps}: No valid configuration found")

print(f"\nResults saved to: {results_filename}")
print(f"Best configurations saved to: {best_configs_filename}")
print(f"Loss trajectories saved to: {loss_traj_filename}")
