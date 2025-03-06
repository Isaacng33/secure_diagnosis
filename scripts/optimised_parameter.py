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
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = nn.CrossEntropyLoss()(outputs, y_batch)
            loss.backward()
            optimizer.step()
    
    # Direct evaluation on test set
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test).argmax(dim=1)
        accuracy = accuracy_score(y_test.numpy(), test_preds.numpy())
    
    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    return epsilon, accuracy

# Parameter grid including 1.1 noise multiplier
param_grid = {
    'noise_multiplier': [0.5, 0.7, 1.0, 1.1],
    'batch_size': [64, 128, 256],
    'epochs': [10, 15],
    'max_grad_norm': [1.0]
}

# Track best configurations for ε=1-5
epsilon_targets = list(range(0, 6))
best_configs = {eps: {'accuracy': 0, 'params': None} for eps in epsilon_targets}

# Grid search
for nm, bs, ep, mgn in product(param_grid['noise_multiplier'],
                              param_grid['batch_size'],
                              param_grid['epochs'],
                              param_grid['max_grad_norm']):
    print(f"\nTesting: noise_mult={nm}, batch={bs}, epochs={ep}, max_grad_norm={mgn}")
    epsilon, acc = train_and_evaluate(nm, bs, ep, mgn)
    
    if 0 <= epsilon <= 5:
        rounded_eps = int(round(epsilon))
        if acc > best_configs[rounded_eps]['accuracy']:
            best_configs[rounded_eps] = {
                'accuracy': acc,
                'params': {
                    'noise_multiplier': nm,
                    'batch_size': bs,
                    'epochs': ep,
                    'max_grad_norm': mgn,
                    'learning_rate': 0.1 * (bs / 64)
                }
            }
    print(f"ε={epsilon:.2f}, Accuracy={acc:.4f}")

# Print final results
print("\n=== Best Configurations for ε=1-5 ===")
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