import numpy as np
import pandas as pd
import sys
import torch
from torch import nn
from itertools import product
from tqdm import tqdm
import json
from datetime import datetime
from main_neural_network_mlp import DatasetMLP, MLPModel, train_model, evaluate_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLPModelCustom(nn.Module):
    """MLP model with customizable architecture"""
    def __init__(self, input_size, output_size, hidden_layers, dropout_rate):
        super(MLPModelCustom, self).__init__()
        
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # First layer
        self.layers.append(nn.Linear(input_size, hidden_layers[0]))
        
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.dropouts.append(nn.Dropout(dropout_rate))
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        
        # Output layer
        self.dropouts.append(nn.Dropout(dropout_rate))
        self.layers.append(nn.Linear(hidden_layers[-1], output_size))
        
        self.relu = nn.ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.relu(layer(x))
            if i < len(self.dropouts) - 1:
                x = self.dropouts[i](x)
        
        x = self.layers[-1](x)
        return x


def train_model_custom(model, dataset, epochs, batch_size, learning_rate, optimizer_type='SGD', weight_decay=1e-4, momentum=0.9):
    """Train model with custom parameters"""
    criterion = nn.MSELoss()
    
    if optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, 
                                   nesterov=True, weight_decay=weight_decay)
    elif optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(dataset.X_train, dataset.y_train), 
        batch_size=batch_size, 
        shuffle=True
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=learning_rate, 
        steps_per_epoch=len(dataloader), 
        epochs=epochs
    )
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        
        epoch_loss /= len(dataloader)
    
    return epoch_loss


def hyperparameter_search(train_path, test_path, output_file='hyperparameter_results.json'):
    """Perform grid search over hyperparameters"""
    
    # Define hyperparameter grid
    param_grid = {
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.13, 0.15],
        'batch_size': [64, 128, 256, 512],
        'epochs': [150, 200, 250],
        'hidden_layers': [
            [128, 128, 64],
            [256, 128, 64],
            [128, 64, 32],
            [256, 256, 128],
            [512, 256, 128]
        ],
        'dropout_rate': [0.0, 0.1, 0.2, 0.3],
        'optimizer': ['SGD', 'Adam', 'AdamW'],
        'weight_decay': [1e-5, 1e-4, 1e-3]
    }
    
    # Load dataset
    print(f"Loading dataset from {train_path} and {test_path}")
    dataset = DatasetMLP(train_path, test_path)
    
    input_size = dataset.X_train.shape[1]
    output_size = dataset.y_train.shape[1]
    
    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))
    
    print(f"Total combinations to test: {len(combinations)}")
    
    results = []
    best_score = float('inf')
    best_params = None
    best_model_state = None
    
    # Progress bar for all combinations
    pbar = tqdm(total=len(combinations), desc='Hyperparameter Search')
    
    for combo in combinations:
        params = dict(zip(keys, combo))
        
        try:
            # Create model
            model = MLPModelCustom(
                input_size=input_size,
                output_size=output_size,
                hidden_layers=params['hidden_layers'],
                dropout_rate=params['dropout_rate']
            ).to(device)
            
            # Train model
            final_loss = train_model_custom(
                model=model,
                dataset=dataset,
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                learning_rate=params['learning_rate'],
                optimizer_type=params['optimizer'],
                weight_decay=params['weight_decay']
            )
            
            # Evaluate model
            mse, rmse, r2 = evaluate_model(model, dataset)
            avg_rmse = np.mean(rmse)
            avg_r2 = np.mean(r2)
            
            # Store results
            result = {
                'params': {k: (v if not isinstance(v, list) else str(v)) for k, v in params.items()},
                'final_train_loss': float(final_loss),
                'avg_rmse': float(avg_rmse),
                'avg_r2': float(avg_r2),
                'rmse_per_output': rmse.tolist(),
                'r2_per_output': r2.tolist(),
                'mse_per_output': mse.tolist()
            }
            results.append(result)
            
            # Update best model
            if avg_rmse < best_score:
                best_score = avg_rmse
                best_params = params.copy()
                best_model_state = model.state_dict().copy()
            
            # Update progress bar
            pbar.set_postfix({
                'best_rmse': f'{best_score:.6f}',
                'current_rmse': f'{avg_rmse:.6f}'
            })
            
        except Exception as e:
            print(f"\nError with params {params}: {e}")
            results.append({
                'params': {k: (v if not isinstance(v, list) else str(v)) for k, v in params.items()},
                'error': str(e)
            })
        
        pbar.update(1)
    
    pbar.close()
    
    # Save results
    output_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'train_file': train_path,
        'test_file': test_path,
        'total_combinations': len(combinations),
        'best_params': {k: (v if not isinstance(v, list) else str(v)) for k, v in best_params.items()},
        'best_rmse': float(best_score),
        'all_results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print("BEST HYPERPARAMETERS FOUND:")
    print(f"{'='*60}")
    for key, value in best_params.items():
        print(f"{key}: {value}")
    print(f"\nBest Average RMSE: {best_score:.6f}")
    print(f"\nResults saved to: {output_file}")
    
    # Save best model
    if best_model_state is not None:
        best_model_path = output_file.replace('.json', '_best_model.pth')
        torch.save(best_model_state, best_model_path)
        print(f"Best model saved to: {best_model_path}")
    
    return best_params, best_score, results


def quick_search(train_path, test_path, output_file='hyperparameter_results_quick.json'):
    """Perform a quicker, more focused hyperparameter search"""
    
    # Smaller, more focused grid
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.13],
        'batch_size': [128, 256],
        'epochs': [200],
        'hidden_layers': [
            [128, 128, 64],
            [256, 128, 64],
        ],
        'dropout_rate': [0.1, 0.2],
        'optimizer': ['SGD', 'Adam'],
        'weight_decay': [1e-4]
    }
    
    # Load dataset
    print(f"Loading dataset from {train_path} and {test_path}")
    dataset = DatasetMLP(train_path, test_path)
    
    input_size = dataset.X_train.shape[1]
    output_size = dataset.y_train.shape[1]
    
    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))
    
    print(f"Total combinations to test: {len(combinations)}")
    
    results = []
    best_score = float('inf')
    best_params = None
    best_model_state = None
    
    pbar = tqdm(total=len(combinations), desc='Quick Hyperparameter Search')
    
    for combo in combinations:
        params = dict(zip(keys, combo))
        
        try:
            model = MLPModelCustom(
                input_size=input_size,
                output_size=output_size,
                hidden_layers=params['hidden_layers'],
                dropout_rate=params['dropout_rate']
            ).to(device)
            
            final_loss = train_model_custom(
                model=model,
                dataset=dataset,
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                learning_rate=params['learning_rate'],
                optimizer_type=params['optimizer'],
                weight_decay=params['weight_decay']
            )
            
            mse, rmse, r2 = evaluate_model(model, dataset)
            avg_rmse = np.mean(rmse)
            avg_r2 = np.mean(r2)
            
            result = {
                'params': {k: (v if not isinstance(v, list) else str(v)) for k, v in params.items()},
                'final_train_loss': float(final_loss),
                'avg_rmse': float(avg_rmse),
                'avg_r2': float(avg_r2),
                'rmse_per_output': rmse.tolist(),
                'r2_per_output': r2.tolist(),
                'mse_per_output': mse.tolist()
            }
            results.append(result)
            
            if avg_rmse < best_score:
                best_score = avg_rmse
                best_params = params.copy()
                best_model_state = model.state_dict().copy()
            
            pbar.set_postfix({
                'best_rmse': f'{best_score:.6f}',
                'current_rmse': f'{avg_rmse:.6f}'
            })
            
        except Exception as e:
            print(f"\nError with params {params}: {e}")
        
        pbar.update(1)
    
    pbar.close()
    
    output_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'train_file': train_path,
        'test_file': test_path,
        'total_combinations': len(combinations),
        'best_params': {k: (v if not isinstance(v, list) else str(v)) for k, v in best_params.items()},
        'best_rmse': float(best_score),
        'all_results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print("BEST HYPERPARAMETERS FOUND:")
    print(f"{'='*60}")
    for key, value in best_params.items():
        print(f"{key}: {value}")
    print(f"\nBest Average RMSE: {best_score:.6f}")
    print(f"\nResults saved to: {output_file}")
    
    if best_model_state is not None:
        best_model_path = output_file.replace('.json', '_best_model.pth')
        torch.save(best_model_state, best_model_path)
        print(f"Best model saved to: {best_model_path}")
    
    return best_params, best_score, results


def ultra_quick_search(train_path, test_path, output_file='hyperparameter_results_ultra_quick.json'):
    """Perform a very fast hyperparameter search with minimal combinations"""
    
    # Very minimal, focused grid
    param_grid = {
        'learning_rate': [0.1, 0.13],
        'batch_size': [256],
        'epochs': [200],
        'hidden_layers': [
            [128, 128, 64],
            [256, 128, 64],
        ],
        'dropout_rate': [0.1],
        'optimizer': ['SGD'],
        'weight_decay': [1e-4]
    }
    
    # Load dataset
    print(f"Loading dataset from {train_path} and {test_path}")
    dataset = DatasetMLP(train_path, test_path)
    
    input_size = dataset.X_train.shape[1]
    output_size = dataset.y_train.shape[1]
    
    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))
    
    print(f"Total combinations to test: {len(combinations)}")
    
    results = []
    best_score = float('inf')
    best_params = None
    best_model_state = None
    
    pbar = tqdm(total=len(combinations), desc='Ultra Quick Hyperparameter Search')
    
    for combo in combinations:
        params = dict(zip(keys, combo))
        
        try:
            model = MLPModelCustom(
                input_size=input_size,
                output_size=output_size,
                hidden_layers=params['hidden_layers'],
                dropout_rate=params['dropout_rate']
            ).to(device)
            
            final_loss = train_model_custom(
                model=model,
                dataset=dataset,
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                learning_rate=params['learning_rate'],
                optimizer_type=params['optimizer'],
                weight_decay=params['weight_decay']
            )
            
            mse, rmse, r2 = evaluate_model(model, dataset)
            avg_rmse = np.mean(rmse)
            avg_r2 = np.mean(r2)
            
            result = {
                'params': {k: (v if not isinstance(v, list) else str(v)) for k, v in params.items()},
                'final_train_loss': float(final_loss),
                'avg_rmse': float(avg_rmse),
                'avg_r2': float(avg_r2),
                'rmse_per_output': rmse.tolist(),
                'r2_per_output': r2.tolist(),
                'mse_per_output': mse.tolist()
            }
            results.append(result)
            
            if avg_rmse < best_score:
                best_score = avg_rmse
                best_params = params.copy()
                best_model_state = model.state_dict().copy()
            
            pbar.set_postfix({
                'best_rmse': f'{best_score:.6f}',
                'current_rmse': f'{avg_rmse:.6f}'
            })
            
        except Exception as e:
            print(f"\nError with params {params}: {e}")
        
        pbar.update(1)
    
    pbar.close()
    
    output_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'train_file': train_path,
        'test_file': test_path,
        'total_combinations': len(combinations),
        'best_params': {k: (v if not isinstance(v, list) else str(v)) for k, v in best_params.items()},
        'best_rmse': float(best_score),
        'all_results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print("BEST HYPERPARAMETERS FOUND:")
    print(f"{'='*60}")
    for key, value in best_params.items():
        print(f"{key}: {value}")
    print(f"\nBest Average RMSE: {best_score:.6f}")
    print(f"\nResults saved to: {output_file}")
    
    if best_model_state is not None:
        best_model_path = output_file.replace('.json', '_best_model.pth')
        torch.save(best_model_state, best_model_path)
        print(f"Best model saved to: {best_model_path}")
    
    return best_params, best_score, results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python hyperparameter_search.py <train_file> <test_file> [mode]")
        print("  mode: 'full' for complete search, 'quick' for faster search, 'ultra' for ultra fast (default: ultra)")
        sys.exit(1)
    
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else 'ultra'
    
    if mode == 'full':
        print("Running FULL hyperparameter search (this will take a long time)...")
        best_params, best_score, results = hyperparameter_search(train_file, test_file)
    elif mode == 'quick':
        print("Running QUICK hyperparameter search...")
        best_params, best_score, results = quick_search(train_file, test_file)
    else:
        print("Running ULTRA QUICK hyperparameter search...")
        best_params, best_score, results = ultra_quick_search(train_file, test_file)
