import numpy as np
import pandas as pd
import torch
from torch import nn
import json
import random
from main_neural_network_mlp import DatasetMLP, MLPModel, train_model, evaluate_model


def random_hyperparameter_search(dataset, n_iterations=50, save_path='random_search_results.json'):
	"""
	Esegue una ricerca random degli iperparametri.
	
	Args:
		dataset: Il dataset da utilizzare
		n_iterations: Numero di combinazioni random da provare
		save_path: Path dove salvare i risultati
	
	Returns:
		best_params: I migliori iperparametri trovati
		best_score: Il miglior score ottenuto
	"""
	
	# Definisci gli spazi di ricerca per gli iperparametri
	param_distributions = {
		'hidden_layer_1': [64, 128, 256, 512],
		'hidden_layer_2': [32, 64, 128, 256],
		'hidden_layer_3': [16, 32, 64, 128],
		'dropout_rate_1': [0.0, 0.1, 0.2, 0.3],
		'dropout_rate_2': [0.0, 0.1, 0.2, 0.3, 0.4],
		'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2],
		'batch_size': [32, 64, 128, 256, 512],
		'epochs': [100, 150, 200, 250, 300],
		'weight_decay': [0.0, 1e-5, 1e-4, 1e-3]
	}
	
	results = []
	best_score = float('inf')
	best_params = None
	best_model_state = None
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	print(f"Starting random hyperparameter search with {n_iterations} iterations...")
	print(f"Using device: {device}\n")
	
	for i in range(n_iterations):
		# Campiona parametri random
		params = {
			key: random.choice(values) 
			for key, values in param_distributions.items()
		}
		
		print(f"\n{'='*60}")
		print(f"Iteration {i+1}/{n_iterations}")
		print(f"{'='*60}")
		print("Testing parameters:")
		for key, value in params.items():
			print(f"  {key}: {value}")
		
		try:
			# Crea modello custom con i parametri scelti
			class CustomMLPModel(nn.Module):
				def __init__(self, input_size, output_size, h1, h2, h3, d1, d2):
					super(CustomMLPModel, self).__init__()
					self.fc1 = nn.Linear(input_size, h1)
					self.dropout1 = nn.Dropout(d1)
					self.fc2 = nn.Linear(h1, h2)
					self.dropout2 = nn.Dropout(d2)
					self.fc3 = nn.Linear(h2, h3)
					self.dropout3 = nn.Dropout(d2)
					self.fc4 = nn.Linear(h3, output_size)
					self.relu = nn.ReLU()

				def forward(self, x):
					x = self.relu(self.fc1(x))
					x = self.dropout1(x)
					x = self.relu(self.fc2(x))
					x = self.dropout2(x)
					x = self.relu(self.fc3(x))
					x = self.dropout3(x)
					x = self.fc4(x)
					return x
			
			model = CustomMLPModel(
				input_size=dataset.X_train.shape[1],
				output_size=dataset.y_train.shape[1],
				h1=params['hidden_layer_1'],
				h2=params['hidden_layer_2'],
				h3=params['hidden_layer_3'],
				d1=params['dropout_rate_1'],
				d2=params['dropout_rate_2']
			).to(device)
			
			# Addestra il modello
			train_model_custom(
				model, 
				dataset, 
				epochs=params['epochs'],
				batch_size=params['batch_size'],
				learning_rate=params['learning_rate'],
				weight_decay=params['weight_decay']
			)
			
			# Valuta il modello
			mse, rmse, r2 = evaluate_model(model, dataset)
			avg_mse = float(np.mean(mse))
			avg_rmse = float(np.mean(rmse))
			avg_r2 = float(np.mean(r2))
			
			print(f"\nResults:")
			print(f"  Average MSE: {avg_mse:.6f}")
			print(f"  Average RMSE: {avg_rmse:.6f}")
			print(f"  Average R²: {avg_r2:.6f}")
			
			# Salva i risultati
			result = {
				'iteration': i + 1,
				'params': params,
				'avg_mse': avg_mse,
				'avg_rmse': avg_rmse,
				'avg_r2': avg_r2,
				'mse_per_output': mse.tolist(),
				'rmse_per_output': rmse.tolist(),
				'r2_per_output': r2.tolist()
			}
			results.append(result)
			
			# Aggiorna il miglior modello (basato su MSE)
			if avg_mse < best_score:
				best_score = avg_mse
				best_params = params.copy()
				best_model_state = model.state_dict().copy()
				print(f"\n🎯 NEW BEST MODEL FOUND! MSE: {best_score:.6f}")
			
		except Exception as e:
			print(f"\n❌ Error with parameters: {e}")
			continue
	
	# Salva tutti i risultati
	with open(save_path, 'w') as f:
		json.dump({
			'all_results': results,
			'best_params': best_params,
			'best_score': best_score
		}, f, indent=4)
	
	# Salva il miglior modello
	if best_model_state is not None:
		model_save_path = save_path.replace('.json', '_best_model.pth')
		torch.save(best_model_state, model_save_path)
		print(f"\n✅ Best model saved to: {model_save_path}")
	
	print(f"\n{'='*60}")
	print("SEARCH COMPLETED")
	print(f"{'='*60}")
	print("\nBest parameters found:")
	for key, value in best_params.items():
		print(f"  {key}: {value}")
	print(f"\nBest MSE: {best_score:.6f}")
	print(f"\nAll results saved to: {save_path}")
	
	return best_params, best_score


def train_model_custom(model, dataset, epochs=200, batch_size=256, learning_rate=0.01, weight_decay=1e-4):
	"""Versione modificata di train_model con weight_decay configurabile"""
	from torch.utils.data import DataLoader
	
	criterion = nn.MSELoss()
	optimizer = torch.optim.SGD(
		model.parameters(), 
		lr=learning_rate, 
		momentum=0.9, 
		nesterov=True, 
		weight_decay=weight_decay
	)
	
	dataloader = DataLoader(
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
	
	best_val_loss = float('inf')
	best_model_state = None
	
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
		
		# Validate on test set
		if dataset.X_test is not None:
			model.eval()
			with torch.no_grad():
				val_outputs = model(dataset.X_test)
				val_loss = criterion(val_outputs, dataset.y_test).item()
			
			# Save best model
			if val_loss < best_val_loss:
				best_val_loss = val_loss
				best_model_state = model.state_dict().copy()
	
	# Restore best model
	if best_model_state is not None:
		model.load_state_dict(best_model_state)


if __name__ == "__main__":
	import sys
	
	if len(sys.argv) < 3:
		print("Usage: python random_hyperparameter_search.py <train_file> <test_file> [n_iterations]")
		sys.exit(1)
	
	train_file = sys.argv[1]
	test_file = sys.argv[2]
	n_iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 50
	
	# Carica il dataset
	dataset = DatasetMLP(train_file, test_file)
	
	# Esegui la ricerca
	best_params, best_score = random_hyperparameter_search(
		dataset, 
		n_iterations=n_iterations,
		save_path='random_hyperparameter_results.json'
	)
