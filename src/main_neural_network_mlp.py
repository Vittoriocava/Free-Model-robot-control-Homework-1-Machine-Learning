import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DatasetMLP:
	def __init__(self, file_path_train, file_path_test=None):
		self.file_path_train = file_path_train
		self.file_path_test = file_path_test
		self.model = None
		self.scaler_X = StandardScaler()
		self.scaler_y = StandardScaler()
		
		# Load and scale data once during initialization
		X_train_raw, y_train_raw = self.load_data_train()
		self.X_train = self.scaler_X.fit_transform(X_train_raw)
		self.y_train = self.scaler_y.fit_transform(y_train_raw)
		
		X_test_raw, y_test_raw = self.load_data_test()
		if X_test_raw is not None:
			self.X_test = self.scaler_X.transform(X_test_raw)
			self.y_test = self.scaler_y.transform(y_test_raw)  # Scale for predictions
			self.X_test_raw = X_test_raw
			self.y_test_raw = y_test_raw  # Keep original for evaluation
		else:
			self.X_test = None
			self.y_test = None
			self.X_test_raw = None
			self.y_test_raw = None
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print(f"Using device: {self.device}")

		self.X_train = torch.tensor(self.X_train, dtype=torch.float32).to(self.device)
		self.y_train = torch.tensor(self.y_train, dtype=torch.float32).to(self.device)
		if self.X_test is not None:
			self.X_test = torch.tensor(self.X_test, dtype=torch.float32).to(self.device)
			self.y_test = torch.tensor(self.y_test, dtype=torch.float32).to(self.device)


		

		
		
	def load_data_train(self):
		data = pd.read_csv(self.file_path_train)
		if len(data.columns) == 10:
			#first 7 columns are features, last 3 are target
			X_train = data.iloc[:, :7].values
			y_train = data.iloc[:, 7:].values
		elif len(data.columns) == 14:
			#first 10 columns are features, last 4 are target
			X_train = data.iloc[:, :10].values
			y_train = data.iloc[:, 10:].values
		elif len(data.columns) == 18:
			#first 12 columns are features, last 6 are target
			X_train = data.iloc[:, :12].values
			y_train = data.iloc[:, 12:].values
		else:
			raise ValueError("Unexpected number of columns in training data")
		
		return X_train, y_train

	def load_data_test(self):
		if self.file_path_test is None:
			return None, None
		data = pd.read_csv(self.file_path_test)
		if len(data.columns) == 10:
			#first 7 columns are features, last 3 are target
			X_test = data.iloc[:, :7].values
			y_test = data.iloc[:, 7:].values
		elif len(data.columns) == 14:
			#first 10 columns are features, last 4 are target
			X_test = data.iloc[:, :10].values
			y_test = data.iloc[:, 10:].values
		elif len(data.columns) == 18:
			#first 12 columns are features, last 6 are target
			X_test = data.iloc[:, :12].values
			y_test = data.iloc[:, 12:].values
		else:
			raise ValueError("Unexpected number of columns in test data")
		
		return X_test, y_test
	
class MLPModel(nn.Module):
	def __init__(self, input_size, output_size):
		super(MLPModel, self).__init__()
		self.fc1 = nn.Linear(input_size, 256)
		# self.dropout1 = nn.Dropout(0.1)
		self.fc2 = nn.Linear(256, 128)
		self.dropout2 = nn.Dropout(0.2)
		self.fc3 = nn.Linear(128, 64)
		self.dropout3 = nn.Dropout(0.2)
		self.fc4 = nn.Linear(64, output_size)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.relu(self.fc1(x))
		# x = self.dropout1(x)
		x = self.relu(self.fc2(x))
		x = self.dropout2(x)
		x = self.relu(self.fc3(x))
		x = self.dropout3(x)
		x = self.fc4(x)
		return x
	
	

def train_model(model, dataset, epochs=200, batch_size=256, learning_rate=0.01):
	criterion = nn.MSELoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=1e-4)
	
	
	dataloader = DataLoader(torch.utils.data.TensorDataset(dataset.X_train, dataset.y_train), batch_size=batch_size, shuffle=True)
	scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(dataloader), epochs=epochs)
	
	# Track best model
	best_val_loss = float('inf')
	best_model_state = None
	
	pbar = tqdm(total=epochs, desc='Training', unit='epoch')
	
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
			
			pbar.update(1)
			pbar.set_postfix(train_loss=f'{epoch_loss:.4f}', val_loss=f'{val_loss:.4f}', lr=f'{optimizer.param_groups[0]["lr"]:.6f}')
		else:
			pbar.update(1)
			pbar.set_postfix(loss=f'{epoch_loss:.4f}', lr=f'{optimizer.param_groups[0]["lr"]:.6f}')
	
	pbar.close()
	
	# Restore best model
	if best_model_state is not None:
		model.load_state_dict(best_model_state)
		print(f'\nRestored best model with validation loss: {best_val_loss:.6f}')


def predict(model, X):
	model.eval()
	with torch.no_grad():
		predictions = model(X)
	return predictions.cpu().numpy()


def evaluate_model(model, dataset):
	predictions_scaled = predict(model, dataset.X_test)
	predictions = dataset.scaler_y.inverse_transform(predictions_scaled)

	y_test_original = dataset.y_test_raw
	mse = np.mean((predictions - y_test_original) ** 2, axis=0)
	rmse = np.sqrt(mse)
	r2_scores = 1 - (np.sum((y_test_original - predictions) ** 2, axis=0) / np.sum((y_test_original - np.mean(y_test_original, axis=0)) ** 2, axis=0))
	
	return mse, rmse, r2_scores


def print_evaluation(model, dataset):
	mse, rmse, r2 = evaluate_model(model, dataset)
	num_outputs = dataset.y_test_raw.shape[1]
	for i in range(num_outputs):
		print(f'Output {i+1}:')
		print(f'  Mean Squared Error: {mse[i]:.6f}')
		print(f'  Root Mean Squared Error: {rmse[i]:.6f}')
		print(f'  R² Score: {r2[i]:.6f}')
	print(f'Averaged over all outputs:')
	print(f'   Mean Squared Error: {np.mean(mse):.6f}')
	print(f'   Root Mean Squared Error: {np.mean(rmse):.6f}')
	print(f'   R² Score: {np.mean(r2):.6f}')
	
def make_graphs(model, dataset):
	predictions_scaled = predict(model, dataset.X_test)
	predictions = dataset.scaler_y.inverse_transform(predictions_scaled)
	
	# Scatter plot of actual vs predicted for each output
	num_outputs = dataset.y_test_raw.shape[1]
	for i in range(num_outputs):
		plt.figure()
		plt.scatter(dataset.y_test_raw[:, i], predictions[:, i], alpha=0.5)
		plt.xlabel('Actual Values')
		plt.ylabel('Predicted Values')
		plt.title(f'Actual vs Predicted for Output {i+1}')
		plt.plot([dataset.y_test_raw[:, i].min(), dataset.y_test_raw[:, i].max()], [dataset.y_test_raw[:, i].min(), dataset.y_test_raw[:, i].max()], 'k--', lw=2)
		plt.savefig(f'figures/actual_vs_predicted_output_joint_mlp_{i+1}.png')
		plt.show()
		plt.close()


def MSE_graph(model, dataset):
	predictions_scaled = predict(model, dataset.X_test)
	predictions = dataset.scaler_y.inverse_transform(predictions_scaled)
	mse_values = (dataset.y_test_raw - predictions) ** 2
	normalized_mse_values = mse_values / np.mean(dataset.y_test_raw**2)
	plt.plot(normalized_mse_values)
	plt.xlabel('Sample Index')
	plt.ylabel('Normalized Mean Squared Error')
	plt.title('Normalized Mean Squared Error for Each Sample')
	plt.savefig('figures/normalized_mse_for_each_sample_mlp.png')
	plt.show()
	plt.close()


def RMSE_graph(model, dataset):
	predictions_scaled = predict(model, dataset.X_test)
	predictions = dataset.scaler_y.inverse_transform(predictions_scaled)
	rmse_values = np.sqrt((dataset.y_test_raw - predictions) ** 2)
	normalized_rmse_values = rmse_values / np.mean(dataset.y_test_raw)
	plt.plot(normalized_rmse_values)
	plt.xlabel('Sample Index')
	plt.ylabel('Normalized RMSE')
	plt.title('Normalized RMSE for Each Sample')
	plt.savefig('figures/normalized_rmse_for_each_sample_mlp.png')
	plt.show()
	plt.close()


def r2_graph(model, dataset):
	predictions_scaled = predict(model, dataset.X_test)
	predictions = dataset.scaler_y.inverse_transform(predictions_scaled)
	num_outputs = dataset.y_test_raw.shape[1]
	r2_scores = []
	for i in range(num_outputs):
		ss_res = np.sum((dataset.y_test_raw[:, i] - predictions[:, i]) ** 2)
		ss_tot = np.sum((dataset.y_test_raw[:, i] - np.mean(dataset.y_test_raw[:, i])) ** 2)
		r2 = 1 - (ss_res / ss_tot)
		r2_scores.append(r2)
	plt.bar(range(1, num_outputs + 1), r2_scores)
	plt.xlabel('Output Index')
	plt.ylabel('R2 Score')
	plt.title('R2 Score for Each Output')
	plt.savefig('figures/r2_score_for_each_output_mlp.png')
	plt.show()
	plt.close()


def save_model(model, file_path):
	torch.save(model.state_dict(), file_path)


def load_model(file_path, input_size, output_size):
	model = MLPModel(input_size, output_size).to(device)
	model.load_state_dict(torch.load(file_path))
	model.eval()
	return model
	

if __name__ == "__main__":
	
	
	dataset = DatasetMLP(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)

	model = MLPModel(input_size=dataset.X_train.shape[1], output_size=dataset.y_train.shape[1]).to(device)
	train_model(model, dataset, epochs=200, batch_size=128, learning_rate=0.1)
	print_evaluation(model, dataset)
	make_graphs(model, dataset)
	MSE_graph(model, dataset)
	RMSE_graph(model, dataset)
	r2_graph(model, dataset)
	save_model(model, './models/mlp_model.pth')


	