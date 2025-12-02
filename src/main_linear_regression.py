import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import joblib
import sys
import matplotlib.pyplot as plt
from datetime import datetime

class LinearRegressionModel:
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
            self.y_test = y_test_raw  # Keep original scale for evaluation
            self.X_test_raw = X_test_raw
            self.y_test_raw = y_test_raw
        else:
            self.X_test = None
            self.y_test = None
            self.X_test_raw = None
            self.y_test_raw = None

    def load_data_train(self):
        data = pd.read_csv(self.file_path_train)
        # X_train = data.iloc[:, :-1].values  # Features
        # y_train = data.iloc[:, -1].values   # Target variabli
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

    def train_model(self):
        # Train on pre-scaled data
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        # Predict on pre-scaled test data
        predictions_scaled = self.model.predict(self.X_test)
        # Inverse transform predictions back to original scale
        predictions = self.scaler_y.inverse_transform(predictions_scaled)
        
        # Calculate metrics per output
        mse = np.mean((self.y_test - predictions) ** 2, axis=0)
        rmse = np.sqrt(mse)
        r2_scores = 1 - (np.sum((self.y_test - predictions) ** 2, axis=0) / 
                        np.sum((self.y_test - np.mean(self.y_test, axis=0)) ** 2, axis=0))
        
        return mse, rmse, r2_scores
    
    def run(self):
        # Handle case where no separate test file was provided
        if self.X_test is None:
            # Split the scaled training data
            X_train, X_test, y_train, y_test = train_test_split(
                self.X_train, self.y_train, test_size=0.2, random_state=42)
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            # For y_test, we need to inverse transform to get original scale for metrics
            self.y_test = self.scaler_y.inverse_transform(y_test)
        
        self.train_model()
        self.print_evaluation()
    
    def print_evaluation(self):
        mse, rmse, r2 = self.evaluate_model()
        num_outputs = self.y_test.shape[1]
        
        print(f'\n=== Model Performance ===')
        for i in range(num_outputs):
            print(f'Output {i+1}:')
            print(f'  Mean Squared Error: {mse[i]:.6f}')
            print(f'  Root Mean Squared Error: {rmse[i]:.6f}')
            print(f'  R² Score: {r2[i]:.6f}')
        print(f'Averaged over all outputs:')
        print(f'   Mean Squared Error: {np.mean(mse):.6f}')
        print(f'   Root Mean Squared Error: {np.mean(rmse):.6f}')
        print(f'   R² Score: {np.mean(r2):.6f}')
    
    def savemodel(self, file_name):
        # Save model and scalers together
        joblib.dump({
            'model': self.model,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y
        }, file_name)
        print(f'Model saved to {file_name}')

    def loadmodel(self, file_name):
        # Load model and scalers together
        saved_data = joblib.load(file_name)
        self.model = saved_data['model']
        self.scaler_X = saved_data['scaler_X']
        self.scaler_y = saved_data['scaler_y']
        print(f'Model loaded from {file_name}')
    
    def run_loaded_model(self):
        if self.model is None:
            print("No model loaded. Please load a model first.")
            return
        
        self.print_evaluation()
        
    def make_graphs(self):
        if self.model is None:
            print("No model loaded. Please load a model first.")
            return
        predictions_scaled = self.model.predict(self.X_test)
        predictions = self.scaler_y.inverse_transform(predictions_scaled)
        
        # Scatter plot of actual vs predicted for each output
        num_outputs = self.y_test.shape[1]
        num_joints = num_outputs
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        for i in range(num_outputs):
            plt.figure()
            plt.scatter(self.y_test[:, i], predictions[:, i], alpha=0.5)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(f'Actual vs Predicted for Output {i+1}')
            plt.plot([self.y_test[:, i].min(), self.y_test[:, i].max()], [self.y_test[:, i].min(), self.y_test[:, i].max()], 'k--', lw=2)
            plt.savefig(f'figures/actual_vs_predicted_output_joint_lr_{i+1}_{num_joints}joints_{timestamp}.png')
            plt.show()
            plt.close()

    def MSE_graph(self):
        if self.model is None:
            print("No model loaded. Please load a model first.")
            return
        predictions_scaled = self.model.predict(self.X_test)
        predictions = self.scaler_y.inverse_transform(predictions_scaled)
        mse_values = (self.y_test - predictions) ** 2
        # Use variance instead of mean squared to avoid division issues with centered data
        normalized_mse_values = mse_values / (np.var(self.y_test) + 1e-8)
        num_joints = self.y_test.shape[1]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.plot(normalized_mse_values)
        plt.xlabel('Sample Index')
        plt.ylabel('Normalized Mean Squared Error')
        plt.title('Normalized Mean Squared Error for Each Sample')
        plt.savefig(f'figures/normalized_mse_for_each_sample_lr_{num_joints}joints_{timestamp}.png')
        plt.show()
        plt.close()
    
    def RMSE_graph(self):
        if self.model is None:
            print("No model loaded. Please load a model first.")
            return
        predictions_scaled = self.model.predict(self.X_test)
        predictions = self.scaler_y.inverse_transform(predictions_scaled)
        rmse_values = np.sqrt((self.y_test - predictions) ** 2)
        # Use std instead of mean to avoid division by near-zero for centered data
        normalized_rmse_values = rmse_values / (np.std(self.y_test) + 1e-8)
        num_joints = self.y_test.shape[1]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.plot(normalized_rmse_values)
        plt.xlabel('Sample Index')
        plt.ylabel('Normalized RMSE')
        plt.title('Normalized RMSE for Each Sample')
        plt.savefig(f'figures/normalized_rmse_for_each_sample_lr_{num_joints}joints_{timestamp}.png')
        plt.show()
        plt.close()
    
    def r2_graph(self):
        if self.model is None:
            print("No model loaded. Please load a model first.")
            return
        predictions_scaled = self.model.predict(self.X_test)
        predictions = self.scaler_y.inverse_transform(predictions_scaled)
        num_outputs = self.y_test.shape[1]
        num_joints = num_outputs
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        r2_scores = []
        for i in range(num_outputs):
            ss_res = np.sum((self.y_test[:, i] - predictions[:, i]) ** 2)
            ss_tot = np.sum((self.y_test[:, i] - np.mean(self.y_test[:, i])) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            r2_scores.append(r2)
        plt.bar(range(1, num_outputs + 1), r2_scores)
        plt.xlabel('Output Index')
        plt.ylabel('R2 Score')
        plt.title('R2 Score for Each Output')
        plt.savefig(f'figures/r2_score_for_each_output_lr_{num_joints}joints_{timestamp}.png')
        plt.show()
        plt.close()

if __name__ == "__main__":

    model = LinearRegressionModel(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    model.run()
    
    # Save model with timestamp and joint count
    num_joints = model.y_test.shape[1]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model.savemodel(f'./models/linear_regression_model_{num_joints}joints_{timestamp}.pkl')
    
    # Generate graphs
    model.make_graphs()
    model.MSE_graph()
    model.RMSE_graph()
    model.r2_graph()
    