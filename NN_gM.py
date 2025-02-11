import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.metrics import mean_absolute_error, r2_score
import copy
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_1samp

# Read data
# training_data = pd.read_csv("C:/Users/youse/Desktop/QMPLUS DATA MARCH 2021/QMPLUS DATA Top 15 features selected/Training Data 80%.csv")
# testing_data = pd.read_csv("C:/Users/youse/Desktop/QMPLUS DATA MARCH 2021/QMPLUS DATA Top 15 features selected/Testing Data 20%.csv")
training_data = pd.read_csv("QMPLUS DATA MARCH 2021/QMPLUS DATA PCA/Training Data 80%.csv")
testing_data = pd.read_csv("QMPLUS DATA MARCH 2021/QMPLUS DATA PCA/Testing Data 20%.csv")
X_train = training_data.drop(columns=["taxi_time"])
y_train = training_data["taxi_time"]
X_test = testing_data.drop(columns=["taxi_time"])
y_test = testing_data["taxi_time"]

# Scale 
# UNCOMMENT when using Top 15 features dataset
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Convert data to tensor objects
X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

# Build the NN model
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 50),
    nn.ReLU(),
    nn.Linear(50, 25),
    nn.ReLU(),
    nn.Linear(25, 1)
)

# Set training parameters
loss_fct = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
epochs = 300
batch_size = 10
batch_start = torch.arange(0, len(X_train), batch_size)

# Train the model and keep track of the metrics throughout the process to plot them later
Best_Training_RMSE = np.inf
Best_Testing_RMSE = np.inf
Best_Model_Weights = None
Metrics_History = []
Training_MSE_History = []
Testing_MSE_History = []
Training_RMSE_History = []
Testing_RMSE_History = []
Training_MAE_History = []
Testing_MAE_History = []
Training_R2_History = []
Testing_R2_History = []
for Epoch in range(epochs):
    model.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
        bar.set_description(f"Epoch {Epoch}")
        for Start in bar:
            Batch_Inputs = X_train[Start:Start+batch_size]
            Batch_Targets = y_train[Start:Start+batch_size]
            Batch_Predictions = model(Batch_Inputs)
            Loss = loss_fct(Batch_Predictions, Batch_Targets)
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()
            bar.set_postfix(mse=float(Loss))
    model.eval()
    with torch.no_grad():
        Training_Predictions = model(X_train)
        Training_MSE = loss_fct(Training_Predictions, y_train)
        Training_RMSE = np.sqrt(float(Training_MSE))
        Training_MAE = mean_absolute_error(y_train, Training_Predictions)
        Training_R2 = r2_score(y_train, Training_Predictions)
        Testing_Predictions = model(X_test)
        Testing_MSE = loss_fct(Testing_Predictions, y_test)
        Testing_RMSE = np.sqrt(float(Testing_MSE))
        Testing_MAE = mean_absolute_error(y_test, Testing_Predictions)
        Testing_R2 = r2_score(y_test, Testing_Predictions)
    Training_MSE_History.append(Training_MSE.item())
    Testing_MSE_History.append(Testing_MSE.item())
    Training_RMSE_History.append(Training_RMSE)
    Testing_RMSE_History.append(Testing_RMSE)
    Training_MAE_History.append(Training_MAE)
    Testing_MAE_History.append(Testing_MAE)
    Training_R2_History.append(Training_R2)
    Testing_R2_History.append(Testing_R2)
    print(f"Epoch {Epoch+1}/{epochs}")
    print(f"Training MSE: {Training_MSE:.2f}, RMSE: {Training_RMSE:.2f}, MAE: {Training_MAE:.2f}, R²: {Training_R2:.2f}")
    print(f"Testing MSE: {Testing_MSE:.2f}, RMSE: {Testing_RMSE:.2f}, MAE: {Testing_MAE:.2f}, R²: {Testing_R2:.2f}")
    if Training_RMSE < Best_Training_RMSE:
        Best_Training_RMSE = Training_RMSE
        Best_Model_Weights = copy.deepcopy(model.state_dict())        
    if Testing_RMSE < Best_Testing_RMSE:
        Best_Testing_RMSE = Testing_RMSE
        Best_Model_Weights = copy.deepcopy(model.state_dict())
model.load_state_dict(Best_Model_Weights)
print("Best Training RMSE: %.2f" % Best_Training_RMSE)
print("Best Testing RMSE: %.2f" % Best_Testing_RMSE)

# Plot metrics
# RMSE
plt.plot(Training_RMSE_History, label='Training RMSE', color='blue')
plt.plot(Testing_RMSE_History, label='Testing RMSE', color='red')
plt.title('RMSE Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.show()
# MSE
plt.plot(Training_MSE_History, label='Training MSE', color='blue')
plt.plot(Testing_MSE_History, label='Testing MSE', color='red')
plt.title('MSE Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.show()
# MAE
plt.plot(Training_MAE_History, label='Training MAE', color='blue')
plt.plot(Testing_MAE_History, label='Testing MAE', color='red')
plt.title('MAE Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)
plt.show()
# R-Squared
plt.plot(Training_R2_History, label='Training R²', color='blue')
plt.plot(Testing_R2_History, label='Testing R²', color='red')
plt.title('R² Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('R²')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the model
model.eval()
with torch.no_grad():
    Final_Testing_Predictions = model(X_test).numpy().flatten()
    Final_Testing_Targets = y_test.numpy().flatten()

# Calculate t stats and p values.
residuals = Final_Testing_Targets - Final_Testing_Predictions
t_stat, p_value = ttest_1samp(residuals, 0)
print("T-test on residuals:")
print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
if p_value < 0.05:
    print("The mean residual is significantly different from zero (p < 0.05).")
else:
    print("The mean residual is not significantly different from zero (p >= 0.05).")