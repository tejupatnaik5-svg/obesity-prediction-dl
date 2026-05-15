import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Fix numpy compatibility for SHAP
np.bool = bool
np.int = int

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import joblib
import shap

# Create folder for figures
os.makedirs("figures", exist_ok=True)

# Load Dataset
df = pd.read_csv("dataset/Obesity_dataset.csv")

features = df.drop("Obesity level class", axis=1)
target = df["Obesity level class"]

# Encoding
X_encoded = pd.get_dummies(features, drop_first=True)

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(target)

# Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(X_encoded)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    scaled_features, y_encoded, test_size=0.2, random_state=42
)

# Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Model
input_size = X_encoded.shape[1]

class ObesityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 7)
        )

    def forward(self, x):
        return self.layers(x)

model = ObesityModel()

# Training
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 40
loss_values = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for batch_X, batch_y in train_loader:
        preds = model(batch_X)
        loss = loss_fn(preds, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Calculate average loss per epoch
    avg_loss = epoch_loss / len(train_loader)
    loss_values.append(avg_loss)

    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")



# Save loss graph
import os



plt.figure()
plt.plot(range(1, len(loss_values)+1), loss_values)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss vs Epoch")
plt.grid()

# SAVE
file_path = "figures/loss.png"
plt.savefig(file_path)

print(f"\nLoss graph saved at: {file_path}")

plt.close()

# Evaluation
model.eval()

with torch.no_grad():
    outputs = model(X_test_tensor)
    probs = F.softmax(outputs, dim=1)
    _, predictions = torch.max(probs, 1)

accuracy = (predictions == y_test_tensor).sum().item() / y_test_tensor.size(0)
print(f"\nModel accuracy: {accuracy:.4f}")

predicted_labels = target_encoder.inverse_transform(predictions.numpy())

# SHAP explainability
print("\nGenerating SHAP explanations...")

try:
    background = X_train[:50]

    def model_predict(x):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            return model(x_tensor).numpy()

    explainer = shap.Explainer(model_predict, background)
    shap_values = explainer(X_test[:5])

    feature_names = X_encoded.columns

    print("\nTop Feature Contributions (SHAP):\n")

    for i in range(5):
        print(f"\nSample {i+1} ({predicted_labels[i]}):")

        contributions = list(zip(feature_names, shap_values.values[i]))

        # Sort by importance
        contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)

        for feature, value in contributions[:5]:
            print(f"{feature}: {float(value):.4f}")

except Exception as e:
    print("SHAP explanation failed:", e)

# Confidence
confidence, _ = torch.max(probs, 1)

print("\nSample predictions with confidence:")

for i in range(min(5, len(confidence))):
    print(f"{predicted_labels[i]} -> {confidence[i].item():.4f}")

# Low Confidence
print("\nLow confidence cases:\n")

for i in range(len(confidence)):
    if confidence[i].item() < 0.6:
        print(f"{predicted_labels[i]} -> {confidence[i].item():.4f}")

# Save Files
torch.save(model.state_dict(), "model.pth")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X_encoded).columns, "columns.pkl")
joblib.dump(target_encoder, "label_encoder.pkl")

print("\nModel and preprocessing files saved successfully")