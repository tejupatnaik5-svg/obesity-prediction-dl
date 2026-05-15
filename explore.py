import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("dataset/Obesity_dataset.csv")

# Keep first 17 columns
df = df.iloc[:, :17]

# Split features and target
features = df.drop("Obesity level class", axis=1)
target = df["Obesity level class"]

# One-hot encoding
X_encoded = pd.get_dummies(features, drop_first=True)

# Save column order for later use in app
columns = X_encoded.columns
joblib.dump(columns, "columns.pkl")

# Encode Target
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(target)

joblib.dump(target_encoder, "target_encoder.pkl")

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y_encoded,
    test_size=0.2,
    random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")

# Convert to tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Model
input_size = X_train.shape[1]

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
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 40
losses = []

for epoch in range(epochs):
    model.train()

    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs, 1)

    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)

print(f"\nTest Accuracy: {accuracy:.4f}")

# Save Model
torch.save(model.state_dict(), "model.pth")

# Save training loss graph
plt.figure()
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("loss.png")

print("\n Model, scaler, columns, encoder saved successfully!")
print("loss.png generated successfully!")