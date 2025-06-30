import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# 1. Load data
VITAL_NUMERIC = ['age', 'weight', 'height', 'symptom_onset_days']
VITAL_CATEGORICAL = ['sex', 'ethnicity', 'chronic_diseases', 'allergies', 'medications']
VITAL_BINARY = ['smoker', 'alcohol_use']

# Read header to get all columns
df = pd.read_csv("symptom_dataset.csv")
counts = df["disease"].value_counts()
df = df[df["disease"].isin(counts[counts > 1].index)]

symptom_cols = [c for c in df.columns if c not in ['disease'] + VITAL_NUMERIC + VITAL_CATEGORICAL + VITAL_BINARY]
X_symptoms = df[symptom_cols].astype(np.float32)
X_num = df[VITAL_NUMERIC].astype(np.float32)
X_bin = df[VITAL_BINARY].astype(np.float32)
X_cat = df[VITAL_CATEGORICAL].astype(str)
y = df["disease"].values

# Preprocessing
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_cat_ohe = ohe.fit_transform(X_cat)

X_all = np.concatenate([X_symptoms.values, X_num_scaled, X_bin.values, X_cat_ohe], axis=1)

# Save preprocessors
with open("scaler_symptom_dataset.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("ohe_symptom_dataset.pkl", "wb") as f:
    pickle.dump(ohe, f)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Print summary
print(f"X shape: {X_all.shape}")
print(f"Number of unique diseases: {len(set(y))}")
print(pd.Series(y).value_counts())

# K-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accs = []
for fold, (train_idx, test_idx) in enumerate(skf.split(X_all, y_encoded)):
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    class SymptomDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_ds = SymptomDataset(X_train, y_train)
    test_ds = SymptomDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    class MLP(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            )
        def forward(self, x):
            return self.net(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(X_all.shape[1], len(le.classes_)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(30):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # Optionally print: print(f"Fold {fold+1} Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    acc = correct / total
    accs.append(acc)
    print(f"Fold {fold+1} Accuracy: {acc:.2%}")

# Save final model and encoders on all data
model = MLP(X_all.shape[1], len(le.classes_)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
train_ds = SymptomDataset(X_all, y_encoded)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
for epoch in range(30):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "best_model_symptom_dataset.pth")
with open("label_encoder_symptom_dataset.pkl", "wb") as f:
    pickle.dump(le, f)
print(f"Mean CV Accuracy: {np.mean(accs):.2%}")
print("Model and encoders saved.")

# After all preprocessing steps (just before model definition)
print(f"[DEBUG] Shape of X after preprocessing: {X_all.shape}")
input_dim = X_all.shape[1]
print(f"[DEBUG] input_dim used for model: {input_dim}")

# After saving the model
with open("model_input_dim.txt", "w") as f:
    f.write(str(input_dim))
print(f"[DEBUG] Saved model input_dim: {input_dim}")    
