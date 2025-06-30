import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle

# Config
CSV_FILE = 'symptom_dataset.csv'
MODEL_FILE = 'best_model_symptom_dataset.pth'
LABEL_ENCODER_FILE = 'label_encoder_symptom_dataset.pkl'
SCALER_FILE = 'scaler_symptom_dataset.pkl'
OHE_FILE = 'ohe_symptom_dataset.pkl'
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Vital feature lists
VITAL_NUMERIC = ['age', 'weight', 'height', 'symptom_onset_days']
VITAL_CATEGORICAL = ['sex', 'ethnicity', 'chronic_diseases', 'allergies', 'medications']
VITAL_BINARY = ['smoker', 'alcohol_use']

# 1. Load data
print('Loading data...')
df = pd.read_csv(CSV_FILE)
counts = df['disease'].value_counts()
df = df[df['disease'].isin(counts[counts > 1].index)]  # Remove classes with <2 samples

symptom_cols = [c for c in df.columns if c not in ['disease'] + VITAL_NUMERIC + VITAL_CATEGORICAL + VITAL_BINARY]
X_symptoms = df[symptom_cols].astype(np.float32)
X_num = df[VITAL_NUMERIC].astype(np.float32)
X_bin = df[VITAL_BINARY].astype(np.float32)
X_cat = df[VITAL_CATEGORICAL].astype(str)
y = df['disease'].values

# 2. Preprocessing
print('Preprocessing...')
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_cat_ohe = ohe.fit_transform(X_cat)

X_all = np.concatenate([X_symptoms.values, X_num_scaled, X_bin.values, X_cat_ohe], axis=1)

# Save preprocessors
with open(SCALER_FILE, 'wb') as f:
    pickle.dump(scaler, f)
with open(OHE_FILE, 'wb') as f:
    pickle.dump(ohe, f)

# 3. Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
with open(LABEL_ENCODER_FILE, 'wb') as f:
    pickle.dump(le, f)

print(f'X shape: {X_all.shape}')
print(f'Number of unique diseases: {len(set(y))}')

# 4. Dataset
class SymptomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 5. Model
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

# 6. Train/val split
X_train, X_val, y_train, y_val = train_test_split(X_all, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)
train_ds = SymptomDataset(X_train, y_train)
val_ds = SymptomDataset(X_val, y_val)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# 7. Training
model = MLP(X_all.shape[1], len(le.classes_)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

print('Training...')
best_val_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        out = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * Xb.size(0)
        _, preds = torch.max(out, 1)
        correct += (preds == yb).sum().item()
        total += Xb.size(0)
    train_loss /= total
    train_acc = correct / total

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            out = model(Xb)
            loss = criterion(out, yb)
            val_loss += loss.item() * Xb.size(0)
            _, preds = torch.max(out, 1)
            val_correct += (preds == yb).sum().item()
            val_total += Xb.size(0)
    val_loss /= val_total
    val_acc = val_correct / val_total

    print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_FILE)
        print(f"Best model saved at epoch {epoch}.")

print("Training complete.") 