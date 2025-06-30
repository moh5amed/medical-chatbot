import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

# Config
CSV_FILE = 'symptom_dataset.csv'
SCALER_FILE = 'scaler_symptom_dataset.pkl'
OHE_FILE = 'ohe_symptom_dataset.pkl'
LABEL_ENCODER_FILE = 'label_encoder_symptom_dataset.pkl'
RF_MODEL_FILE = 'rf_model_symptom_dataset.pkl'

# Vital feature lists
VITAL_NUMERIC = ['age', 'weight', 'height', 'symptom_onset_days']
VITAL_CATEGORICAL = ['sex', 'ethnicity', 'chronic_diseases', 'allergies', 'medications']
VITAL_BINARY = ['smoker', 'alcohol_use']

print('Loading data...')
df = pd.read_csv(CSV_FILE)
counts = df['disease'].value_counts()
df = df[df['disease'].isin(counts[counts > 1].index)]

symptom_cols = [c for c in df.columns if c not in ['disease'] + VITAL_NUMERIC + VITAL_CATEGORICAL + VITAL_BINARY]
X_symptoms = df[symptom_cols].astype(np.float32)
X_num = df[VITAL_NUMERIC].astype(np.float32)
X_bin = df[VITAL_BINARY].astype(np.float32)
X_cat = df[VITAL_CATEGORICAL].astype(str)
y = df['disease'].values

# Load preprocessors
with open(SCALER_FILE, 'rb') as f:
    scaler = pickle.load(f)
with open(OHE_FILE, 'rb') as f:
    ohe = pickle.load(f)
with open(LABEL_ENCODER_FILE, 'rb') as f:
    le = pickle.load(f)

X_num_scaled = scaler.transform(X_num)
X_cat_ohe = ohe.transform(X_cat)
X_all = np.concatenate([X_symptoms.values, X_num_scaled, X_bin.values, X_cat_ohe], axis=1)
y_encoded = le.transform(y)

print('Training Random Forest...')
rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
rf.fit(X_all, y_encoded)

with open(RF_MODEL_FILE, 'wb') as f:
    pickle.dump(rf, f)
print(f'Random Forest model saved to {RF_MODEL_FILE}') 