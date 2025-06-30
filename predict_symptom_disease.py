import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import shap

# Vital feature lists
VITAL_NUMERIC = ['age', 'weight', 'height', 'symptom_onset_days']
VITAL_CATEGORICAL = ['sex', 'ethnicity', 'chronic_diseases', 'allergies', 'medications']
VITAL_BINARY = ['smoker', 'alcohol_use']

# 1. Load preprocessors and label encoder
with open('scaler_symptom_dataset.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('ohe_symptom_dataset.pkl', 'rb') as f:
    ohe = pickle.load(f)
with open('label_encoder_symptom_dataset.pkl', 'rb') as f:
    le = pickle.load(f)

# 2. Get symptom names from the training set (for user input)
# We'll use the ohe and scaler to determine the correct input size
# But for user input, we need the symptom names (not vital features)
with open('symptom_dataset.csv', 'r', encoding='utf-8') as f:
    header = f.readline().strip().split(',')
all_features = header[1:]  # skip 'disease'
symptom_names = [c for c in all_features if c not in VITAL_NUMERIC + VITAL_CATEGORICAL + VITAL_BINARY]

# 3. Model definition (must match training)
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

# 4. User input for vital features
print("\n--- Patient Information ---")
def get_numeric(prompt, minval, maxval):
    while True:
        val = input(f"{prompt} ({minval}-{maxval}): ").strip()
        if val == '':
            print(f"Please enter a value between {minval} and {maxval}.")
            continue
        try:
            v = float(val)
            if minval <= v <= maxval:
                return v
        except:
            pass
        print(f"Invalid input. Enter a number between {minval} and {maxval}.")

def get_categorical(prompt, options):
    options_str = '/'.join(options)
    options_lower = [opt.lower() for opt in options]
    while True:
        val = input(f"{prompt} ({options_str}): ").strip().lower()
        if val in options_lower:
            return options[options_lower.index(val)]
        print(f"Invalid input. Choose from: {options_str}")

def get_binary(prompt):
    while True:
        val = input(f"{prompt} (0=No, 1=Yes): ").strip()
        if val in ['0', '1']:
            return int(val)
        print("Invalid input. Enter 0 or 1.")

# Options for categorical features
SEXES = ['M', 'F']
ETHNICITIES = ['White', 'Black', 'Asian', 'Hispanic', 'Other']
CHRONIC_DISEASES = ['diabetes', 'hypertension', 'asthma', 'none']
ALLERGIES = ['penicillin', 'nuts', 'pollen', 'none']
MEDICATIONS = ['metformin', 'lisinopril', 'albuterol', 'none']

vital_numeric = [
    get_numeric('Age', 0, 120),
    get_numeric('Weight (kg)', 30, 200),
    get_numeric('Height (cm)', 100, 250),
    get_numeric('Symptom onset (days ago)', 0, 60)
]
vital_categorical = [
    get_categorical('Sex', SEXES),
    get_categorical('Ethnicity', ETHNICITIES),
    get_categorical('Chronic diseases', CHRONIC_DISEASES),
    get_categorical('Allergies', ALLERGIES),
    get_categorical('Medications', MEDICATIONS)
]
vital_binary = [
    get_binary('Smoker'),
    get_binary('Alcohol use')
]

# 5. User input for symptoms
print("\n--- Symptom Entry ---")
print("Enter symptoms you are experiencing (comma-separated, partial names allowed, case-insensitive):")
user_symptoms = input("Symptoms: ").strip().lower().split(',')
user_symptoms = [s.strip() for s in user_symptoms if s.strip()]

# Fuzzy match user symptoms to known symptoms
matched = {}
for us in user_symptoms:
    matches = [s for s in symptom_names if us in s.lower()]
    if matches:
        matched[matches[0]] = us
    else:
        print(f"Warning: Symptom '{us}' not recognized and will be ignored.")

symptom_vector = np.zeros(len(symptom_names), dtype=np.float32)
for i, s in enumerate(symptom_names):
    if s in matched:
        while True:
            sev = input(f"Severity for '{s}' (0-1, blank=1): ").strip()
            if sev == '':
                symptom_vector[i] = 1.0
                break
            try:
                v = float(sev)
                if 0.0 <= v <= 1.0:
                    symptom_vector[i] = v
                    break
            except:
                pass
            print("Invalid input. Enter a number between 0 and 1, or leave blank for 1.")

# 6. Preprocess input using the same pipeline as training
X_num_scaled = scaler.transform([vital_numeric])
X_cat_ohe = ohe.transform([vital_categorical])
X_bin = np.array([vital_binary], dtype=np.float32)
X_input = np.concatenate([symptom_vector.reshape(1, -1), X_num_scaled, X_bin, X_cat_ohe], axis=1)

# 7. Load models
input_dim = X_input.shape[1]
num_classes = len(le.classes_)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP(input_dim, num_classes).to(device)
model.load_state_dict(torch.load('best_model_symptom_dataset.pth', map_location=device))
model.eval()

with open('rf_model_symptom_dataset.pkl', 'rb') as f:
    rf = pickle.load(f)

# 8. Predict with both models and ensemble
# MLP prediction
with torch.no_grad():
    logits = model(torch.tensor(X_input, dtype=torch.float32).to(device))
    mlp_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
# RF prediction
rf_probs = rf.predict_proba(X_input)[0]
# Ensemble (average)
ensemble_probs = (mlp_probs + rf_probs) / 2
pred_idx = np.argmax(ensemble_probs)
pred_disease = le.inverse_transform([pred_idx])[0]
print(f"\nEnsemble Predicted Disease: {pred_disease}")

# Show top 3 predictions
print("Top 3 predictions (ensemble):")
top3 = np.argsort(ensemble_probs)[::-1][:3]
for idx in top3:
    disease = le.inverse_transform([idx])[0]
    print(f"  {disease}: {ensemble_probs[idx]:.2f}")

# 9. SHAP explainability for MLP
print("\nTop features contributing to MLP prediction:")
mlp_pred_idx = np.argmax(mlp_probs)
explainer = shap.DeepExplainer(model, torch.tensor(X_input, dtype=torch.float32).to(device))
shap_values = explainer.shap_values(torch.tensor(X_input, dtype=torch.float32).to(device))
if len(shap_values) == 1:
    shap_vals = np.abs(shap_values[0][0])
else:
    shap_vals = np.abs(shap_values[mlp_pred_idx][0])
feature_names = symptom_names + VITAL_NUMERIC + VITAL_BINARY + list(ohe.get_feature_names_out(VITAL_CATEGORICAL))
top_feat_idx = np.argsort(shap_vals)[::-1][:5]
for i in top_feat_idx:
    print(f"  {feature_names[i]}: {shap_vals[i]:.3f}") 