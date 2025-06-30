import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import shap

st.set_page_config(page_title="AI Symptom Checker", layout="centered")
st.title("🩺 AI Symptom & Disease Predictor")

# Load encoders and models
@st.cache_resource
def load_resources():
    with open('scaler_symptom_dataset.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('ohe_symptom_dataset.pkl', 'rb') as f:
        ohe = pickle.load(f)
    with open('label_encoder_symptom_dataset.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('rf_model_symptom_dataset.pkl', 'rb') as f:
        rf = pickle.load(f)
    # Get features
    with open('symptom_dataset.csv', 'r', encoding='utf-8') as f:
        header = f.readline().strip().split(',')
    all_features = header[1:]
    symptom_names = [c for c in all_features if c not in VITAL_NUMERIC + VITAL_CATEGORICAL + VITAL_BINARY]
    return scaler, ohe, le, rf, all_features, symptom_names

# Model definition
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

# Vital feature lists
VITAL_NUMERIC = ['age', 'weight', 'height', 'symptom_onset_days']
VITAL_CATEGORICAL = ['sex', 'ethnicity', 'chronic_diseases', 'allergies', 'medications']
VITAL_BINARY = ['smoker', 'alcohol_use']
SEXES = ['M', 'F']
ETHNICITIES = ['White', 'Black', 'Asian', 'Hispanic', 'Other']
CHRONIC_DISEASES = ['diabetes', 'hypertension', 'asthma', 'none']
ALLERGIES = ['penicillin', 'nuts', 'pollen', 'none']
MEDICATIONS = ['metformin', 'lisinopril', 'albuterol', 'none']

scaler, ohe, le, rf, all_features, symptom_names = load_resources()

# User input form
with st.form("patient_form"):
    st.subheader("Patient Information")
    age = st.number_input('Age', min_value=0, max_value=120, value=30)
    weight = st.number_input('Weight (kg)', min_value=30, max_value=200, value=70)
    height = st.number_input('Height (cm)', min_value=100, max_value=250, value=170)
    onset = st.number_input('Symptom onset (days ago)', min_value=0, max_value=60, value=3)
    sex = st.selectbox('Sex', SEXES)
    ethnicity = st.selectbox('Ethnicity', ETHNICITIES)
    chronic = st.selectbox('Chronic diseases', CHRONIC_DISEASES)
    allergies = st.selectbox('Allergies', ALLERGIES)
    meds = st.selectbox('Medications', MEDICATIONS)
    smoker = st.radio('Smoker', [0, 1], format_func=lambda x: 'Yes' if x else 'No')
    alcohol = st.radio('Alcohol use', [0, 1], format_func=lambda x: 'Yes' if x else 'No')
    st.subheader("Symptoms")
    selected_symptoms = st.multiselect('Select symptoms', symptom_names)
    symptom_severity = {}
    for s in selected_symptoms:
        symptom_severity[s] = st.slider(f"Severity for '{s}'", 0.0, 1.0, 1.0, 0.05)
    submitted = st.form_submit_button("Predict Disease")

if submitted:
    # Build input vector
    vital_numeric = [age, weight, height, onset]
    vital_categorical = [sex, ethnicity, chronic, allergies, meds]
    vital_binary = [smoker, alcohol]
    symptom_vector = np.zeros(len(symptom_names), dtype=np.float32)
    for i, s in enumerate(symptom_names):
        if s in symptom_severity:
            symptom_vector[i] = symptom_severity[s]
    X_num_scaled = scaler.transform([vital_numeric])
    X_cat_ohe = ohe.transform([vital_categorical])
    X_bin = np.array([vital_binary], dtype=np.float32)
    X_input = np.concatenate([symptom_vector.reshape(1, -1), X_num_scaled, X_bin, X_cat_ohe], axis=1)
    # Load MLP
    input_dim = X_input.shape[1]
    num_classes = len(le.classes_)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(input_dim, num_classes).to(device)
    model.load_state_dict(torch.load('best_model_symptom_dataset.pth', map_location=device))
    model.eval()
    # Predict
    with torch.no_grad():
        logits = model(torch.tensor(X_input, dtype=torch.float32).to(device))
        mlp_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    rf_probs = rf.predict_proba(X_input)[0]
    ensemble_probs = (mlp_probs + rf_probs) / 2
    pred_idx = np.argmax(ensemble_probs)
    pred_disease = le.inverse_transform([pred_idx])[0]
    st.success(f"Predicted Disease: {pred_disease}")
    # Top 3
    st.subheader("Top 3 predictions (ensemble):")
    top3 = np.argsort(ensemble_probs)[::-1][:3]
    for idx in top3:
        disease = le.inverse_transform([idx])[0]
        st.write(f"{disease}: {ensemble_probs[idx]:.2f}")
    # SHAP explainability
    st.subheader("Top features contributing to MLP prediction:")
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
        st.write(f"{feature_names[i]}: {shap_vals[i]:.3f}") 