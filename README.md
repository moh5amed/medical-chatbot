# 🩺 AI Symptom & Disease Predictor

A web-based AI tool for predicting diseases from patient symptoms and vital information, powered by an ensemble of deep learning (MLP) and Random Forest models, with SHAP explainability.

---

## Features
- **Web-based UI** (Streamlit)
- **Ensemble predictions** (MLP + Random Forest)
- **Top-3 disease predictions**
- **SHAP explainability**: See which features contributed most to the prediction
- **Flexible input**: Patient demographics, vitals, and symptoms with severity

---

## Setup & Usage

### 1. **Clone the repository**
```
git clone <your-repo-url>
cd <your-repo-directory>
```

### 2. **Install requirements**
```
pip install -r requirements.txt
```

### 3. **Make sure these files are present:**
- `app.py` (the Streamlit app)
- `symptom_dataset.csv` (your dataset)
- `best_model_symptom_dataset.pth` (MLP model)
- `rf_model_symptom_dataset.pkl` (Random Forest model)
- `scaler_symptom_dataset.pkl`, `ohe_symptom_dataset.pkl`, `label_encoder_symptom_dataset.pkl` (preprocessing objects)

### 4. **Run the app locally**
```
streamlit run app.py
```
Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Deployment

### **Streamlit Community Cloud**
1. Push your code and all required files to GitHub.
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and deploy your app.

### **Hugging Face Spaces**
1. Create a new Space, choose Streamlit, and upload your files.

---

## Example Usage
1. Enter patient info (age, sex, etc.)
2. Select symptoms and their severity
3. Click "Predict Disease"
4. View the top-3 predictions and the most important features for the AI's decision

---

## Credits
- Built with [Streamlit](https://streamlit.io/), [PyTorch](https://pytorch.org/), [scikit-learn](https://scikit-learn.org/), and [SHAP](https://shap.readthedocs.io/)
- AI model and data generation by Mohamed Hossam

---
