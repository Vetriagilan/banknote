# =========================
# 📘 BANKNOTE AUTHENTICATION WITH MLFLOW
# =========================

# --- Imports ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
import mlflow
import mlflow.sklearn

# --- Load Dataset ---
# Dataset should contain: variance, skewness, curtosis, entropy, class
df = pd.read_csv("BankNote_Authentication.csv")

X = df[['variance', 'skewness', 'curtosis', 'entropy']]
y = df['class']

# --- Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Scale Data ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Define Model ---
C_value = 1.0
gamma_value = 'scale'
kernel_type = 'rbf'

model = SVC(kernel=kernel_type, C=C_value, gamma=gamma_value, probability=True, random_state=42)

# --- Set MLflow Experiment ---
mlflow.set_experiment("Banknote Authentication")

with mlflow.start_run():
    # --- Train Model ---
    model.fit(X_train_scaled, y_train)
    
    # --- Predictions ---
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # --- Metrics ---
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    
    # --- Log Parameters ---
    mlflow.log_param("kernel", kernel_type)
    mlflow.log_param("C", C_value)
    mlflow.log_param("gamma", gamma_value)
    
    # --- Log Metrics ---
    mlflow.log_metric("Accuracy", acc)
    mlflow.log_metric("Precision", prec)
    mlflow.log_metric("Recall", rec)
    mlflow.log_metric("F1", f1)
    mlflow.log_metric("ROC_AUC", roc)
    
    # --- Log Artifacts (Model & Scaler) ---
    mlflow.sklearn.log_model(model, "banknote_model")
    mlflow.sklearn.log_model(scaler, "scaler")

    print("✅ Model logged successfully with MLflow!")
    print("Run ID:", mlflow.active_run().info.run_id)
    print(f"Accuracy: {acc:.3f} | F1: {f1:.3f} | ROC-AUC: {roc:.3f}")

# --- Save Locally (Optional) ---
with open("banknote_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model and Scaler saved locally as .pkl files")
