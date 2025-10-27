import streamlit as st
import numpy as np
import joblib
import pickle
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError


# Helper to load with joblib first, then pickle as fallback
def load_artifact(path):
    try:
        return joblib.load(path)
    except Exception:
        with open(path, 'rb') as f:
            return pickle.load(f)


# Load model and scaler (show clear error in the app if missing)
try:
    model = load_artifact('banknote_model.pkl')
except Exception as e:
    st.error(f"Failed to load model artifact 'banknote_model.pkl': {e}")
    st.stop()

try:
    scaler = load_artifact('scaler.pkl')
except Exception:
    scaler = None

# Verify the loaded model is fitted (gives clearer guidance than raw AttributeError)
try:
    check_is_fitted(model)
    model_is_fitted = True
except Exception:
    model_is_fitted = False

if not model_is_fitted:
    st.error("⚠️ The loaded model does not appear to be fitted. Prediction cannot proceed.\n\nPlease train the model and save a fitted estimator to 'banknote_model.pkl', or provide a fitted model file.")
    st.caption(f"Model type: {type(model)}")
    st.stop()

st.title("💵 Banknote Authentication App")
st.write("Predict whether a banknote is **Genuine or Fake** based on its characteristics.")

# Input fields
st.header("Enter the Banknote Details:")
variance = st.number_input("Variance of Wavelet Transformed Image", format="%.5f")
skewness = st.number_input("Skewness of Wavelet Transformed Image", format="%.5f")
curtosis = st.number_input("Curtosis of Wavelet Transformed Image", format="%.5f")
entropy = st.number_input("Entropy of Image", format="%.5f")

if st.button("Predict"):
    features = np.array([[variance, skewness, curtosis, entropy]])

    # Try scaling
    try:
        scaled_features = scaler.transform(features)
    except Exception:
        scaled_features = features

    # Try predicting
    try:
        if isinstance(model, BaseEstimator):
            prediction = model.predict(scaled_features)[0]
        else:
            raise TypeError("Loaded model is not a valid sklearn estimator.")
    except Exception as e:
        # Show more helpful debug info in the app
        st.error(f"⚠️ Model error during prediction: {e}")
        st.write("Model type:", type(model))
        st.write("Model attributes:", sorted([a for a in dir(model) if not a.startswith('__')])[:50])
        st.stop()

    # Show result
    if prediction == 1:
        st.success("✅ The banknote is **Genuine**.")
    else:
        st.error("🚨 The banknote is **Fake**.")

st.caption("Developed using Streamlit | Model: SVM or Logistic Regression Classifier")
