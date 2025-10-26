# 💵 Banknote Authentication App

A simple **Streamlit web app** that predicts whether a **banknote is genuine or fake** based on statistical features extracted from its image.

---

## 📘 Overview
This project uses a trained machine learning model (SVM or similar) to classify banknotes.  
The model was trained using the **Banknote Authentication Dataset**, which contains features derived from images of genuine and forged banknotes.

---

## 🧠 Model Files
| File | Description |
|------|--------------|
| `banknote_model.pkl` | Trained machine learning model |
| `scaler.pkl` | Preprocessing scaler (used for feature normalization) |
| `app.py` | Streamlit app for deployment |
| `bank.ipynb` | Jupyter notebook used for training and evaluation |

---

## ⚙️ Installation

### 1. Clone or download the repository
```bash
git clone https://github.com/yourusername/banknote-authentication-app.git
cd banknote-authentication-app
```

### 2. Install dependencies
Make sure you have Python 3.8+ installed, then run:
```bash
pip install streamlit numpy scikit-learn
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

Once started, Streamlit will open the app automatically in your browser  
(default URL: http://localhost:8501)

---

## 🧾 Input Features
| Feature | Description |
|----------|--------------|
| **Variance** | Variance of Wavelet Transformed Image |
| **Skewness** | Skewness of Wavelet Transformed Image |
| **Curtosis** | Curtosis of Wavelet Transformed Image |
| **Entropy** | Entropy of Image |

---

## 🎯 Output
The model predicts whether the banknote is:

- ✅ **Genuine**
- 🚨 **Fake**

---

## 🛠 Example
1. Enter the values for variance, skewness, curtosis, and entropy.  
2. Click **"Predict"**.  
3. The app will display the result instantly.

---

## 📄 License
This project is released under the MIT License.  
Feel free to modify and use it for learning or deployment purposes.

---

## 👨‍💻 Author
Developed by **Vetriagilan**  
📧 Contact: [your-email@example.com]
