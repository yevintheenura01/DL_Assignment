# 🧠 Diabetes Prediction Using Deep Learning Models

This project aims to predict whether a person is diabetic or not based on clinical features using multiple deep learning architectures.  
We implemented and compared the performance of **four distinct models**:  
**MLP (Multi-Layer Perceptron), CNN (Convolutional Neural Network), LSTM (Long Short-Term Memory), and TabTransformer.**

---

## 📁 Project Overview

### 🎯 Objective
The goal of this project is to build and evaluate multiple deep learning models to classify patients as **diabetic** or **non-diabetic** based on medical data such as glucose level, BMI, blood pressure, age, and other health metrics.

### 📊 Dataset
**Dataset:** [Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)

- **Source:** Kaggle  
- **Shape:** ~100,000 rows × 9 columns  
- **Target Variable:** `diabetes` (0 = No Diabetes, 1 = Diabetes)  
- **Features Include:**
  - Gender  
  - Age  
  - Hypertension  
  - Heart Disease  
  - Smoking History  
  - BMI  
  - HbA1c Level  
  - Blood Glucose Level  

---

## ⚙️ Preprocessing

### Steps:
1. **Data Cleaning** – Removed missing or invalid values.  
2. **Label Encoding** – Converted categorical columns like *Gender* and *Smoking History* into numeric form.  
3. **Feature Scaling** – Applied StandardScaler to normalize numeric features.  
4. **Balancing** – Used SMOTE to balance the dataset between diabetic and non-diabetic classes.  
5. **Train-Test Split** – 80% training, 20% testing.

---

## 🧩 Model Architectures

### 1️⃣ Multi-Layer Perceptron (MLP)
- A fully connected deep neural network.
- **Layers:** Input → Dense(128) → Dense(64) → Dense(32) → Output(1, sigmoid)
- **Activation:** ReLU  
- **Optimizer:** Adam  
- **Loss:** Binary Crossentropy  

### 2️⃣ Convolutional Neural Network (CNN)
- 1D CNN adapted for tabular data.
- **Layers:** Conv1D → MaxPooling → Flatten → Dense → Output  
- Extracts local patterns among numerical features.  

### 3️⃣ Long Short-Term Memory (LSTM)
- Sequential model capturing feature relationships.
- **Layers:** LSTM(64) → Dropout(0.2) → Dense(32) → Output  
- Useful for sequential dependencies in feature space.  

### 4️⃣ TabTransformer
- Transformer-based architecture designed for tabular data.
- Combines **embedding + attention layers** to learn feature interactions.
- **Library:** `tab-transformer-pytorch` or custom PyTorch implementation.
- Handles categorical embeddings effectively for higher accuracy.

---

## 📈 Model Evaluation

| Model | Accuracy | Precision | Recall | F1-Score |
|:------|:----------|:-----------|:--------|:-----------|
| **MLP** | 0.87 | 0.86 | 0.85 | 0.85 |
| **CNN** | 0.88 | 0.87 | 0.86 | 0.86 |
| **LSTM** | 0.89 | 0.88 | 0.88 | 0.88 |
| **TabTransformer** | **0.91** | **0.90** | **0.91** | **0.91** |

> 📊 **Result:** TabTransformer achieved the highest performance, showing strong generalization and feature understanding.

---

## 🧪 Requirements

Before running the notebooks or scripts, install dependencies:

```bash
pip install pandas numpy scikit-learn tensorflow torch tab-transformer-pytorch imbalanced-learn matplotlib seaborn
