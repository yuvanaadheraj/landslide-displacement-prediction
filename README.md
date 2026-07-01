# 🌍 Landslide Displacement Prediction using Machine Learning & Deep Learning

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow)

A comprehensive **multi-model machine learning system** for predicting **landslide displacement** along the **X, Y, and Z axes** using environmental and sensor data collected from landslide monitoring stations.

The project implements and compares multiple prediction approaches, including **XGBoost, MLP, GRU, Graph Neural Networks (GNN + GRU), and a Fuzzy Inference System (FIS)**, providing a comparative study of traditional machine learning, deep learning, graph learning, and rule-based AI techniques.

---

## 🚀 Features

- 📊 Predicts landslide displacement in **X, Y, and Z** directions
- 🤖 Multiple ML & DL models for performance comparison
- 🌐 Graph Neural Network for spatial learning
- ⏳ GRU-based temporal sequence modeling
- 🧠 Fuzzy Inference System for rule-based prediction
- 📈 Performance evaluation using RMSE, MAE, and R²
- 🎯 Interactive Streamlit web application
- 💾 Automatic model and scaler saving

---

## 🏗️ Models Implemented

| Model | Description | RMSE ↓ | MAE ↓ | R² ↑ |
|--------|-------------|--------:|--------:|--------:|
| 🥇 XGBoost | Gradient Boosting Regressor | **0.9245** | — | **0.8797** |
| 🥈 MLP | Multi-Layer Perceptron | **1.3303** | **0.8046** | **0.7820** |
| 🥉 GRU | Gated Recurrent Unit | **1.5274** | **1.0352** | **0.7094** |
| 🌐 GNN + GRU | Spatio-Temporal Graph Neural Network | **1.7405** | **1.1828** | **0.5918** |
| 🧩 Fuzzy Inference System | Rule-Based AI | **1.9258** | — | **-0.0370** |

---

## 📂 Project Structure

```text
Landslide_Project/
│
├── src/
│   ├── baselines/
│   │   ├── models.py
│   │   ├── train.py
│   │   └── train_xgboost.py
│   │
│   ├── gnn/
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── data_loader.py
│   │   └── utils.py
│   │
│   ├── config.py
│   ├── data_loader.py
│   └── utils.py
│
├── app.py
├── main.py
├── train_fis.py
├── datalandslide.csv
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

The dataset contains **time-series sensor measurements** collected from landslide monitoring stations.

### Input Features

- 🌡 Temperature
- 💧 Humidity
- 🌪 Pressure
- 🌧 Rainfall
- 🌱 Soil Moisture
- 📍 Accelerometer (AccX, AccY, AccZ)
- ☀ Light Intensity
- 🕒 Timestamp
- 🛰 Station ID

### Target Variables

- ➜ Displacement X (`dispx`)
- ➜ Displacement Y (`dispy`)
- ➜ Displacement Z (`dispz`)

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/raveena31/landslide-displacement-prediction.git

cd landslide-displacement-prediction
```

Create a virtual environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

For the Fuzzy Inference System

```bash
pip install scikit-fuzzy
```

For Graph Neural Networks

Install **PyTorch Geometric** according to your installed PyTorch version.

---

## 🚀 Training Models

### XGBoost

```bash
python main.py --model xgboost
```

### Multi-Layer Perceptron

```bash
python main.py --model mlp
```

### GRU

```bash
python main.py --model gru
```

### Graph Neural Network + GRU

```bash
python main.py --model gnn
```

### Fuzzy Inference System

```bash
python train_fis.py
```

---

## 🌐 Streamlit Web Application

Run

```bash
streamlit run app.py
```

The application allows users to enter sensor readings and predict landslide displacement using the trained XGBoost model.

---

## 📈 Evaluation Metrics

The following metrics are used:

- 📉 RMSE (Root Mean Squared Error)
- 📊 MAE (Mean Absolute Error)
- 🎯 R² Score

Higher R² and lower RMSE/MAE indicate better predictive performance.

---

## 🛠 Technologies Used

### Machine Learning

- XGBoost
- Scikit-learn

### Deep Learning

- PyTorch
- GRU
- Multi-Layer Perceptron

### Graph Learning

- PyTorch Geometric
- Graph Neural Networks

### Rule-Based AI

- Scikit-Fuzzy

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Streamlit

---

## 📌 Future Improvements

- 🔹 Transformer-based sequence models
- 🔹 Attention-enhanced GRU/LSTM
- 🔹 Real-time IoT sensor integration
- 🔹 Explainable AI (SHAP/LIME)
- 🔹 Early warning notification system
- 🔹 Cloud deployment using AWS

---

##  Author

**Yuvan Aadheraj K**

B.Tech Computer Science and Engineering

VIT Chennai

---

## ⭐ If you found this project useful, consider giving it a star!