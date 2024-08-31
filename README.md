# Heart-disease-risk-prediction-project
Create a user-friendly interface for inputting health data and implement a model  for risk assessment.
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load and preprocess the dataset
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    data = pd.read_csv(url, names=column_names)
    data = data.replace('?', pd.NA).dropna()
    data = data.apply(pd.to_numeric)
    return data

# Train the model
@st.cache_data
def train_model(data):
    X = data.drop('target', axis=1)
    y = data['target'].apply(lambda x: 1 if x > 0 else 0)  # Binary classification: 1 (heart disease), 0 (no heart disease)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

# Load data and train model
data = load_data()
model, accuracy = train_model(data)

# Streamlit UI setup
st.title("Heart Disease Risk Prediction")
st.write(f"Model Accuracy: {accuracy:.2f}")

# User input function
def user_input_features():
    age = st.sidebar.slider('Age', 20, 80, 50)
    sex = st.sidebar.selectbox('Sex', (0, 1))  # 0 = female, 1 = male
    cp = st.sidebar.selectbox('Chest Pain Type (CP)', (0, 1, 2, 3))
    trestbps = st.sidebar.slider('Resting Blood Pressure (trestbps)', 90, 200, 120)
    chol = st.sidebar.slider('Serum Cholesterol in mg/dl (chol)', 100, 400, 200)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', (0, 1))
    restecg = st.sidebar.selectbox('Resting ECG Results (restecg)', (0, 1, 2))
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved (thalach)', 60, 200, 150)
    exang = st.sidebar.selectbox('Exercise-Induced Angina (exang)', (0, 1))
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise (oldpeak)', 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment (slope)', (0, 1, 2))
    ca = st.sidebar.slider('Number of Major Vessels (ca) Colored by Fluoroscopy', 0, 3, 0)
    thal = st.sidebar.selectbox('Thalassemia (thal)', (1, 2, 3))
    
    features = {'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
                'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
                'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal}
    return pd.DataFrame(features, index=[0])

input_df = user_input_features()

# Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Prediction')
heart_disease_risk = 'High Risk of Heart Disease' if prediction[0] == 1 else 'Low Risk of Heart Disease'
st.write(heart_disease_risk)

st.subheader('Prediction Probability')
st.write(f"Probability of No Heart Disease: {prediction_proba[0][0]:.2f}")
st.write(f"Probability of Heart Disease: {prediction_proba[0][1]:.2f}")
