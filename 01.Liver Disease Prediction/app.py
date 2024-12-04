import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


model = pickle.load(open('logistic_regression.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
lb = pickle.load(open('label_encoder.pkl', 'rb'))


st.title("Liver Disease Prediction App")


st.header("Enter patient information:")
age = st.number_input("Age", min_value=0, max_value=100, value=40)
gender = st.selectbox("Gender", ["Male", "Female"])
total_bilirubin = st.number_input("Total Bilirubin", min_value=0.0, value=0.8)
direct_bilirubin = st.number_input("Direct Bilirubin", min_value=0.0, value=0.2)
alkaline_phosphotase = st.number_input("Alkaline Phosphotase", min_value=0, value=180)
alamine_aminotransferase = st.number_input("Alamine Aminotransferase", min_value=0, value=25)
aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase", min_value=0, value=35)
total_proteins = st.number_input("Total Proteins", min_value=0.0, value=7.0)
albumin = st.number_input("Albumin", min_value=0.0, value=4.0)
albumin_and_globulin_ratio = st.number_input("Albumin and Globulin Ratio", min_value=0.0, value=1.0)



if st.button("Predict"):
    
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Total_Bilirubin': [total_bilirubin],
        'Direct_Bilirubin': [direct_bilirubin],
        'Alkaline_Phosphotase': [alkaline_phosphotase],
        'Alamine_Aminotransferase': [alamine_aminotransferase],
        'Aspartate_Aminotransferase': [aspartate_aminotransferase],
        'Total_Protiens': [total_proteins],
        'Albumin': [albumin],
        'Albumin_and_Globulin_Ratio': [albumin_and_globulin_ratio]
    })

    input_data['Gender'] = lb.transform(input_data['Gender'])
    input_data_scaled = scaler.transform(input_data)

    
    prediction = model.predict(input_data_scaled)

    
    if prediction[0] == 0:
      st.write("Prediction: Patient has liver disease.")
    else:
      st.write("Prediction: Patient does not have liver disease.")
