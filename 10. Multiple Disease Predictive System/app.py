import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd

liver_model = pickle.load(open('ld_logistic_regression.pkl', 'rb'))
liver_scaler = pickle.load(open('ld_scaler.pkl', 'rb'))
liver_lb = pickle.load(open('ld_label_encoder.pkl', 'rb'))

diabetes_model=pickle.load(open('d_model.pkl','rb'))
diabetes_scaler=pickle.load(open('d_scaler.pkl','rb'))

with st.sidebar:
    selected = option_menu('Multiple Disease Prediciton',
                           ['Liver Disease Prediction',
                            'Diabetes Prediction'],default_index=0)

if (selected == 'Liver Disease Prediction'):
    st.title('Liver Disease Prediction')
    st.header("Enter patient information:")
    col1,col2,col3 = st.columns(3)
 
    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, value=40)
        gender = st.selectbox("Gender", ["Male", "Female"])
        total_bilirubin = st.number_input("Total Bilirubin", min_value=0.0, value=0.8)
        direct_bilirubin = st.number_input("Direct Bilirubin", min_value=0.0, value=0.2)
    with col2:
        alkaline_phosphotase = st.number_input("Alkaline Phosphotase", min_value=0, value=180)
        alamine_aminotransferase = st.number_input("Alamine Aminotransferase", min_value=0, value=25)
        aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase", min_value=0, value=35)
    with col3:
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

        input_data['Gender'] = liver_lb.transform(input_data['Gender'])
        input_data_scaled = liver_scaler.transform(input_data)

    
        prediction = liver_model.predict(input_data_scaled)

    
        if prediction[0] == 0:
            st.write("Prediction: Patient has liver disease.")
        else:
            st.write("Prediction: Patient does not have liver disease.")

if (selected == 'Diabetes Prediction'):
    st.title('Diabetes Prediction')

    

    st.header('Enter patient information')

    c1,c2,c3 = st.columns(3)

    with c1:
        Pregnancies = st.number_input('Number of Pregnancies',step=1,min_value=0,max_value=20)
        Glucose = st.number_input('Glucose',min_value=0.0,max_value=200.0)
        BloodPressure = st.number_input('BloodPressure',min_value=0,max_value=150)

    with c2:
        SkinThickness = st.number_input('SkinThickness',min_value=0.0,max_value=100.0)
        Insulin = st.number_input('Insulin',min_value=0.0,max_value=1000.0)
        BMI = st.number_input('BMI',min_value=0.0,max_value=70.0)
    
    with c3:
        DiabetesPedigreeFunction = st.number_input('DiabetesPedigreeFunction',min_value=0.0,max_value=3.0)
        Age = st.number_input('Age',min_value=21,step=1)



    if st.button("Predict"):
        input_data = pd.DataFrame({
            'Pregnancies':[Pregnancies],
            'Glucose':[Glucose],
            'BloodPressure':[BloodPressure],
            'SkinThickness':[SkinThickness],
            'Insulin':[Insulin],
            'BMI':[BMI],
            'DiabetesPedigreeFunction':[DiabetesPedigreeFunction],
            'Age':[Age]
        })

        input_data_scaled = diabetes_scaler.transform(input_data)
        prediction = diabetes_model.predict(input_data_scaled)

        if prediction[0] == 0:
            st.write("The patient has diabetes")
        else:
            st.write("The patient does not have diabetes")