import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

st.title('Food Delivery time prediction')

model = pickle.load(open('model.sav', 'rb'))

age = st.number_input("Delivery Person Age", min_value=18, max_value=60, value=25)
rating = st.number_input("Delivery Person Rating", min_value=0.0, max_value=5.0, value=4.0,step = 1.0)
distance = st.number_input("Distance (km)", min_value=0.0, value=5.0, step = 0.5)



if st.button("Predict"):
   
        input_data = pd.DataFrame({
            'age': [age],
            'rating': [rating],
            'distance': [distance],
            
        })

           
        prediction = model.predict(input_data)
        st.success(f"Predicted delivery time: {prediction[0][0]:.2f} minutes")


        
