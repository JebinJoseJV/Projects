import streamlit as st
import pickle

st.title('Language detection app')

try:
    with open('model.pkl','rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl','rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    st.error("Files not found")
    st.stop()

input_text = st.text_input("Enter the text here")


if st.button('Detect language'):
    if input_text:
        try:
            data = vectorizer.transform([input_text])
            output = model.predict(data)
            st.write("Predicted language: ",output[0])
        except Exception as e:
            st.error(f"An error has occured: {e}")
    else:
        st.write("Please enter some text")

