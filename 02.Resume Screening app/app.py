# prompt: create streamlit app for the same, app should take input as pdf or word file

import streamlit as st
import pickle
import re
import numpy as np

# Load the saved model and vectorizer
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
svc_model = pickle.load(open('clf.pkl', 'rb'))
le = pickle.load(open("encoder.pkl", 'rb'))

def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)  
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text])
    vectorized_text = vectorized_text.toarray()
    predicted_category = svc_model.predict(vectorized_text)
    predicted_category_name = le.inverse_transform(predicted_category)
    return predicted_category_name[0]

# Streamlit app
st.title("Resume Category Prediction")

uploaded_file = st.file_uploader("Upload a PDF or Word file", type=["pdf", "docx"])

if uploaded_file is not None:
    try:
        # Assuming you have a function to extract text from PDF/Word files
        # Replace this with your actual text extraction logic
        import docx2txt
        if uploaded_file.type == "application/pdf":
            # Handle PDF file
            # For example, using PyPDF2 library:
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page].extract_text()

        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # Handle .docx file
            text = docx2txt.process(uploaded_file)

        else:
             st.write("Unsupported file type.")
             text = ""

        # Predict category
        if text:
            predicted_category = pred(text)
            st.write(f"Predicted Category: {predicted_category}")
        
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.write("Please upload a file.")