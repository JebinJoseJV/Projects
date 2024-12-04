import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))


lg = pickle.load(open('logistic_regresion.pkl','rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl','rb'))
lb = pickle.load(open('label_encoder.pkl','rb'))



def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)


def predict_emotion(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf_vectorizer.transform([cleaned_text])
    predicted_label = lg.predict(input_vectorized)[0]
    predicted_emotion = lb.inverse_transform([predicted_label])[0]
    label = np.max(lg.predict_proba(input_vectorized))
    return predicted_emotion, label



st.title("Emotion Detector")


st.write("Enter text to detect emotions")

user_input = st.text_area("Your Text", height=100)

if st.button("Predict Emotion"):
    if user_input:
        predicted_emotion, probability = predict_emotion(user_input)

        st.write("Predicted Emotion:", predicted_emotion)
        st.write("Probability:", probability)
    else:
        st.warning("Please enter some text.")

