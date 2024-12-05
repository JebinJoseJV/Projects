import pandas as pd
import spacy
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import streamlit as st


nlp = spacy.load('en_core_web_sm')
vocab_size = 10000
sentence_length = 20

def preprocess(text):
    doc = nlp(text)
    lemmalist = [word.lemma_ for word in doc]
    lemma = ' '.join(lemmalist)
    doc = nlp(lemma)
    no_stopwords_list = [word.text for word in doc if not word.is_stop]
    no_stopwords = ' '.join(no_stopwords_list)
    return no_stopwords


# Load the model and other necessary objects
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Model files not found.")
    exit()

# Streamlit app
st.title("Hate Speech Detection App")

input_text = st.text_input("Enter a text:")

if st.button("Predict"):
    if input_text:

        processed_text = preprocess(input_text)
        one_hot_representation = one_hot(processed_text, vocab_size)
        embedded_tweet = pad_sequences([one_hot_representation], padding='post', maxlen=sentence_length)
        prediction = np.argmax(model.predict(embedded_tweet), axis=-1)
        class_mapping = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}

        st.write(f"Prediction: {class_mapping.get(prediction[0], 'Unknown')}")
    else:
        st.write("Please enter a tweet.")