import logging
import streamlit as st
import numpy as np
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
import pickle

# Misalkan model dan tfidf_vectorizer sudah dilatih dan disimpan dalam file .pkl
# Load model dan vectorizer

def load_model_and_vectorizer():
    try:
        model = joblib.load("gradient_boost_model.pkl")
    except Exception as e:
        print(f"Error loading model file: {e}")
        return None, None

    try:
        tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    except Exception as e:
        print(f"Error loading vectorizer file: {e}")
        return None, None
    
    return model, tfidf_vectorizer

def run_ml_app(text, model, tfidf_vectorizer):
    # Transformasi teks menggunakan TF-IDF Vectorizer
    transformed_text = tfidf_vectorizer.transform([text])
    # Prediksi menggunakan model
    prediction = model.predict(transformed_text)
    return prediction[0]

# Membuat aplikasi Streamlit
model, tfidf_vectorizer = load_model_and_vectorizer()
st.title("Aplikasi Prediksi Sentiment Cyberbullying")

# Input teks dari pengguna
user_input = st.text_area("Masukkan teks tweet Anda di sini:")

if st.button("Prediksi"):
    result = run_ml_app(user_input, model, tfidf_vectorizer)
    sentiments = ["Religion", "Age", "Ethnicity", "Gender", "Not Cyberbullying"]
    st.write(f"Sentiment: {sentiments[result]}")
