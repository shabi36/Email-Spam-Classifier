import pickle

# Load the vectorizer and model
with open("model_sms_new1.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("model_sms_new1.pkl", "rb") as f:
    model = pickle.load(f)

import pickle
import streamlit as st
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

nltk.download("punkt")
nltk.download("stopwords")

# Load the vectorizer and model
with open("vectorizer_sms.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("model_sms.pkl", "rb") as f:
    model = pickle.load(f)

st.title("SMS SPAM CLASSIFIER")

input_msg = st.text_area("Enter The Message")

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y.copy()
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y.copy()
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

if st.button("Predict"):
    # 1) Preprocess
    transformed_sms = transform_text(input_msg)

    # 2) Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3) Predict
    result = model.predict(vector_input)[0]

    # 4) Display Result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
