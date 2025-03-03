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

model = pickle.load(open("model_mnb_sms2.pkl", "rb"))

# Streamlit App Title
st.title("SMS SPAM CLASSIFIER")

# User Input
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

    return " ".join(y)  # return as a string

# Prediction on button click
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
