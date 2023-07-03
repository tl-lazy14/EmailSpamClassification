from bs4 import BeautifulSoup
import numpy as np
import streamlit as st
import pickle
import re
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import email

st.set_page_config(
    page_title="Email Spam Classification",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    body {
        background-color: #8DA8E0; /* Màu nền */
        color: #000000; /* Màu chữ */
    }
    </style>
    """,
    unsafe_allow_html=True
)

ps = PorterStemmer()

def html_to_text(email) -> str:
    try:
        soup = BeautifulSoup(email.get_payload(), "html.parser")
        plain = soup.text.replace("=\n", "")
        plain = re.sub(r"\s+", " ", plain)
        return plain.strip()
    except:
        return "nothing"

def email_to_text(email):
    text_content = ""
    for part in email.walk():
        part_content_type = part.get_content_type()
        if part_content_type not in ['text/plain', 'text/html']:
            continue
        if part_content_type == 'text/plain':
            text_content += part.get_payload()
        else:
            text_content += html_to_text(part)
    return text_content

def transform_content(text):
    text = email.message_from_bytes(text.encode())
    text = email_to_text(text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = nltk.word_tokenize(text)
    
    y = []
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)

def transform_subject(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = nltk.word_tokenize(text)
    
    y = []
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.markdown(
    """
    <style>
    .custom-title {
        margin-top: -90px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="custom-title">Email Spam Classification</h1>', unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .input-label {
        font-size: 20px;
        margin-top: -30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="input-label">Enter the email</p>', unsafe_allow_html=True)
input_subject = st.text_input("", placeholder="Tiêu đề")
input_content = st.text_area("", height=500)

if st.button('Predict'):
    # Kiểm tra xem đã nhập input subject hay chưa
    if not input_subject:
        st.header("Bạn chưa nhập tiêu đề email")
    # Kiểm tra xem đã nhập input content hay chưa
    elif not input_content:
        st.header("Bạn chưa nhập nội dung email")
    else:
        # 1. preprocess
        transformed_subject = transform_subject(input_subject)
        transformed_content = transform_content(input_content)
        # 2. vectorize
        vector_subject = tfidf.transform([transformed_subject]).toarray()
        vector_content = tfidf.transform([transformed_content]).toarray()
        vector_input = np.concatenate((vector_subject, vector_content), axis=1)
        # 3. predict
        result = model.predict(vector_input)[0]
        # 4. Display
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")