import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download stopwords
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    text = re.sub(r"http\S+", "", text)                   # Remove URLs
    text = re.sub(r"[^a-zA-Z]", " ", text)                # Remove non-letters
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Load and prepare data
fake = pd.read_csv("C:\\Users\\sushm\\OneDrive\\ppts\\Fake.csv")
print(fake.columns)
fake['content'] = fake['text']
#fake['content'] = fake['title'] + " " + fake['text']
#fake['content'] = fake['title'] + " " + fake['text']
half = len(fake) // 2
fake_news = fake.iloc[:half].copy()
real_news = fake.iloc[half:half*2].copy()
fake_news['label'] = 1  # Fake
real_news['label'] = 0  # Pretend real

# Combine and shuffle
data = pd.concat([fake_news, real_news]).sample(frac=1).reset_index(drop=True)

# Clean the text
data['clean_text'] = data['content'].apply(clean_text)

# Feature extraction using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data['clean_text']).toarray()
y = data['label']

# Train the model
model = MultinomialNB()
model.fit(X, y)

# Streamlit interface
st.title('Fake News Detection')

st.write("""
    Enter the text you want to classify as Fake or Real.
    The model will predict whether the news is Fake or Real based on its content.
""")

# Input text box for user to enter news
user_input = st.text_area("Enter News Title and Text", "")

if user_input:
    # Clean user input and transform using TF-IDF
    cleaned_input = clean_text(user_input)
    input_vector = tfidf.transform([cleaned_input]).toarray()

    # Predict with the trained model
    prediction = model.predict(input_vector)
    label = "Fake" if prediction[0] == 1 else "Real"
    
    st.write(f"Prediction: {label}")
    
    # Display the model's decision-making process (optional)
    if st.checkbox("Show Model Details"):
        st.write(f"Input Text: {user_input}")
        st.write(f"Cleaned Text: {cleaned_input}")
        st.write(f"Prediction: {label}")
