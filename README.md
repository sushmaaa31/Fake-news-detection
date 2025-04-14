# 📰 Fake News Detection

This Streamlit app helps classify whether a news article is **Fake** or **Real** using text analysis and machine learning.

---

## 🚀 Features

- 📂 Upload & process the dataset (`Fake.csv`)
- 🧹 Clean text (remove URLs, punctuation, stopwords)
- ✨ Extract features using TF-IDF
- 🤖 Classify news using **Multinomial Naive Bayes**
- 📊 Optional display of processed input and prediction steps

---

## 🖼 Sample Output

- 📝 Text input for classification
- 🧼 Cleaned version of the input shown
- 🔍 Model output: **Fake** or **Real**
- ✅ Checkbox to view detailed model steps

---

## 🧰 Tech Stack

- 🐍 Python
- 🌐 Streamlit
- 📊 pandas, numpy, matplotlib, seaborn
- 🧠 scikit-learn (TF-IDF, Naive Bayes)
- 🔤 nltk for stopwords and text processing

---

## 🔧 How to Run

### 1. Clone this repo

```bash
git clone https://github.com/sushmaaa31/fake-news-detector.git
cd fake-news-detector
###2. Install dependencies
pip install -r requirements.txt
3. Run the app
streamlit run code.py
---
fake-news-detector/
├── code.py             # Main Streamlit app
├── Fake.csv            # Dataset
├── requirements.txt    # Required packages (optional)
└── README.md           # Project documentation
