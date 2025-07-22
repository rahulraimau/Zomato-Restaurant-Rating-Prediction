import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import re
import os

# Paths
MODEL_PATH = r"C:\Users\DELL\PycharmProjects\zomato_rating\restaurant_rating_tuned_xgboost_model.joblib"
DATA_PATH =r"C:\Users\DELL\PycharmProjects\zomato_rating\Zomato Restaurant reviews.csv"

# ---------- Page Setup ----------
st.set_page_config(page_title="Zomato Review Rating Predictor", layout="centered")
st.title("🍽️ Zomato Restaurant Rating Predictor")
st.write("Predict restaurant review ratings using a tuned XGBoost model.")

# ---------- Load Dataset ----------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# Show available columns
st.write("📌 Available Columns:", df.columns.tolist())

# Detect review column
REVIEW_COL = "Review" if "Review" in df.columns else df.columns[0]

if st.checkbox("Show sample reviews"):
    st.dataframe(df[[REVIEW_COL, 'Rating']].dropna().head())

# ---------- Clean Text ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

# ---------- Build Vectorizer ----------
@st.cache_resource
def build_vectorizer(texts):
    texts_cleaned = texts.apply(clean_text)
    vectorizer = TfidfVectorizer(max_features=30, stop_words='english')
    vectorizer.fit(texts_cleaned)
    return vectorizer

vectorizer = build_vectorizer(df[REVIEW_COL].astype(str))
model = joblib.load(MODEL_PATH)

# ---------- WordCloud ----------
if st.button("Generate WordCloud"):
    text = " ".join(df[REVIEW_COL].dropna().astype(str))
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

# ---------- Rating Distribution ----------
if st.button("Show Rating Distribution"):
    st.bar_chart(df['Rating'].value_counts().sort_index())

# ---------- Review Input ----------
st.subheader("📝 Enter a Review for Rating Prediction")
user_input = st.text_area("Type or paste a restaurant review:")

if st.button("Predict Rating"):
    if user_input.strip():
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        st.success(f"⭐ Predicted Rating: **{round(prediction, 1)} / 5**")
    else:
        st.warning("Please enter a valid review.")

# ---------- Footer ----------
st.markdown("---")
st.caption("🔍 Built by Rahul Rai | Zomato NLP + XGBoost Model")