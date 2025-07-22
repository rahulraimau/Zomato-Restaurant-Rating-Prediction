import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import re
import os

# ---------- Constants ----------
MODEL_PATH = "restaurant_rating_tuned_xgboost_model.joblib"
DATA_PATH = "Zomato Restaurant reviews.csv"

# ---------- Page Config ----------
st.set_page_config(page_title="Zomato Review Rating Predictor", layout="centered")
st.title("🍽️ Zomato Restaurant Rating Predictor")
st.write("Predict review ratings using your model!")

# ---------- Load Data ----------
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

if os.path.exists(DATA_PATH):
    df = load_data(DATA_PATH)
    st.success("✅ Data loaded successfully.")
else:
    st.error("❌ Data file not found. Please upload Zomato Restaurant reviews.csv.")
    st.stop()

# ---------- Determine Review Column ----------
REVIEW_COL = "Review" if "Review" in df.columns else df.columns[0]

# ---------- Display Sample ----------
if st.checkbox("📊 Show sample data"):
    st.dataframe(df[[REVIEW_COL, 'Rating']].dropna().head())

# ---------- Load Model ----------
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    st.success("✅ Model loaded successfully.")
else:
    st.warning("⚠️ Model file not found. Upload `restaurant_rating_tuned_xgboost_model.joblib` to root.")
    st.stop()

# ---------- Vectorizer ----------
def build_vectorizer(text_data):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    vectorizer.fit(text_data)
    return vectorizer

vectorizer = build_vectorizer(df[REVIEW_COL].astype(str))

# ---------- WordCloud ----------
if st.checkbox("☁️ Show WordCloud"):
    text = " ".join(df[REVIEW_COL].astype(str).dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# ---------- Predict ----------
st.subheader("🔍 Predict Rating from Review")
user_input = st.text_area("Enter your review", "")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned = re.sub(r"[^a-zA-Z\s]", "", user_input.lower())
        transformed = vectorizer.transform([cleaned])
        prediction = model.predict(transformed)[0]
        st.success(f"⭐ Predicted Rating: **{round(prediction, 2)}**")
# ---------- Footer ----------
st.markdown("---")
st.caption("🔍 Built by Rahul Rai | Zomato NLP + XGBoost Model")
