import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import joblib

# Set up Streamlit page
st.set_page_config(page_title="Zomato Rating Predictor", layout="centered")
st.title("🍽️ Zomato Restaurant Rating Predictor")
st.write("Predict review ratings using your model!")

# Load data
@st.cache_data
def load_data():
    try:
        return pd.read_csv("Zomato Restaurant reviews.csv")  # ✅ correct: file name as string
    except FileNotFoundError:
        st.error("CSV file not found. Please make sure 'Zomato Restaurant reviews.csv' is uploaded.")
        return pd.DataFrame()

df = load_data()

# Show sample
if not df.empty:
    st.subheader("Sample Data")
    st.write(df.head())
else:
    st.stop()

# Load model
@st.cache_resource
def load_model():
    try:
        return joblib.load("xgboost_model.joblib")
    except:
        st.warning("Model file not found. Upload `xgboost_model.joblib` to root.")
        return None

model = load_model()

# Predict from review input
st.subheader("🔍 Predict Rating from Review")
review_input = st.text_area("Enter your review")

if st.button("Predict Rating") and model:
    # Minimal cleaning
    cleaned = review_input.lower()
    tfidf = joblib.load("tfidf_vectorizer.joblib")
    vec = tfidf.transform([cleaned])
    prediction = model.predict(vec)
    st.success(f"Predicted Rating: ⭐ {round(prediction[0], 2)}")

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
