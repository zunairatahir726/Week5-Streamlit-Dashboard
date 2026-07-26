import streamlit as st
import pandas as pd
import joblib
import re
import string

# ----------------------------
# PAGE CONFIG
# ----------------------------

st.set_page_config(
    page_title="Amazon Reviews Sentiment Analysis",
    page_icon="🛍️",
    layout="wide"
)

# ----------------------------
# LOAD MODEL & DATA
# ----------------------------

model = joblib.load("best_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
df = pd.read_csv("amazon_reviews.csv")

# ----------------------------
# SIMPLE TEXT CLEANING
# ----------------------------

STOP_WORDS = {
"a","an","the","is","are","am","was","were","be","been","being",
"to","of","in","on","for","with","and","or","at","by","this","that",
"it","its","my","your","our","their","i","you","he","she","they",
"we","me","him","her","them","as","from","have","has","had"
}

def clean_text(text):

    if pd.isna(text):
        return ""

    text = text.lower()

    text = re.sub(r"http\\S+", "", text)

    text = text.translate(str.maketrans("", "", string.punctuation))

    words = text.split()

    words = [word for word in words if word not in STOP_WORDS]

    return " ".join(words)

# ----------------------------
# SIDEBAR
# ----------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a Page",
    [
        "🏠 Home",
        "📊 Data Overview",
        "😊 Sentiment Predictor"
    ]
)

# ----------------------------
# HOME PAGE
# ----------------------------

if page == "🏠 Home":

    st.title("🛍️ Amazon Reviews Sentiment Analysis Dashboard")

    st.markdown("""
## Welcome

This dashboard predicts whether an Amazon customer review is **Positive** or **Negative** using Machine Learning.

### Dataset
- Amazon Alexa Reviews Dataset

### Models Used
- Multinomial Naive Bayes
- Logistic Regression

Use the sidebar to navigate through the dashboard.
""")
 # ======================================
# DATA OVERVIEW
# ======================================

elif page == "📊 Data Overview":

    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    st.title("📊 Data Overview")

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("First Five Records")
    st.dataframe(df.head())

    st.subheader("Feedback Distribution")
    st.bar_chart(df["feedback"].value_counts())

    # Positive Word Cloud
    st.subheader("Positive Reviews Word Cloud")

    positive_text = " ".join(
        df[df["feedback"] == 1]["verified_reviews"].fillna("").astype(str)
    )

    positive_wc = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(positive_text)

    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(positive_wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    # Negative Word Cloud
    st.subheader("Negative Reviews Word Cloud")

    negative_text = " ".join(
        df[df["feedback"] == 0]["verified_reviews"].fillna("").astype(str)
    )

    negative_wc = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(negative_text)

    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(negative_wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
    # ======================================
# SENTIMENT PREDICTOR
# ======================================

elif page == "😊 Sentiment Predictor":

    st.title("😊 Sentiment Predictor")

    review = st.text_area("Enter an Amazon Review")

    if st.button("Predict"):

        if review.strip() == "":
            st.warning("Please enter a review.")

        else:

            cleaned_review = clean_text(review)

            review_vector = vectorizer.transform([cleaned_review])

            prediction = model.predict(review_vector)[0]

            confidence = model.predict_proba(review_vector).max()

            if prediction == 1:
                st.success("😊 Positive Review")
            else:
                st.error("☹️ Negative Review")

            st.write(f"### Confidence Score: {confidence:.2%}")