import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Movie Rating Predictor")

st.title("🎬 Movie Average Rating Prediction")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "netflix_model.pkl")
model = joblib.load(MODEL_PATH)


st.write("Movie details fill pannunga 👇")

movie_title = st.text_input("Movie Title")
genre = st.text_input("Genre")

release_year = st.number_input(
    "Release Year", min_value=1900, max_value=2100, value=2020
)

num_reviews = st.number_input(
    "Number of Reviews", min_value=0, value=100
)

review_highlights = st.text_area("Review Highlights")

life_change_minute = st.text_input(
    "Minute of Life-Changing Insight"
)

how_discovered = st.text_input("How Discovered")

meaningful_advice = st.text_area("Meaningful Advice Taken")

suggested_percent = st.text_input(
    "Suggested to Friends/Family (Y/N %)"
)

# ===== PREDICTION =====
if st.button("Predict Average Rating"):
    input_df = pd.DataFrame([{
        "Movie Title": movie_title,
        "Genre": genre,
        "Release Year": release_year,
        "Number of Reviews": num_reviews,
        "Review Highlights": review_highlights,
        "Minute of Life-Changing Insight": life_change_minute,
        "How Discovered": how_discovered,
        "Meaningful Advice Taken": meaningful_advice,
        "Suggested to Friends/Family (Y/N %)": suggested_percent
    }])

    prediction = model.predict(input_df)

    st.success(f"⭐ Predicted Average Rating: {prediction[0]:.2f}")
    st.balloons()
