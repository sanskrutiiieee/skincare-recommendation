
import streamlit as st
import pandas as pd
import joblib

# ------------------------------
# Load model and encoders
# ------------------------------
model = joblib.load("skincare_model.pkl")
encoders = joblib.load("label_encoders.pkl")

# ------------------------------
# Page
# ------------------------------
st.set_page_config(
    page_title="Skincare Recommendation",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 AI Skincare Recommendation System")
st.write("Get personalized skincare recommendations based on your skin type and climate.")

# ------------------------------
# User Inputs
# ------------------------------

skin_type = st.selectbox(
    "✨ Skin Type",
    ["Oily", "Dry", "Combination", "Normal", "Sensitive"]
)

climate = st.selectbox(
    "🌍 Climate",
    [
        "Tropical",
        "Arid/Desert",
        "Temperate",
        "Cold/Nordic",
        "Humid Subtropical"
    ]
)

humidity = st.selectbox(
    "💧 Humidity Level",
    [
        "Low",
        "Medium",
        "High"
    ]
)

# ------------------------------
# Prediction
# ------------------------------

if st.button("Get Recommendation"):

    user_input = pd.DataFrame({
        "skin_type": [skin_type],
        "climate": [climate],
        "humidity_level": [humidity]
    })

    try:

        for col in user_input.columns:
            user_input[col] = encoders[col].transform(user_input[col])

        prediction = model.predict(user_input)

        cleanser = encoders["cleanser_type"].inverse_transform([prediction[0][0]])[0]
        moisturizer = encoders["moisturizer_type"].inverse_transform([prediction[0][1]])[0]
        sunscreen = encoders["sunscreen_type"].inverse_transform([prediction[0][2]])[0]

        st.success("Your Personalized Routine")

        st.write("🧼 Cleanser :", cleanser)
        st.write("💧 Moisturizer :", moisturizer)
        st.write("☀️ Sunscreen :", sunscreen)

        st.balloons()

    except Exception as e:
        st.error(e)