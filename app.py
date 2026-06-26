app.py
import streamlit as st

st.title("🌿 Skincare Recommendation System")

st.write("Welcome to my Machine Learning Project!")

skin_type = st.selectbox(
    "Select your skin type",
    ["Oily", "Dry", "Combination", "Normal", "Sensitive"]
)

st.write("You selected:", skin_type)

import joblib

model = joblib.load("skincare_model.pkl")
encoders = joblib.load("label_encoders.pkl")

for col in ["skin_type", "climate", "humidity_level"]:
    user_input[col] = encoders[col].transform(user_input[col])

prediction = model.predict(user_input)

cleanser = encoders["cleanser_type"].inverse_transform([prediction[0][0]])[0]
moisturizer = encoders["moisturizer_type"].inverse_transform([prediction[0][1]])[0]
sunscreen = encoders["sunscreen_type"].inverse_transform([prediction[0][2]])[0]

st.success("Your Recommended Skincare Routine")

st.write("🧼 Cleanser:", cleanser)
st.write("💧 Moisturizer:", moisturizer)
st.write("☀ Sunscreen:", sunscreen)

