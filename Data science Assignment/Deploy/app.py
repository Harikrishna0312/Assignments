import streamlit as st
import pandas as pd
import pickle


with open("diabetes_model.pkl", "rb") as file:
    model = pickle.load(file)

features = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")

st.title(" Diabetes Prediction App")
st.write("Enter the patient details below and click **Predict**.")


inputs = {}

inputs["Pregnancies"] = st.number_input("Pregnancies", min_value=0, step=1)
inputs["Glucose"] = st.number_input("Glucose Level", min_value=0)
inputs["BloodPressure"] = st.number_input("Blood Pressure", min_value=0)
inputs["SkinThickness"] = st.number_input("Skin Thickness", min_value=0)
inputs["Insulin"] = st.number_input("Insulin Level", min_value=0)
inputs["BMI"] = st.number_input("BMI", min_value=0.0, format="%.2f")
inputs["DiabetesPedigreeFunction"] = st.number_input(
    "Diabetes Pedigree Function", min_value=0.0, format="%.3f"
)
inputs["Age"] = st.number_input("Age", min_value=0, step=1)


input_df = pd.DataFrame([inputs])

if st.button(" Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f" **Diabetic**\n\nProbability: **{probability:.2f}**")
    else:
        st.success(f" **Not Diabetic**\n\nProbability: **{probability:.2f}**")
