import streamlit as st
import pandas as pd
import joblib


model =joblib.load('regression_heart.pkl')
scaler =joblib.load("scaler.pkl")
expected_columns= joblib.load("columns.pkl")


st.title("heart stroke prediction by deepak")
st.markdown("this is a simple web app to predict heart stroke using machine learning model")
age=st.slider("Age",18,100,40)
sex=st.selectbox("SEX", options=["M", "F"])
chest_pain=st.selectbox("Chest Pain Type", options=["ATA", "NAP", "TA", "ASY"])
resting_bp=st.number_input("Resting Blood Pressure(mm,hg)", 80,200,120)
cholesterol=st.number_input("Cholesterol(mg/dl)", 100,600,200)
fasting_bs=st.selectbox("Fasting Blood Sugar >120 mg/dl", options=[0,1])
resting_ecg=st.selectbox("Resting ECG", options=["Normal", "ST", "LVH"])
max_hr=st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina=st.selectbox("Exercise Induced Angina", options=["Y", "N"])
oldpeak=st.slider("oldpeak(ST Depression)", 0.0, 6.0, 1.0)
st_slope=st.selectbox("ST Slope", options=["Up", "Flat", "Down"])



if st.button("predict"):
    raw_input={
        "Age": age,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG_" + resting_ecg: 1,
        "MaxHR": max_hr,
        "ExerciseAngina_" + exercise_angina: 1,
        "Oldpeak" : oldpeak,
        "ST_Slope_" + st_slope :1
    }
    input_df = pd.DataFrame([raw_input]) 

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]


    if prediction == 1:
        st.error("The model predicts that the patient is at risk of having a heart stroke.")
    else:
        st.success("The model predicts that the patient is not at risk of having a heart stroke.")