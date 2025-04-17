import streamlit as st
import joblib
import pandas as pd

# --- Load the trained model and training columns ---
model = joblib.load("xgboost_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("🎓 Student Final Score (G3) Prediction")
st.markdown("Fill in the details below to predict the student's final G3 score.")

# --- User Inputs ---
G1 = st.number_input("G1 Score (0-20)", min_value=0, max_value=20, step=1)
G2 = st.number_input("G2 Score (0-20)", min_value=0, max_value=20, step=1)
school = st.selectbox("School", options=["GP", "MS"])
sex = st.selectbox("Sex", options=["F", "M"])

# --- Predict button ---
if st.button("Predict G3 Score"):
    # Build input data
    input_data = pd.DataFrame([{
        "G1": G1,
        "G2": G2,
        "school": school,
        "sex": sex
    }])

    # Add average feature and drop G1/G2
    input_data["G1_G2_avg"] = (G1 + G2) / 2
    input_data.drop(columns=["G1", "G2"], inplace=True)

    # One-hot encode categorical variables
    input_data = pd.get_dummies(input_data)

    # Match training column order
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    # Uncomment to debug what goes into the model
    # st.write("Model input preview:", input_data)

    # Predict
    prediction = model.predict(input_data)[0]

    # Option 1: Rounded to 2 decimal places
    st.success(f"🎯 Predicted Final G3 Score: **{round(prediction, 2)}**")

    # Option 2: No decimal
    # st.success(f"🎯 Predicted Final G3 Score: **{int(round(prediction))}**")
