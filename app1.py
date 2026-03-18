import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and columns
models = pickle.load(open("all_models.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# Load dataset
df = pd.read_csv("Life Expectancy Data.csv")
df.columns = df.columns.str.strip().str.replace(' ', '_')

st.title("🌍 Life Expectancy Prediction App")

# 🔹 Model selection
model_name = st.selectbox("Select Model", list(models.keys()))
model = models[model_name]

st.write("### 📌 Enter the details:")

input_data = []

for col in columns:

    # Skip target if present
    if col == "Life_expectancy":
        continue

    # 🔹 If numeric column
    if df[col].dtype != 'object':
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())

        val = st.number_input(
            f"{col} (Range: {round(min_val,2)} - {round(max_val,2)})",
            value=round(mean_val, 2)
        )

    # 🔹 If categorical column (like Status)
    else:
        options = df[col].unique().tolist()
        selected = st.selectbox(f"{col}", options)

        # Convert to numeric (same as training)
        if col == "Status":
            val = 1 if selected == "Developed" else 0
        else:
            val = 0  # fallback

    input_data.append(val)

# 🔹 Prediction
if st.button("Predict"):
    input_array = np.array([input_data])
    prediction = model.predict(input_array)

    st.success(f"🌟 Predicted Life Expectancy: {round(prediction[0], 2)} years")