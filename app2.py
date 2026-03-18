import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==============================
# LOAD FILES
# ==============================
models = pickle.load(open("income_models.pkl", "rb"))
columns = pickle.load(open("income_columns.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Load original dataset (for dropdown values)
df = pd.read_csv("income_evaluation.csv")

# Clean columns
df.columns = df.columns.str.strip().str.replace('-', '_').str.replace(' ', '_')

# Clean object values
for col in df.select_dtypes(include='object'):
    df[col] = df[col].str.strip()

# ==============================
# UI
# ==============================
st.set_page_config(page_title="Income Predictor", layout="wide")

st.title("💰 Income Classification App")
st.write("Predict whether income is **>50K or <=50K**")

# ==============================
# MODEL SELECTION
# ==============================
model_name = st.selectbox("Select Model", list(models.keys()))
model = models[model_name]

st.markdown("---")

# ==============================
# INPUT SECTION
# ==============================
st.subheader("📊 Enter Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", int(df['age'].min()), int(df['age'].max()), 30)
    fnlwgt = st.slider("Final Weight", int(df['fnlwgt'].min()), int(df['fnlwgt'].max()), 100000)
    education_num = st.slider("Education Number", int(df['education_num'].min()), int(df['education_num'].max()), 10)
    hours_per_week = st.slider("Hours per Week", int(df['hours_per_week'].min()), int(df['hours_per_week'].max()), 40)

with col2:
    workclass = st.selectbox("Workclass", sorted(df['workclass'].unique()))
    education = st.selectbox("Education", sorted(df['education'].unique()))
    marital_status = st.selectbox("Marital Status", sorted(df['marital_status'].unique()))
    occupation = st.selectbox("Occupation", sorted(df['occupation'].unique()))
    relationship = st.selectbox("Relationship", sorted(df['relationship'].unique()))
    race = st.selectbox("Race", sorted(df['race'].unique()))
    sex = st.selectbox("Sex", sorted(df['sex'].unique()))
    native_country = st.selectbox("Native Country", sorted(df['native_country'].unique()))

# ==============================
# CREATE INPUT DATAFRAME
# ==============================
input_dict = {
    'age': age,
    'workclass': workclass,
    'fnlwgt': fnlwgt,
    'education': education,
    'education_num': education_num,
    'marital_status': marital_status,
    'occupation': occupation,
    'relationship': relationship,
    'race': race,
    'sex': sex,
    'capital_gain': 0,
    'capital_loss': 0,
    'hours_per_week': hours_per_week,
    'native_country': native_country
}

input_df = pd.DataFrame([input_dict])

# ==============================
# ENCODING (IMPORTANT)
# ==============================
input_encoded = pd.get_dummies(input_df)

# Match training columns
input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

# ==============================
# PREDICTION
# ==============================
st.markdown("---")

if st.button("🚀 Predict Income"):

    input_scaled = scaler.transform(input_encoded)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("💵 Income: >50K (High Income)")
    else:
        st.info("💼 Income: <=50K (Low Income)")