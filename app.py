import streamlit as st
import pickle
import pandas as pd
import joblib
import numpy as np

# Load trained model
model = joblib.load('best_model.pkl')

st.title("Phone Price Prediction App")

# Collect user input (Tetap dalam bentuk kategorik)
conditions = st.selectbox("Condition", ["Condition: Fair", "Condition: Good", "Condition: Mint", 
                                        "Condition: New", "Condition: Very Good"])
name = st.text_input("Phone Model Name")
details = st.text_input("Details about the phone")
camera = st.text_input("Rear Camera Resolution (MP)")
selfie = st.text_input("Selfie Camera Resolution (MP)")
sim_card = st.text_input("SIM Card Type")  # ✅ Tetap text untuk fleksibilitas kategori
sensors = st.text_input("Sensors (e.g., Fingerprint, Accelerometer)")
network = st.text_input("Network Support")  # ✅ Tetap text
storage = st.text_input("Storage Capacity")  # ✅ Tetap text
present_price = st.number_input("Present Price (Rp)", min_value=0)

# Load One-Hot Encoder yang sudah dilatih
with open('preprocessor.pkl', 'rb') as file:
    one_hot_encoder = pickle.load(file)

# Pastikan input tetap dalam bentuk DataFrame
input_df = pd.DataFrame({
    'Conditions': [conditions],
    'Name': [name],
    'Details': [details],
    'Camera': [camera],
    'Selfie': [selfie],
    'SIM Card': [sim_card],
    'Sensors': [sensors],
    'Network': [network],
    'Storage': [storage],
    'Present price': [present_price]
})

# Terapkan One-Hot Encoding ke input Streamlit
encoded_features = one_hot_encoder.transform(input_df)  # ✅ Gunakan .toarray()

# Konversi hasil encoding ke DataFrame
encoded_df = pd.DataFrame(encoded_features)

# Tambahkan "Present price" karena tidak diubah
encoded_df['Present price'] = present_price

# Make prediction
if st.button("Predict Price"):
    predicted_price = model.predict(encoded_df)[0]
    st.write(f"The predicted price for this phone is: Rp {predicted_price:.2f}")
