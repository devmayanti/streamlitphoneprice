import streamlit as st
import pickle
import pandas as pd
import numpy as np
import joblib

# Load the machine learning model
model = joblib.load('best_model.pkl')

st.title("Phone Price Prediction App")

# Collect user input for each feature
conditions = st.selectbox("Condition", ["Condition: Fair", "Condition: Good", "Condition: Mint", 
                                        "Condition: New", "Condition: Very Good"])
name = st.text_input("Phone Model Name")
details = st.text_input("Details about the phone")
release_date = st.text_input("Release Date (e.g., 2020)")
camera = st.text_input("Rear Camera Resolution (MP)")
selfie = st.text_input("Selfie Camera Resolution (MP)")
sim_card = st.text_input("SIM Card Type (e.g., Single, Dual)")
sensors = st.text_input("Sensors (e.g., Fingerprint, Accelerometer)")
network = st.text_input("Network Support (e.g., 4G, 5G)")
storage = st.number_input("Storage Capacity (GB)", min_value=0)
present_price = st.number_input("Present Price (Rp)", min_value=0)

# Load the preprocessing dictionary
ordinal_encoder = np.load('ordinal_encoder.npy', allow_pickle=True).item()
freq_encoder = np.load('freq_encoder.npy', allow_pickle=True).item()

# Prepare the input data as a DataFrame for the model
input_data = pd.DataFrame({
    'Conditions': [conditions],
    'Name': [name],
    'Details': [details],
    'Release Date': [release_date],
    'Camera': [camera],
    'Selfie': [selfie],
    'SIM Card': [sim_card],
    'Sensors': [sensors],
    'Network': [network],
    'Storage': [storage],
    'Present price': [present_price]
})

# Preprocessing the input data
input_data['Conditions'] = input_data['Conditions'].map(ordinal_encoder)
input_data['Name'] = input_data['Name'].map(freq_encoder['Name'])
input_data['Details'] = input_data['Details'].map(freq_encoder['Details'])
input_data['Release Date'] = input_data['Release Date'].map(freq_encoder['Release Date'])
input_data['Camera'] = input_data['Camera'].map(freq_encoder['Camera'])
input_data['Selfie'] = input_data['Selfie'].map(freq_encoder['Selfie'])
input_data['SIM Card'] = input_data['SIM Card'].map(freq_encoder['SIM Card'])
input_data['Sensors'] = input_data['Sensors'].map(freq_encoder['Sensors'])
input_data['Network'] = input_data['Network'].map(freq_encoder['Network'])


# Make prediction
if st.button("Predict Price"):
    predicted_price = model.predict(input_data)[0]
    st.write(f"The predicted price for this phone is: Rp {predicted_price:.2f}")