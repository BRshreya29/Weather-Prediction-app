import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('weather_predictor.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to predict weather
def predict_weather(data):
    prediction = model.predict(data)
    return prediction

# Streamlit app
st.title('Weather Prediction')

# User input for weather data
prcp = st.number_input('PRCP', value=0.0)
snow = st.number_input('SNOW', value=0.0)
snwd = st.number_input('SNWD', value=0.0)
tmax = st.number_input('TMAX', value=0.0)
tmin = st.number_input('TMIN', value=0.0)

# Create a DataFrame from user input
input_data = pd.DataFrame({
    'prcp': [prcp],
    'snow': [snow],
    'snwd': [snwd],
    'tmax': [tmax],
    'tmin': [tmin],
})

# Ensure all necessary columns are present
required_features = ['prcp', 'snow', 'snwd', 'tmax', 'tmin']
for feature in required_features:
    if feature not in input_data.columns:
        input_data[feature] = 0

# Make prediction
if st.button('Predict'):
    prediction = predict_weather(input_data)
    st.write(f'Predicted TMAX for next day: {prediction[0]}')
