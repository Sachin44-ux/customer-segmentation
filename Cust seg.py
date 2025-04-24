# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 12:35:55 2025

@author: ASUS
"""


import numpy as np
import joblib
import streamlit as st

# Loading the saved KMeans model and scaler using joblib
try:
    kmeans_model = joblib.load(r'D:\Customer Segmentation\kmeans_model.pkl')
    scaler = joblib.load(r'D:\Customer Segmentation\scaler.pkl')
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# Function for customer segmentation prediction
def predict_customer_segment(input_data):
    """
    Predicts the customer segment based on input data.

    Parameters:
    input_data (list): List of inputs [Age, Annual Income, Spending Score]

    Returns:
    str: Predicted cluster label and corresponding description
    """
    # Convert input data to numpy array and scale it
    input_data_as_numpy_array = np.asarray(input_data, dtype=float).reshape(1, -1)
    scaled_data = scaler.transform(input_data_as_numpy_array)

    # Predict the cluster
    cluster = kmeans_model.predict(scaled_data)[0]

    # Cluster descriptions
    cluster_descriptions = {
        0: "Low income, low spending",
        1: "Low income, high spending",
        2: "Medium income, medium spending",
        3: "High income, high spending"
    }

    return f'The customer belongs to segment {cluster}: {cluster_descriptions.get(cluster, "Unknown segment")}.'

# Main function for the Streamlit app
def main():
    # Title of the web app
    st.title('Customer Segmentation Prediction Web App')
    st.write("Provide customer data to predict their segment.")

    # Input fields for customer data
    Age = st.text_input('Age')
    Annual_Income = st.text_input('Annual Income (in $)')
    Spending_Score = st.text_input('Spending Score (1-100)')

    # Placeholder for the result
    result = ''

    # Prediction button
    if st.button('Predict Segment'):
        try:
            # Convert inputs to float for prediction
            input_data = [float(Age), float(Annual_Income), float(Spending_Score)]
            result = predict_customer_segment(input_data)
        except ValueError:
            result = 'Please enter valid numerical inputs.'

    # Display the result
    st.success(result)

# Run the app
if __name__ == '__main__':
    main()
