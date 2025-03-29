import streamlit as st
import pandas as pd
import numpy as np
import joblib  # If the model is saved as a pickle file

# Load the trained ML model
model = joblib.load("model.pkl")  # Update with correct model path

# Load dataset from CSV
csv_file_path = "Prediction Data"  # Update with the actual file path
df = pd.read_csv(csv_file_path)

# Streamlit UI
st.title("Food Shortage Early Warning System")

st.sidebar.header("User Input")
country = st.sidebar.text_input("Enter Country Name")
year = st.sidebar.number_input("Enter Year", min_value=2000, max_value=2100, step=1)

if st.sidebar.button("Predict"):
    # Filter dataset for the given country and year
    filtered_data = df[(df["Country"] == country) & (df["Year"] == year)]

    if not filtered_data.empty:
        # Drop non-numeric columns before passing to the model
        X = filtered_data.drop(columns=["Country", "Year"])  

        # Make predictions
        predictions = model.predict(X)

        # Display results
        st.subheader("Predicted Food Shortage Probability")
        st.write(predictions)

        # Optional: Visualization
        st.bar_chart(predictions)

    else:
        st.error("No data found for the given country and year.")

st.sidebar.write("Note: Predictions are based on historical data and trends.")
