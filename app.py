import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained XGBoost model
with open("xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Function to make predictions
def predict_cost(features):
    return model.predict(features)

# Define the app
def main():
    st.title("Cost Prediction")

    # Define the sliders for various factors
    store_sales = st.slider("Store Sales (in millions)", 0.0, 20.0, step=0.1)
    unit_sales = st.slider("Unit Sales (in millions)", 0, 10, step=1)
    total_children = st.slider("Total Children", 0, 10, step=1)
    num_children_at_home = st.slider("Number of Children at Home", 0, 10, step=1)
    avg_cars_at_home = st.slider("Average Cars at Home", 0.0, 5.0, step=0.1)
    gross_weight = st.slider("Gross Weight", 0.0, 30.0, step=0.1)
    recyclable_package = st.slider("Recyclable Package", 0, 1, step=1)
    low_fat = st.slider("Low Fat", 0, 1, step=1)
    units_per_case = st.slider("Units per Case", 0, 50, step=1)
    store_sqft = st.slider("Store Sqft", 0, 50000, step=1)
    coffee_bar = st.slider("Coffee Bar", 0, 1, step=1)
    video_store = st.slider("Video Store", 0, 1, step=1)
    salad_bar = st.slider("Salad Bar", 0, 1, step=1)
    prepared_food = st.slider("Prepared Food", 0, 1, step=1)
    florist = st.slider("Florist", 0, 1, step=1)

    # Create a DataFrame with the input features
    input_data = pd.DataFrame(
        {
            "store_sales": [store_sales],
            "unit_sales": [unit_sales],
            "total_children": [total_children],
            "num_children_at_home": [num_children_at_home],
            "avg_cars_at home(approx)": [avg_cars_at_home],
            "gross_weight": [gross_weight],
            "recyclable_package": [recyclable_package],
            "low_fat": [low_fat],
            "units_per_case": [units_per_case],
            "store_sqft": [store_sqft],
            "coffee_bar": [coffee_bar],
            "video_store": [video_store],
            "salad_bar": [salad_bar],
            "prepared_food": [prepared_food],
            "florist": [florist],
}
)
    
# Normalize the input features
    input_data_scaled = scaler.transform(input_data)

# Make predictions
    if st.button("Predict Cost"):
        predicted_cost = predict_cost(input_data_scaled)
        st.success(f"Predicted Cost: ${predicted_cost[0]:.2f}")

if __name__ == "__main__":
    main()
