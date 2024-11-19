import pickle
import pandas as pd
import streamlit as st
import numpy as np
from geopy.distance import geodesic
from sklearn.preprocessing import LabelEncoder
import requests

# Load scaler and model
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Helper functions
def extract_column_value(df):
    df['City_code'] = df['Delivery_person_ID'].str.split("RES", expand=True)[0]

def extract_date_features(data):
    data["day"] = data.Order_Date.dt.day
    data["month"] = data.Order_Date.dt.month
    data["quarter"] = data.Order_Date.dt.quarter
    data["year"] = data.Order_Date.dt.year
    data['day_of_week'] = data.Order_Date.dt.day_of_week.astype(int)
    data["is_month_start"] = data.Order_Date.dt.is_month_start.astype(int)
    data["is_month_end"] = data.Order_Date.dt.is_month_end.astype(int)
    data["is_quarter_start"] = data.Order_Date.dt.is_quarter_start.astype(int)
    data["is_quarter_end"] = data.Order_Date.dt.is_quarter_end.astype(int)
    data["is_year_start"] = data.Order_Date.dt.is_year_start.astype(int)
    data["is_year_end"] = data.Order_Date.dt.is_year_end.astype(int)
    data['is_weekend'] = np.where(data['day_of_week'].isin([5, 6]), 1, 0)

def calculate_time_diff(df):
    df['Time_Orderd'] = pd.to_timedelta(df['Time_Orderd'])
    df['Time_Order_picked'] = pd.to_timedelta(df['Time_Order_picked'])
    df['Time_Order_picked_formatted'] = df['Order_Date'] + \
        np.where(df['Time_Order_picked'] < df['Time_Orderd'], pd.DateOffset(days=1), pd.DateOffset(days=0)) + df['Time_Order_picked']
    df['Time_Ordered_formatted'] = df['Order_Date'] + df['Time_Orderd']

    df['Time_Order_picked_formatted'] = pd.to_datetime(df['Time_Order_picked_formatted'])
    df['order_prepare_time'] = (df['Time_Order_picked_formatted'] - df['Time_Ordered_formatted']).dt.total_seconds() / 60

    df['order_prepare_time'].fillna(df['order_prepare_time'].median(), inplace=True)
    df.drop(['Time_Orderd', 'Time_Order_picked', 'Time_Ordered_formatted', 'Time_Order_picked_formatted', 'Order_Date'], axis=1, inplace=True)

def calculate_distance(df):
    restaurant_coordinates = df[['Restaurant_latitude', 'Restaurant_longitude']].to_numpy()
    delivery_coordinates = df[['Delivery_location_latitude', 'Delivery_location_longitude']].to_numpy()
    df['distance'] = np.array([geodesic(restaurant, delivery).km for restaurant, delivery in zip(restaurant_coordinates, delivery_coordinates)])

def label_encoding(df):
    categorical_columns = df.select_dtypes(include='object').columns
    label_encoder = LabelEncoder()
    df[categorical_columns] = df[categorical_columns].apply(lambda col: label_encoder.fit_transform(col))

def get_lat_long_opencage(address, api_key):
    url = f"https://api.opencagedata.com/geocode/v1/json?q={address}&key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data["results"]:
            latitude = data["results"][0]["geometry"]["lat"]
            longitude = data["results"][0]["geometry"]["lng"]
            return latitude, longitude
    return None, None

# Styling function
def set_background_color():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: black;
            color: white;
        }
        .stTextInput, .stNumberInput, .stSelectbox, .stDateInput, .stTimeInput {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid white;
            border-radius: 8px;
            color: white;
        }
        .stTextInput input, .stNumberInput input, .stSelectbox select {
            color: white;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            border-radius: 8px;
            border: none;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        h1, h2, h3, h4, h5, h6 {
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background color
set_background_color()

# Streamlit app
st.title('Delivery Time Prediction')

# User input
delivery_person_id = st.text_input('Delivery Person ID', 'BANGRES19DEL01')
age = st.number_input('Delivery Person Age', min_value=18, max_value=65, value=30)
ratings = st.number_input('Delivery Person Ratings', min_value=1.0, max_value=5.0, value=4.5)
order_date = st.date_input('Order Date')
time_ordered = st.time_input('Time Ordered')
time_order_picked = st.time_input('Time Order Picked')
weather = st.selectbox('Weather Conditions', ['Sunny', 'Cloudy', 'Rainy', 'Foggy'])
traffic = st.selectbox('Road Traffic Density', ['Low', 'Medium', 'High', 'Jam'])
vehicle_condition = st.number_input('Vehicle Condition', min_value=0, max_value=10, value=7)
order_type = st.selectbox('Type of Order', ['Snack', 'Meal', 'Drinks', 'Buffet'])
vehicle_type = st.selectbox('Type of Vehicle', ['Bike', 'Scooter', 'Car', 'Truck'])
multiple_deliveries = st.number_input('Multiple Deliveries', min_value=0, max_value=5, value=0)
festival = st.selectbox('Festival', ['No', 'Yes'])
city = st.selectbox('City', ['Urban', 'Semi-Urban', 'Metropolitan'])
restaurant_address = st.text_input('Restaurant Address')
delivery_address = st.text_input('Delivery Address')

api_key = "0bf4c293cabe46bfbca2fa2afa4c0455"
restaurant_lat, restaurant_long = get_lat_long_opencage(restaurant_address, api_key)
delivery_lat, delivery_long = get_lat_long_opencage(delivery_address, api_key)

if st.button("Get ETA for Delivery!"):
    input_data = pd.DataFrame({
        'Delivery_person_ID': [delivery_person_id],
        'Delivery_person_Age': [age],
        'Delivery_person_Ratings': [ratings],
        'Restaurant_latitude': [restaurant_lat],
        'Restaurant_longitude': [restaurant_long],
        'Delivery_location_latitude': [delivery_lat],
        'Delivery_location_longitude': [delivery_long],
        'Order_Date': [order_date],
        'Time_Orderd': [time_ordered],
        'Time_Order_picked': [time_order_picked],
        'Weather_conditions': [weather],
        'Road_traffic_density': [traffic],
        'Vehicle_condition': [vehicle_condition],
        'Type_of_order': [order_type],
        'Type_of_vehicle': [vehicle_type],
        'multiple_deliveries': [multiple_deliveries],
        'Festival': [festival],
        'City': [city]
    })

    input_data['Order_Date'] = pd.to_datetime(input_data['Order_Date'])
    extract_column_value(input_data)
    extract_date_features(input_data)
    calculate_time_diff(input_data)
    calculate_distance(input_data)
    label_encoding(input_data)
    input_data = input_data.drop(['Delivery_person_ID'], axis=1)

    # Scale and predict
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)

    # Display the result
    st.write(f'**Your food will arrive in approximately {prediction[0]} minutes.**')
