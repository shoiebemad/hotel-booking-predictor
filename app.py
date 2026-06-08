
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# page config
st.set_page_config(
    page_title="Hotel Booking Predictor",
    page_icon="🏨",
    layout="centered"
)

# custom CSS styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1c1c1c, #2b2b2b, #3a3a3a);
        color: #f0f0f0;
    }
    .stButton>button {
        background-color: #c0392b;
        color: white;
        border-radius: 10px;
        height: 50px;
        width: 100%;
        font-size: 18px;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #a93226;
    }
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #f0f0f0 !important;
        font-size: 15px;
    }
    h1, h2, h3, h4 {
        color: #f0f0f0 !important;
    }
    .stMarkdown p {
        color: #cccccc !important;
    }
    </style>
""", unsafe_allow_html=True)

# loading the saved model and preprocessing tools
rf_mad_model = joblib.load('rf_mad_model.pkl')
scaler_mad = joblib.load('scaler_mad.pkl')

# app title and description
st.title("Hotel Booking Cancellation Predictor")
st.markdown("Fill in the booking details below to predict whether the booking will be canceled or not.")
st.markdown("---")

# section 1 - booking info
st.subheader("Booking Information")
col1, col2 = st.columns(2)

with col1:
    hotel = st.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])
    lead_time = st.slider("Lead Time (days in advance)", 0, 700, 50)
    market_segment = st.selectbox("Booking Channel", ["Online TA", "Offline TA/TO", "Direct", "Corporate", "Groups"])

with col2:
    adr = st.number_input("Average Price Per Night", 0, 1000, 100)
    days_in_waiting_list = st.number_input("Days in Waiting List", 0, 400, 0)
    customer_type = st.selectbox("Customer Type", ["Transient", "Transient-Party", "Contract", "Group"])

st.markdown("---")

# section 2 - guest info
st.subheader("Guest Information")
col3, col4 = st.columns(2)

with col3:
    adults = st.number_input("Number of Adults", 1, 10, 2)
    children = st.number_input("Number of Children", 0, 10, 0)
    is_repeated_guest = st.selectbox("Returning Guest?", ["No", "Yes"])

with col4:
    previous_cancellations = st.number_input("Previous Cancellations", 0, 20, 0)
    previous_bookings_not_canceled = st.number_input("Previous Successful Bookings", 0, 50, 0)
    booking_changes = st.number_input("Number of Booking Changes", 0, 20, 0)

st.markdown("---")

# section 3 - special requests
st.subheader("Special Requests")
total_of_special_requests = st.slider("Number of Special Requests", 0, 5, 0)

st.markdown("---")

# converting inputs
hotel_encoded = 0 if hotel == "City Hotel" else 1
is_repeated_guest_encoded = 1 if is_repeated_guest == "Yes" else 0
has_special_requests = 1 if total_of_special_requests > 0 else 0
is_late_booking = 1 if lead_time < 7 else 0
market_segment_map = {"Online TA": 4, "Offline TA/TO": 3, "Direct": 2, "Corporate": 1, "Groups": 0}
market_segment_encoded = market_segment_map[market_segment]
customer_type_map = {"Transient": 3, "Transient-Party": 2, "Contract": 0, "Group": 1}
customer_type_encoded = customer_type_map[customer_type]

# building the full input
input_data = pd.DataFrame([{
    'hotel': hotel_encoded,
    'lead_time': lead_time,
    'arrival_date_year': 2016,
    'arrival_date_month': 5.50,
    'arrival_date_week_number': 27.17,
    'arrival_date_day_of_month': 15.78,
    'stays_in_weekend_nights': 0.93,
    'stays_in_week_nights': 2.50,
    'adults': adults,
    'children': children,
    'babies': 0.01,
    'meal': 0.56,
    'country': 93.26,
    'market_segment': market_segment_encoded,
    'distribution_channel': 2.59,
    'is_repeated_guest': is_repeated_guest_encoded,
    'previous_cancellations': previous_cancellations,
    'previous_bookings_not_canceled': previous_bookings_not_canceled,
    'reserved_room_type': 0.99,
    'assigned_room_type': 1.33,
    'booking_changes': booking_changes,
    'agent': 74.76,
    'company': 10.79,
    'days_in_waiting_list': days_in_waiting_list,
    'customer_type': customer_type_encoded,
    'adr': adr,
    'required_car_parking_spaces': 0.06,
    'total_of_special_requests': total_of_special_requests,
    'has_special_requests': has_special_requests,
    'is_late_booking': is_late_booking
}])

# scaling the same columns we scaled during training
scaled_cols = ['lead_time', 'adr', 'stays_in_weekend_nights', 'stays_in_week_nights']
input_data[scaled_cols] = scaler_mad.transform(input_data[scaled_cols])

# predict button
if st.button("Predict"):
    prediction = rf_mad_model.predict(input_data)[0]
    probability = rf_mad_model.predict_proba(input_data)[0]

    if prediction == 1:
        st.error(f"This booking is likely to be CANCELED: Confidence: {round(probability[1]*100, 1)}%")
    else:
        st.success(f"This booking is likely to SHOW UP: Confidence: {round(probability[0]*100, 1)}%")
