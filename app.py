import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import pickle

with open("rf_model_pickle.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("scaler_pickle.pkl", "rb") as f:
    scaler = pickle.load(f)


st.title("🚰 Water Consumption Prediction App")
st.markdown("This app predicts future water consumption using AI models.")


date_input = st.date_input("Select a Date for Prediction:", pd.to_datetime("2025-01-01"))
num_days = st.slider("Predict for how many days?", 1, 30, 7)

future_dates = pd.date_range(start=date_input, periods=num_days, freq='D')


X_future = np.random.rand(len(future_dates), 7)
X_future_scaled = scaler.transform(X_future)


rf_preds = rf_model.predict(X_future_scaled)


fig = go.Figure()
fig.add_trace(go.Scatter(x=future_dates, y=rf_preds, mode='lines', name='Random Forest', line=dict(color='red', dash='dash')))
fig.update_layout(title="Predicted Water Consumption", xaxis_title="Date", yaxis_title="Consumption")

st.plotly_chart(fig)
