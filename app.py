import pandas as pd
import streamlit as st
from prophet import Prophet
import matplotlib.pyplot as plt
import os
# NOTE: You will need to install the openai or together library for the AI component
# import openai 
# import together

# --- 1. CONFIGURATION ---
DATA_FILE = 'healthcare_inventory.csv'
# NOTE: Set your API Key as a Streamlit Secret or environment variable
# API_KEY = os.environ.get("OPENAI_API_KEY")

# --- 2. DATA LOADING & PREPROCESSING (Cached for speed) ---
# @st.cache_resource is used for caching objects like ML models
@st.cache_resource
def load_and_preprocess_data(data_path):
    """Loads, preprocesses data, and trains all Prophet models."""
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error(f"Error: {data_path} not found. Please ensure the file is in the same directory.")
        return None, None

    # Preprocessing steps from your notebook
    df['Date'] = pd.to_datetime(df['Date'])
    df['Avg_Usage_Per_Day'] = df['Avg_Usage_Per_Day'].fillna(df['Avg_Usage_Per_Day'].mean()) 

    # Prepare and train models
    unique_items = df['Item_Name'].unique()
    forecast_models = {}
    
    for item in unique_items:
        item_df = df[df['Item_Name'] == item].copy()
        prophet_df = item_df.reset_index()[['Date', 'Avg_Usage_Per_Day']]
        prophet_df.rename(columns={'Date': 'ds', 'Avg_Usage_Per_Day': 'y'}, inplace=True)
        
        # Initialize and fit Prophet model with seasonalities
        model = Prophet(daily_seasonality=True, yearly_seasonality=True, changepoint_prior_scale=0.05)
        model.fit(prophet_df)
        forecast_models[item] = model
        
    return df, forecast_models

# --- 3. FORECASTING LOGIC ---
def generate_forecast(model, historical_df, forecast_days=365):
    """Generates a forecast for a given model and historical data."""
    
    # Create future dataframe for 365 days (or until the end of 2026, as in your notebook)
    # Using 365 days for simplicity, but you can adjust this logic:
    future = model.make_future_dataframe(periods=forecast_days, freq='D')
    
    # Generate predictions
    forecast = model.predict(future)
    
    # Combine historical and forecasted data for plotting
    historical_data = historical_df.reset_index().rename(columns={'Date': 'ds', 'Avg_Usage_Per_Day': 'y'})
    
    # Ensure 'yhat' is not used for historical data to avoid errors in plotting
    historical_data['yhat'] = historical_data['y'] 
    
    return forecast, historical_data

# --- 4. AI INTEGRATION (LLM Implementation) ---
# NOTE: This uses Streamlit's built-in secrets management for security
def get_llm_recommendations(item_name, forecast_data):
    """Calls the LLM (OpenAI) to get inventory recommendations based on forecast data."""
    
    # 1. Check if the API key is available in Streamlit secrets
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("🚨 OpenAI API Key not found in Streamlit Secrets. Please configure your secrets to enable LLM recommendations.")
        return "LLM recommendations disabled until API key is configured."
    
    # 2. Initialize the OpenAI Client
    try:
        # We must import and initialize the client here to avoid initialization errors 
        # when Streamlit first loads before secrets are available.
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
    except ImportError:
        st.error("🚨 The 'openai' library is not installed. Check your requirements.txt file.")
        return "LLM recommendations disabled."

    # ... (Rest of the code to calculate avg_yhat, max_yhat, etc., remains the same)
    
    # Filter the forecast data to get key metrics for the next year (2026)
    forecast_2026 = forecast_data[forecast_data['ds'].dt.year == 2026]
    
    if forecast_2026.empty:
        return "Not enough data to generate LLM recommendations for 2026."

    avg_yhat = forecast_2026['yhat'].mean()
    max_yhat = forecast_2026['yhat_upper'].max()
    min_yhat = forecast_2026['yhat_lower'].min()
    
    # 3. Construct the detailed prompt
    prompt = f"""
    Analyze the following inventory demand forecast for '{item_name}':
    - The average predicted daily demand (yhat) for 2026 is {avg_yhat:.0f} units.
    - The maximum expected daily demand (upper bound, yhat_upper) is {max_yhat:.0f} units.
    ... (the rest of your prompt text)
    """
    
    # 4. Make the API Call
    st.info("💡 **AI Recommendation (LLM Integration):** Generating inventory insights...")
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # A cost-effective choice
            messages=[
                {"role": "system", "content": "You are a senior inventory management consultant. Provide precise, data-driven advice."},
                {"role": "user", "content": prompt}
            ]
        )
        llm_output = response.choices[0].message.content
        return llm_output
        
    except Exception as e:
        st.error(f"Error calling the LLM API: {e}")
        return "Failed to retrieve LLM recommendations due to an API error."
