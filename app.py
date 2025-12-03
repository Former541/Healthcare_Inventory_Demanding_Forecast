import pandas as pd
import streamlit as st
from prophet import Prophet
import matplotlib.pyplot as plt
import os
# The 'openai' library is required and must be listed in requirements.txt

# --- 1. CONFIGURATION ---
DATA_FILE = 'healthcare_inventory.csv'

# --- 2. DATA LOADING & PREPROCESSING (Cached for speed) ---
# @st.cache_resource caches the expensive operation of loading and training models
@st.cache_resource
def load_and_preprocess_data(data_path):
    """Loads, preprocesses data, and trains all Prophet models."""
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error(f"Error: {data_path} not found. Please ensure the file is in the same directory.")
        return None, None

    # Preprocessing steps
    df['Date'] = pd.to_datetime(df['Date'])
    df['Avg_Usage_Per_Day'] = df['Avg_Usage_Per_Day'].fillna(df['Avg_Usage_Per_Day'].mean()) 

    # Prepare and train models
    unique_items = df['Item_Name'].unique()
    forecast_models = {}
    
    for item in unique_items:
        item_df = df[df['Item_Name'] == item].copy()
        prophet_df = item_df.reset_index()[['Date', 'Avg_Usage_Per_Day']]
        prophet_df.rename(columns={'Date': 'ds', 'Avg_Usage_Per_Day': 'y'}, inplace=True)
        
        # Initialize and fit Prophet model
        model = Prophet(daily_seasonality=True, yearly_seasonality=True, changepoint_prior_scale=0.05)
        model.fit(prophet_df)
        forecast_models[item] = model
        
    return df, forecast_models

# --- 3. FORECASTING LOGIC ---
def generate_forecast(model, historical_df, forecast_days=365):
    """Generates a forecast for a given model and historical data."""
    
    future = model.make_future_dataframe(periods=forecast_days, freq='D')
    forecast = model.predict(future)
    
    historical_data = historical_df.reset_index().rename(columns={'Date': 'ds', 'Avg_Usage_Per_Day': 'y'})
    historical_data['yhat'] = historical_data['y'] 
    
    return forecast, historical_data

# --- 4. AI INTEGRATION (LLM Implementation) ---
def get_llm_recommendations(item_name, forecast_data):
    """Calls the LLM (OpenAI) to get inventory recommendations based on forecast data."""
    
    # 1. Check if the API key is available in Streamlit secrets
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("🚨 OpenAI API Key not found in Streamlit Secrets. Please configure your secrets to enable LLM recommendations.")
        return "LLM recommendations disabled until API key is configured."
    
    # 2. Initialize the OpenAI Client
    try:
        from openai import OpenAI
        # Uses the key stored in Streamlit secrets
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
    except ImportError:
        st.error("🚨 The 'openai' library is not installed. Check your requirements.txt file.")
        return "LLM recommendations disabled."

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
    - The minimum expected daily demand (lower bound, yhat_lower) is {min_yhat:.0f} units.
    
    Provide a professional, one-paragraph summary of the key inventory trends and then list three specific, actionable inventory management recommendations.
    
    Format your response with the summary first, followed by a numbered list of recommendations.
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

# --- 5. STREAMLIT APPLICATION ---
def main():
    st.set_page_config(layout="wide", page_title="AI Inventory Demand Forecaster")

    st.title("🏥 AI-Driven Healthcare Inventory Demand Forecaster")
    st.markdown("---")

    # Load data and models
    df, forecast_models = load_and_preprocess_data(DATA_FILE)
    
    if df is None or not forecast_models:
        st.warning("Application could not load data or train models. Please check your `healthcare_inventory.csv` file and its data integrity.")
        return # Stop execution if data loading failed

    # Sidebar for User Input
    with st.sidebar:
        st.header("App Controls")
        item_names = sorted(forecast_models.keys())
        selected_item = st.selectbox(
            "Select Inventory Item:",
            item_names
        )
        st.markdown("---")
        st.subheader("Forecasting Parameters")
        forecast_days = st.slider(
            "Days to Forecast (from last historical date):", 
            min_value=30, max_value=730, value=365, step=30
        )

    # Main Content Area
    st.header(f"Demand Forecast and Analysis for: **{selected_item}**")
    
    # Get model and historical data for the selected item
    model = forecast_models[selected_item]
    historical_df = df[df['Item_Name'] == selected_item]
    
    # Generate Forecast
    forecast_data, historical_data = generate_forecast(model, historical_df, forecast_days)

    # 1. Plot Visualization
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Demand Forecast Visualization")
        
        # Use Prophet's plotting utility via Matplotlib
        fig = model.plot(forecast_data)
        ax = fig.gca()
        
        # Add historical data points (the 'y' values used for training)
        ax.plot(historical_data['ds'], historical_data['y'], 'k.', label='Historical Data')
        
        ax.set_title(f"Demand Forecast for {selected_item}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Average Daily Usage")
        
        # Adjust legend to be comprehensive
        ax.legend(['Historical Data', 'Forecasted Mean', 'Confidence Interval'], loc='upper left')
        
        st.pyplot(fig, use_container_width=True)

    # 2. LLM Recommendations (AI Component)
    with col2:
        st.subheader("LLM Inventory Recommendations")
        recommendations = get_llm_recommendations(selected_item, forecast_data)
        st.markdown(recommendations)
        
    st.markdown("---")
    
   # 3. Forecast Data Table (for detailed review)
st.subheader("Detailed Forecast Data (Next 30 Days)")

# Filter for just the next 30 days of the *future* prediction
# Ensure 'Date' column is used and is converted to a simple value if it's the index
last_historical_date = historical_df['Date'].max()

# Convert to a simple timestamp object for reliable comparison if necessary
if not pd.api.types.is_datetime64_any_dtype(last_historical_date):
    last_historical_date = pd.to_datetime(last_historical_date)

future_forecast = forecast_data[forecast_data['ds'] > last_historical_date]
    
    # Select key columns and rename them for clarity
display_cols = ['ds','yhat','yhat_lower','yhat_upper']
display_df = future_forecast[display_cols].head(30)
    
display_df.columns = ['Date', 'Predicted Mean Usage', 'Lower Bound (Optimistic)', 'Upper Bound (Pessimistic)']
    
    # Format the numerical columns for better display
    st.dataframe(display_df.style.format({
        'Predicted Mean Usage': "{:.2f}",
        'Lower Bound (Optimistic)': "{:.2f}",
        'Upper Bound (Pessimistic)': "{:.2f}"
    }), hide_index=True)


if __name__ == "__main__":
    main()
