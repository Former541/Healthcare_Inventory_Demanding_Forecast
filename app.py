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

# --- 4. AI INTEGRATION (LLM Placeholder) ---
def get_llm_recommendations(item_name, forecast_data):
    """Placeholder for LLM call to get inventory recommendations."""
    
    # Extract key forecast metrics (e.g., Q1 2026 average and range)
    forecast_2026 = forecast_data[forecast_data['ds'].dt.year == 2026]
    
    if forecast_2026.empty:
        return "Not enough data to generate LLM recommendations for 2026."

    avg_yhat = forecast_2026['yhat'].mean()
    max_yhat = forecast_2026['yhat_upper'].max()
    min_yhat = forecast_2026['yhat_lower'].min()
    
    # Simple Mockup of LLM Response (REPLACE WITH REAL API CALL)
    st.info("💡 **AI Recommendation (LLM Integration):** Generating inventory insights...")
    
    # A prompt you would send to the LLM (OpenAI/Together.ai):
    # prompt = f"Analyze the forecast for '{item_name}'. The average daily demand for 2026 is {avg_yhat:.0f} units, with a peak potential of {max_yhat:.0f} units. Provide three specific, actionable inventory management recommendations."
    
    recommendations = f"""
    Based on the Prophet forecast for **{item_name}** in 2026 (Avg. Daily Demand: **{avg_yhat:.0f}** units):
    
    1. **Optimize Reorder Point (ROP):** Set the ROP to accommodate a safety stock equivalent to the maximum predicted daily demand (**{max_yhat:.0f}** units) plus lead time usage.
    2. **Monitor Peak Seasonality:** The model suggests demand spikes. Check the plot for high-demand periods (e.g., month/quarter) and pre-order 4-6 weeks in advance of those times.
    3. **Budget Review:** Given the forecast, review the budget for {item_name} purchase orders, factoring in a **{max_yhat:.0f}** unit upper-bound to prevent stockouts.
    """
    
    return recommendations

# --- 5. STREAMLIT APPLICATION ---
def main():
    st.set_page_config(layout="wide", page_title="AI Inventory Demand Forecaster")

    st.title("🏥 AI-Driven Healthcare Inventory Demand Forecaster")
    st.markdown("---")

    # Load data and models
    df, forecast_models = load_and_preprocess_data(DATA_FILE)
    
    if df is None:
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
    last_historical_date = historical_df.index.max()
    future_forecast = forecast_data[forecast_data['ds'] > last_historical_date]
    
    # Select key columns and rename them for clarity
    display_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
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
