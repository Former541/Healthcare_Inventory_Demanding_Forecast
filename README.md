# Healthcare_Inventory_Demanding_Forecast
This is a Forecast made to see what demands are need for medical supply's for the next coming year of 

🏥 Healthcare Inventory Demand Forecasting
Project Goal
The primary objective of this project is to forecast daily demand (Average Usage Per Day) for key healthcare inventory items for the entire year of 2026 using historical data. The forecast is intended to inform inventory management, reduce stockouts, and optimize holding costs.

Data Source
Dataset: healthcare inventory.csv

Historical Range: October 2024 to February 2026 (based on the index of your processed DataFrame).

Key Feature Forecasted: Avg_Usage_Per_Day

## Methodology
The forecasting was conducted using a time-series analysis approach, specifically leveraging the Prophet library.

1.Data Preparation:

Loaded the healthcare inventory.csv into a Pandas DataFrame.

Converted the Date column to a datetime object and set it as the index.

Engineered time-based features (Year, Month, Day, DayOfWeek).

Confirmed no missing values were present in the dataset.

2.Model Training:

A separate Prophet model was trained for each of the five unique inventory items: Ventilator, Surgical Mask, IV Drip, Gloves, and X-ray Machine.

The models were configured to include daily and yearly seasonality for improved accuracy in capturing cyclical demand patterns.

3.Forecasting:

Predictions were generated for all items extending through December 31, 2026.

The forecasts include the predicted mean demand (yhat), along with a lower (yhat_lower) and upper (yhat_upper) confidence interval.

## Key Trends and Insights (2026 Forecast)
Overall Demand: The models successfully generated demand predictions for all five items. The forecasts generally follow the historical patterns but extend into 2026, including their respective confidence intervals.

(Action Item: Review the plots generated in your notebook to add specific, descriptive trends here, such as:)

Ventilator: The forecast indicates a slight upward trend in average usage throughout 2026, with noticeable daily or weekly spikes.

Surgical Mask: Demand appears to be highly seasonal, potentially peaking during specific months of the year.

## Inventory Optimization: The forecasts provide a crucial input for inventory management:

The yhat value can be used as the target stock level.

The upper bound (yhat_upper) represents a worst-case high demand scenario, which should be considered for setting maximum capacity.

Dependencies & Installation
This project requires the following Python libraries:

Bash

pip install pandas matplotlib prophet
Usage
The forecasting pipeline is implemented in the Mod_5.ipynb notebook. To re-run the analysis:

Ensure the healthcare inventory.csv file is in the same directory.

Run all cells in the Mod_5.ipynb notebook.
