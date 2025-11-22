# Healthcare_Inventory_Demanding_Forecast
This is a Forecast made to see what demands are need for medical supply's for the next coming year of 

# 🏥 Healthcare Inventory Demand Forecasting

This project forecasts the daily demand (Average Usage Per Day) for key medical supplies for the entire year of 2026 using historical data. The forecast informs inventory management, reduces stockouts, and optimizes holding costs.

## Project Goal
The primary objective of this project is to **forecast daily demand** for key healthcare inventory items for the entire year of **2026**.

## Data Source
* **Dataset:** `healthcare inventory.csv`
* **Historical Range:** October 2024 to February 2026
* **Key Feature Forecasted:** `Avg_Usage_Per_Day`

## Methodology
The forecasting was conducted using a **time-series analysis** approach, specifically leveraging the **Prophet** library.

### 1. Data Preparation
* Loaded the `healthcare inventory.csv` into a Pandas DataFrame.
* Converted the `Date` column to a datetime object and set it as the index.
* Engineered time-based features (Year, Month, Day, DayOfWeek).
* Confirmed **no missing values** were present in the dataset.

### 2. Model Training
* A separate **Prophet model** was trained for each of the five unique inventory items.
* Models were configured to include **daily** and **yearly seasonality**.

### 3. Forecasting
* Predictions were generated for all items extending through **December 31, 2026**.
* Forecasts include the **predicted mean demand** (`yhat`), along with **lower** (`yhat_lower`) and **upper** (`yhat_upper`) confidence intervals.

## 🔑 Key Trends and Insights (2026 Forecast)
The models successfully generated demand predictions, providing crucial insights for inventory planning:

* **Ventilator:** The forecast indicates a slight **upward trend** in usage with noticeable **daily or weekly spikes**, suggesting a need for increased buffer stock on certain weekdays.
* **Surgical Mask:** Demand appears to be highly **seasonal**, peaking significantly during the late fall and early winter months (Q4 and Q1).
* **IV Drip:** This item shows the **most stable demand** with a tight confidence interval, indicating highly predictable usage.
* **Gloves:** Usage is high volume and follows a **moderate yearly cycle** with slight increases observed in summer months.
* **X-ray Machine (Usage):** The forecast shows a **clear, consistent upward linear trend** for 2026, signaling a long-term increase in demand for imaging services.

## Inventory Optimization
The forecasts provide a crucial input for inventory management:
* The `yhat` value can be used as the **target stock level**.
* The **upper bound** (`yhat_upper`) represents a **worst-case high demand** scenario, useful for setting maximum capacity or safety stock.

## Dependencies & Installation
This project requires the following Python libraries:

```bash
pip install pandas matplotlib prophet
