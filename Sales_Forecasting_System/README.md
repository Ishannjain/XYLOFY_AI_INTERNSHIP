# Sales Forecasting System

This project builds an end-to-end sales forecasting and demand intelligence system using historical sales data. It includes data exploration, time-series analysis, forecasting, anomaly detection, and customer demand segmentation.

## Project Goal

The main goal is to forecast future sales, identify seasonal patterns, detect unusual sales behavior, and segment products or regions based on demand characteristics.

## Dataset

The analysis uses the historical sales dataset stored in `train.csv` as the primary source for forecasting and business insights. A supplemental file, `vgsales.csv`, is also loaded for additional context.

## Main Workflow

1. **Data Loading and Exploration**
   - Load the sales dataset
   - Clean and prepare the data
   - Extract date-based features such as year, month, quarter, and season
   - Review missing values, duplicates, and data types

2. **Time-Series Analysis**
   - Aggregate sales into monthly data
   - Visualize trend and seasonal patterns
   - Decompose the time series into observed, trend, seasonal, and residual components
   - Check stationarity using the Augmented Dickey-Fuller (ADF) test

3. **Forecasting Models**
   - **SARIMA**: A statistical forecasting model that handles trend and seasonality well for time-series data.
   - **Prophet**: A robust forecasting library designed for business time series with trend and seasonality components.
   - **XGBoost**: A machine-learning model using lag-based features to predict future sales.

4. **Segment-Level Forecasting**
   - Forecast sales for major product categories and regions such as Furniture, Technology, Office Supplies, West Region, and East Region.

5. **Anomaly Detection**
   - Identify unusual sales patterns using:
     - Isolation Forest
     - Rolling Z-Score

6. **Clustering / Demand Segmentation**
   - Group product sub-categories based on sales volume, growth, volatility, and average order value.

## Models Used

### 1. SARIMA
- Suitable for monthly and seasonal time-series data
- Captures autoregression, integration, and moving average components
- Works well when the data shows strong trend and seasonality

### 2. Prophet
- Designed for time series with daily, weekly, or yearly seasonality
- Handles missing values and trend changes automatically
- Useful for business forecasting tasks

### 3. XGBoost
- Tree-based machine learning algorithm
- Uses lagged features and seasonality-related inputs
- Good for non-linear relationships in structured forecasting data

## Tech Stack

- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical charts and plots
- **Statsmodels**: Time-series analysis and ADF testing
- **Prophet**: Forecasting model
- **XGBoost**: Gradient boosting regression model
- **Scikit-learn**: Clustering, preprocessing, and anomaly detection
- **Streamlit**: Optional dashboard/web app interface

## Files in the Project

- `train.csv`: Main sales dataset
- `vgsales.csv`: Additional supplemental dataset
- `analysis.ipynb`: Main notebook containing the full workflow
- `app.py`: Streamlit-based dashboard application
- `Charts/`: Folder containing generated plots and exported outputs

## How to Run

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Open and run the notebook:
   ```bash
   jupyter notebook analysis.ipynb
   ```

3. To run the dashboard:
   ```bash
   streamlit run app.py
   ```

## Summary

This system combines classical statistical forecasting and modern machine learning methods to provide a practical and explainable solution for sales forecasting and demand analysis.
