import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
sns.set_theme(style='darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Create Charts directory if it doesn't exist
os.makedirs('Charts', exist_ok=True)

# Define Season mapping
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

# ==========================================================
# TASK 1: Data Loading & Preprocessing
# ==========================================================
print("--- Running Task 1: Data Prep & EDA ---")
df = pd.read_csv('train.csv')

# Parse dates
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True)

# Feature extraction
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Week Number'] = df['Order Date'].dt.isocalendar().week
df['Day of Week'] = df['Order Date'].dt.day_name()
df['Quarter'] = df['Order Date'].dt.quarter
df['Season'] = df['Month'].apply(get_season)

# Check missing and duplicates
missing_vals = df.isnull().sum().to_dict()
duplicate_count = df.duplicated().sum()
print(f"Missing values: {missing_vals}")
print(f"Duplicate rows: {duplicate_count}")

# Aggregations
daily_sales = df.groupby('Order Date')['Sales'].sum().reset_index()
weekly_sales = df.resample('W', on='Order Date')['Sales'].sum().reset_index()
monthly_sales = df.resample('MS', on='Order Date')['Sales'].sum().reset_index()

# EDA Questions
# Q1: Highest product category revenue
category_revenue = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
highest_category = category_revenue.index[0]
highest_category_rev = category_revenue.values[0]
print(f"Q1: Category with highest revenue: {highest_category} (${highest_category_rev:,.2f})")

# Q2: Region with most consistent sales growth over 4 years
region_yearly = df.groupby(['Region', 'Year'])['Sales'].sum().unstack(level=0)
region_growth = region_yearly.pct_change().dropna()
# Consistency defined as lowest volatility (std dev) of growth rates
region_growth_std = region_growth.std()
most_consistent_region = region_growth_std.idxmin()
print(f"Q2: Region with most consistent growth: {most_consistent_region} (std dev of growth: {region_growth_std.min():.4f})")

# Q3: Average shipping time by region
df['Ship Time'] = (df['Ship Date'] - df['Order Date']).dt.days
avg_ship_time_global = df['Ship Time'].mean()
avg_ship_time_region = df['Ship Time'].groupby(df['Region']).mean()
print(f"Q3: Average Ship Time: {avg_ship_time_global:.2f} days")
print("Average Ship Time by Region:\n", avg_ship_time_region)

# Q4: Consistently spiking months (seasonality)
monthly_patterns = df.groupby(['Year', 'Month'])['Sales'].sum().unstack(level=0)
avg_monthly_sales = monthly_patterns.mean(axis=1)
peak_month = avg_monthly_sales.idxmax()
print(f"Q4: Consistently spiking month (on average): Month {peak_month} (${avg_monthly_sales[peak_month]:,.2f})")

# Save EDA findings to CSV/txt for app and report
eda_summary = pd.DataFrame({
    'Category': category_revenue.index,
    'Revenue': category_revenue.values
})
eda_summary.to_csv('Charts/eda_category_revenue.csv', index=False)
avg_ship_time_region.to_frame('Avg Ship Time').to_csv('Charts/eda_ship_time_region.csv')
monthly_patterns.to_csv('Charts/eda_monthly_patterns.csv')

# ==========================================================
# TASK 2: Time Series Analysis & Decomposition
# ==========================================================
print("\n--- Running Task 2: Decomposition & Stationarity ---")
# Set Index for time series
ts_monthly = monthly_sales.set_index('Order Date')['Sales']

# Plot trend
plt.figure(figsize=(12, 6))
plt.plot(ts_monthly.index, ts_monthly.values, marker='o', color='#1f77b4', linewidth=2)
plt.title('Overall Monthly Sales Trend (2015-2018)', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Total Sales ($)')
plt.tight_layout()
plt.savefig('Charts/monthly_sales_trend.png', dpi=300)
plt.close()

# Time series decomposition
decomposition = seasonal_decompose(ts_monthly, model='additive', period=12)
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
decomposition.observed.plot(ax=axes[0], color='#1f77b4', legend=False)
axes[0].set_ylabel('Observed')
axes[0].set_title('Time Series Decomposition', fontsize=14, fontweight='bold')
decomposition.trend.plot(ax=axes[1], color='#ff7f0e', legend=False)
axes[1].set_ylabel('Trend')
decomposition.seasonal.plot(ax=axes[2], color='#2ca02c', legend=False)
axes[2].set_ylabel('Seasonal')
decomposition.resid.plot(ax=axes[3], color='#d62728', style='o', legend=False)
axes[3].set_ylabel('Residual')
axes[3].axhline(0, color='black', linestyle='--')
plt.xlabel('Date')
plt.tight_layout()
plt.savefig('Charts/time_series_decomposition.png', dpi=300)
plt.close()

# Stationarity test (ADF)
adf_result = adfuller(ts_monthly)
print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")
print("Critical Values:")
for key, value in adf_result[4].items():
    print(f"\t{key}: {value:.4f}")

# Differencing if non-stationary
is_stationary = adf_result[1] < 0.05
print(f"Is series stationary? {is_stationary}")
if not is_stationary:
    ts_diff = ts_monthly.diff().dropna()
    adf_diff = adfuller(ts_diff)
    print("\n--- ADF Test on Differenced Series ---")
    print(f"ADF Statistic (Diff): {adf_diff[0]:.4f}")
    print(f"p-value (Diff): {adf_diff[1]:.4f}")
    # Save diff plot
    plt.figure(figsize=(12, 5))
    plt.plot(ts_diff.index, ts_diff.values, marker='o', color='purple')
    plt.title('1st Order Differenced Monthly Sales', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Charts/differenced_sales.png', dpi=300)
    plt.close()

# ==========================================================
# TASK 3: Sales Forecasting & Model Comparisons
# ==========================================================
print("\n--- Running Task 3: Forecasting Models ---")
# Split data: Last 3 months (Oct, Nov, Dec 2018) as Test set
train_size = len(ts_monthly) - 3
train_ts = ts_monthly.iloc[:train_size]
test_ts = ts_monthly.iloc[train_size:]

print(f"Train period: {train_ts.index.min()} to {train_ts.index.max()} ({len(train_ts)} months)")
print(f"Test period: {test_ts.index.min()} to {test_ts.index.max()} ({len(test_ts)} months)")

# Evaluation helper
def calculate_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape

# ----------------- Model 1: SARIMA -----------------
# Grid search on train set for best parameters
best_aic = float('inf')
best_order = (1, 1, 1)
best_seasonal = (1, 1, 1, 12)

# Simple grid search loop to avoid errors
for p in [0, 1, 2]:
    for d in [0, 1]:
        for q in [0, 1]:
            for P in [0, 1]:
                for D in [0, 1]:
                    for Q in [0, 1]:
                        try:
                            model = SARIMAX(train_ts, order=(p,d,q), seasonal_order=(P,D,Q,12),
                                            enforce_stationarity=False, enforce_invertibility=False)
                            res = model.fit(disp=False)
                            if res.aic < best_aic:
                                best_aic = res.aic
                                best_order = (p, d, q)
                                best_seasonal = (P, D, Q, 12)
                        except:
                            continue

print(f"Best SARIMA Order: {best_order} x {best_seasonal} with AIC: {best_aic:.2f}")

# Fit best model on train and forecast test
sarima_model = SARIMAX(train_ts, order=best_order, seasonal_order=best_seasonal,
                       enforce_stationarity=False, enforce_invertibility=False)
sarima_res = sarima_model.fit(disp=False)
sarima_pred = sarima_res.forecast(steps=3)

# Fit best model on full dataset for future forecast
sarima_full = SARIMAX(ts_monthly, order=best_order, seasonal_order=best_seasonal,
                      enforce_stationarity=False, enforce_invertibility=False)
sarima_full_res = sarima_full.fit(disp=False)
sarima_future = sarima_full_res.get_forecast(steps=3)
sarima_future_mean = sarima_future.predicted_mean
sarima_conf = sarima_future.conf_int()

# ----------------- Model 2: Prophet -----------------
# Format train
df_prophet_train = train_ts.reset_index().rename(columns={'Order Date': 'ds', 'Sales': 'y'})
prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
prophet_model.fit(df_prophet_train)
future_df_test = prophet_model.make_future_dataframe(periods=3, freq='MS')
prophet_forecast_all = prophet_model.predict(future_df_test)
prophet_pred = prophet_forecast_all.iloc[-3:]['yhat'].values

# Fit on full data
df_prophet_full = ts_monthly.reset_index().rename(columns={'Order Date': 'ds', 'Sales': 'y'})
prophet_full_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
prophet_full_model.fit(df_prophet_full)
future_df_future = prophet_full_model.make_future_dataframe(periods=3, freq='MS')
prophet_future_forecast = prophet_full_model.predict(future_df_future)
prophet_future = prophet_future_forecast.iloc[-3:]['yhat'].values
prophet_future_lower = prophet_future_forecast.iloc[-3:]['yhat_lower'].values
prophet_future_upper = prophet_future_forecast.iloc[-3:]['yhat_upper'].values

# Save Prophet components plot
fig = prophet_full_model.plot_components(prophet_future_forecast)
fig.savefig('Charts/prophet_seasonality_breakdown.png', dpi=300)
plt.close()

# ----------------- Model 3: XGBoost -----------------
# Prepare supervised learning dataset
def create_xgboost_features(series):
    df_features = pd.DataFrame(series)
    df_features.columns = ['y']
    df_features['Lag_1'] = df_features['y'].shift(1)
    df_features['Lag_2'] = df_features['y'].shift(2)
    df_features['Lag_3'] = df_features['y'].shift(3)
    df_features['Rolling_Mean_3'] = df_features['y'].shift(1).rolling(window=3).mean()
    df_features['Month'] = df_features.index.month
    df_features['Quarter'] = df_features.index.quarter
    # Season feature
    df_features['Season_Num'] = df_features['Month'].apply(lambda x: 1 if x in [12, 1, 2] else (2 if x in [3, 4, 5] else (3 if x in [6, 7, 8] else 4)))
    return df_features.dropna()

# Fit XGBoost on train
xgb_data = create_xgboost_features(ts_monthly)
# Align train data
xgb_train = xgb_data.iloc[:train_size - 3]  # Adjust for dropped lags

X_train = xgb_train.drop(columns=['y'])
y_train = xgb_train['y']

xgb_reg = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
xgb_reg.fit(X_train, y_train)

# Recursive forecast on test
# We will predict next 3 months recursively
xgb_preds = []
last_obs = list(train_ts.iloc[-3:]) # Last 3 values of training set to start lags
# To forecast recursively:
for i in range(3):
    lag1 = last_obs[-1]
    lag2 = last_obs[-2]
    lag3 = last_obs[-3]
    roll3 = np.mean(last_obs[-3:])
    month = test_ts.index[i].month
    quarter = test_ts.index[i].quarter
    season = 1 if month in [12,1,2] else (2 if month in [3,4,5] else (3 if month in [6,7,8] else 4))
    
    x_input = pd.DataFrame([[lag1, lag2, lag3, roll3, month, quarter, season]],
                           columns=['Lag_1', 'Lag_2', 'Lag_3', 'Rolling_Mean_3', 'Month', 'Quarter', 'Season_Num'])
    pred_val = xgb_reg.predict(x_input)[0]
    xgb_preds.append(pred_val)
    last_obs.append(pred_val)

xgb_pred = np.array(xgb_preds)

# Fit on full dataset and predict future (Jan, Feb, Mar 2019)
xgb_reg_full = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
X_full = xgb_data.drop(columns=['y'])
y_full = xgb_data['y']
xgb_reg_full.fit(X_full, y_full)

xgb_futures = []
last_obs_full = list(ts_monthly.iloc[-3:])
future_dates = pd.date_range(start='2019-01-01', periods=3, freq='MS')
for i in range(3):
    lag1 = last_obs_full[-1]
    lag2 = last_obs_full[-2]
    lag3 = last_obs_full[-3]
    roll3 = np.mean(last_obs_full[-3:])
    month = future_dates[i].month
    quarter = future_dates[i].quarter
    season = 1 if month in [12,1,2] else (2 if month in [3,4,5] else (3 if month in [6,7,8] else 4))
    
    x_input = pd.DataFrame([[lag1, lag2, lag3, roll3, month, quarter, season]],
                           columns=['Lag_1', 'Lag_2', 'Lag_3', 'Rolling_Mean_3', 'Month', 'Quarter', 'Season_Num'])
    pred_val = xgb_reg_full.predict(x_input)[0]
    xgb_futures.append(pred_val)
    last_obs_full.append(pred_val)

xgb_future = np.array(xgb_futures)

# Calculate metrics on test set
metrics_sarima = calculate_metrics(test_ts.values, sarima_pred.values)
metrics_prophet = calculate_metrics(test_ts.values, prophet_pred)
metrics_xgb = calculate_metrics(test_ts.values, xgb_pred)

print("\n--- Test Set Metrics Comparison ---")
print(f"SARIMA  - MAE: {metrics_sarima[0]:.2f}, RMSE: {metrics_sarima[1]:.2f}, MAPE: {metrics_sarima[2]:.2f}%")
print(f"Prophet - MAE: {metrics_prophet[0]:.2f}, RMSE: {metrics_prophet[1]:.2f}, MAPE: {metrics_prophet[2]:.2f}%")
print(f"XGBoost - MAE: {metrics_xgb[0]:.2f}, RMSE: {metrics_xgb[1]:.2f}, MAPE: {metrics_xgb[2]:.2f}%")

# Create Model Comparison Table
comparison_df = pd.DataFrame({
    'Model': ['SARIMA', 'Prophet', 'XGBoost'],
    'MAE': [metrics_sarima[0], metrics_prophet[0], metrics_xgb[0]],
    'RMSE': [metrics_sarima[1], metrics_prophet[1], metrics_xgb[1]],
    'MAPE': [metrics_sarima[2], metrics_prophet[2], metrics_xgb[2]],
    'Forecast Month 1 (Jan 2019)': [sarima_future_mean.values[0], prophet_future[0], xgb_future[0]],
    'Forecast Month 2 (Feb 2019)': [sarima_future_mean.values[1], prophet_future[1], xgb_future[1]],
    'Forecast Month 3 (Mar 2019)': [sarima_future_mean.values[2], prophet_future[2], xgb_future[2]]
})
comparison_df.to_csv('Charts/model_comparison.csv', index=False)

# Determine best model
best_model_idx = comparison_df['MAE'].idxmin()
best_model_name = comparison_df.loc[best_model_idx, 'Model']
print(f"\nBest performing model: {best_model_name}")

# Plot Model Predictions vs Actuals on Test Set
plt.figure(figsize=(12, 6))
plt.plot(ts_monthly.index[-12:], ts_monthly.values[-12:], label='Actual Sales', color='black', linewidth=2, marker='o')
plt.plot(test_ts.index, sarima_pred, label='SARIMA Forecast', color='blue', linestyle='--', marker='^')
plt.plot(test_ts.index, prophet_pred, label='Prophet Forecast', color='orange', linestyle='--', marker='s')
plt.plot(test_ts.index, xgb_pred, label='XGBoost Forecast', color='green', linestyle='--', marker='d')
plt.title('Forecasting Model Performance on 3-Month Test Set (Oct - Dec 2018)', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Sales ($)')
plt.legend()
plt.tight_layout()
plt.savefig('Charts/model_test_comparison.png', dpi=300)
plt.close()

# Plot Best Model Future Forecast (Jan - Mar 2019)
plt.figure(figsize=(12, 6))
plt.plot(ts_monthly.index[-12:], ts_monthly.values[-12:], label='Historical Sales (2018)', color='black', linewidth=2, marker='o')
future_dates = pd.date_range(start='2019-01-01', periods=3, freq='MS')

if best_model_name == 'SARIMA':
    plt.plot(future_dates, sarima_future_mean, label='SARIMA Future Forecast', color='blue', marker='o', linewidth=2)
    plt.fill_between(future_dates, sarima_conf.iloc[:, 0], sarima_conf.iloc[:, 1], color='blue', alpha=0.15, label='95% Confidence Interval')
elif best_model_name == 'Prophet':
    plt.plot(future_dates, prophet_future, label='Prophet Future Forecast', color='orange', marker='o', linewidth=2)
    plt.fill_between(future_dates, prophet_future_lower, prophet_future_upper, color='orange', alpha=0.15, label='95% Confidence Interval')
else:
    plt.plot(future_dates, xgb_future, label='XGBoost Future Forecast', color='green', marker='o', linewidth=2)

plt.title(f'3-Month Future Sales Forecast using {best_model_name} (Jan - Mar 2019)', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Sales ($)')
plt.legend()
plt.tight_layout()
plt.savefig('Charts/future_forecast_best_model.png', dpi=300)
plt.close()

# ==========================================================
# TASK 4: Product Category & Region Level Forecasting
# ==========================================================
print("\n--- Running Task 4: Segment Level Forecasting ---")
# Use the best model (which will likely be SARIMA or Prophet based on standard metrics)
# Let's write a function to forecast a segment using the best model
def forecast_segment(segment_df, segment_name, model_type, steps=3):
    # Resample to monthly
    segment_ts = segment_df.resample('MS', on='Order Date')['Sales'].sum()
    # Check if we have 48 months
    all_months = pd.date_range(start='2015-01-01', end='2018-12-01', freq='MS')
    segment_ts = segment_ts.reindex(all_months, fill_value=0)
    
    if model_type == 'SARIMA':
        try:
            model = SARIMAX(segment_ts, order=best_order, seasonal_order=best_seasonal,
                             enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
            forecast = res.forecast(steps=steps).values
        except:
            # Fallback to simple parameters if best parameters don't fit
            model = SARIMAX(segment_ts, order=(1,1,1), seasonal_order=(0,1,1,12),
                             enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
            forecast = res.forecast(steps=steps).values
    elif model_type == 'Prophet':
        df_p = segment_ts.reset_index().rename(columns={'index': 'ds', 'Sales': 'y'})
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(df_p)
        future = m.make_future_dataframe(periods=steps, freq='MS')
        forecast = m.predict(future).iloc[-steps:]['yhat'].values
    else: # XGBoost
        # Features on historical
        xgb_seg_data = create_xgboost_features(segment_ts)
        X = xgb_seg_data.drop(columns=['y'])
        y = xgb_seg_data['y']
        xgb_m = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
        xgb_m.fit(X, y)
        
        xgb_futures = []
        last_obs = list(segment_ts.iloc[-3:])
        f_dates = pd.date_range(start='2019-01-01', periods=steps, freq='MS')
        for i in range(steps):
            lag1 = last_obs[-1]
            lag2 = last_obs[-2]
            lag3 = last_obs[-3]
            roll3 = np.mean(last_obs[-3:])
            month = f_dates[i].month
            quarter = f_dates[i].quarter
            season = 1 if month in [12,1,2] else (2 if month in [3,4,5] else (3 if month in [6,7,8] else 4))
            x_in = pd.DataFrame([[lag1, lag2, lag3, roll3, month, quarter, season]],
                                   columns=['Lag_1', 'Lag_2', 'Lag_3', 'Rolling_Mean_3', 'Month', 'Quarter', 'Season_Num'])
            pred_v = xgb_m.predict(x_in)[0]
            xgb_futures.append(pred_v)
            last_obs.append(pred_v)
        forecast = np.array(xgb_futures)
        
    return segment_ts, forecast

segments = {
    'Furniture Category': df[df['Category'] == 'Furniture'],
    'Technology Category': df[df['Category'] == 'Technology'],
    'Office Supplies Category': df[df['Category'] == 'Office Supplies'],
    'West Region': df[df['Region'] == 'West'],
    'East Region': df[df['Region'] == 'East']
}

segment_forecasts = {}
plt.figure(figsize=(12, 7))

for name, seg_df in segments.items():
    hist, fore = forecast_segment(seg_df, name, best_model_name)
    segment_forecasts[name] = fore
    # Plot forecast
    plt.plot(future_dates, fore, marker='o', label=f'{name} Forecast', linewidth=2)
    # Save predictions
    pd.DataFrame({
        'Date': future_dates,
        'Forecast': fore
    }).to_csv(f'Charts/forecast_{name.lower().replace(" ", "_")}.csv', index=False)

plt.title(f'Segment Level Future Forecasts (Jan - Mar 2019) using {best_model_name}', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Forecasted Sales ($)')
plt.legend()
plt.tight_layout()
plt.savefig('Charts/segment_forecasts_comparison.png', dpi=300)
plt.close()

# Identify strongest upcoming growth segment
# Growth defined as percentage increase from Dec 2018 historical to Mar 2019 forecast
growth_pct = {}
for name, seg_df in segments.items():
    hist_dec18 = seg_df.resample('MS', on='Order Date')['Sales'].sum().iloc[-1]
    fore_mar19 = segment_forecasts[name][-1]
    growth_pct[name] = ((fore_mar19 - hist_dec18) / hist_dec18) * 100

strongest_grower = max(growth_pct, key=growth_pct.get)
print(f"Strongest upcoming growth segment: {strongest_grower} ({growth_pct[strongest_grower]:.2f}% growth)")

# ==========================================================
# TASK 5: Anomaly Detection
# ==========================================================
print("\n--- Running Task 5: Anomaly Detection ---")
# Resample to weekly sales
ts_weekly = df.resample('W', on='Order Date')['Sales'].sum()

# 1. Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
# Reshape for sklearn
sales_features = ts_weekly.values.reshape(-1, 1)
iso_forest.fit(sales_features)
iso_anomalies = iso_forest.predict(sales_features) # -1 is anomaly, 1 is normal
anomaly_indices_if = np.where(iso_anomalies == -1)[0]

# 2. Z-Score anomaly detection (rolling window = 8 weeks)
rolling_mean = ts_weekly.rolling(window=8, min_periods=1).mean()
rolling_std = ts_weekly.rolling(window=8, min_periods=1).std()
# Handle NaN std dev for the first element
rolling_std.iloc[0] = rolling_std.iloc[1] if len(rolling_std) > 1 else 1.0
z_scores = (ts_weekly - rolling_mean) / rolling_std
anomaly_indices_z = np.where(np.abs(z_scores) > 2)[0]

# Create comparison dataframe
anomalies_df = pd.DataFrame(index=ts_weekly.index)
anomalies_df['Sales'] = ts_weekly.values
anomalies_df['IF_Anomaly'] = False
anomalies_df.iloc[anomaly_indices_if, anomalies_df.columns.get_loc('IF_Anomaly')] = True
anomalies_df['Z_Anomaly'] = False
anomalies_df.iloc[anomaly_indices_z, anomalies_df.columns.get_loc('Z_Anomaly')] = True
anomalies_df.to_csv('Charts/weekly_anomalies.csv')

# Plot Anomalies (Isolation Forest vs Z-score)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Isolation Forest plot
ax1.plot(ts_weekly.index, ts_weekly.values, color='#1f77b4', label='Weekly Sales', alpha=0.7)
ax1.scatter(ts_weekly.index[anomaly_indices_if], ts_weekly.values[anomaly_indices_if],
            color='red', label='IF Anomaly (Outlier)', marker='x', s=60, zorder=5)
ax1.set_title('Anomaly Detection using Isolation Forest', fontsize=12, fontweight='bold')
ax1.set_ylabel('Sales ($)')
ax1.legend()

# Z-score plot
ax2.plot(ts_weekly.index, ts_weekly.values, color='#1f77b4', label='Weekly Sales', alpha=0.7)
ax2.scatter(ts_weekly.index[anomaly_indices_z], ts_weekly.values[anomaly_indices_z],
            color='purple', label='Z-Score Anomaly (>2 Std Dev)', marker='o', s=40, zorder=5)
ax2.set_title('Anomaly Detection using Rolling Z-Score (8-Week Window)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Sales ($)')
ax2.set_xlabel('Date')
ax2.legend()

plt.tight_layout()
plt.savefig('Charts/weekly_anomalies_comparison.png', dpi=300)
plt.close()

# Identify overlap
overlap = anomalies_df[anomalies_df['IF_Anomaly'] & anomalies_df['Z_Anomaly']]
print(f"Isolation Forest flagged {len(anomaly_indices_if)} anomalies.")
print(f"Z-Score flagged {len(anomaly_indices_z)} anomalies.")
print(f"Number of common anomalies: {len(overlap)}")

# Save anomaly analysis notes to a text file for reporting
anomaly_notes = []
# Sort anomalies by sales to explain highest and lowest
high_anomalies = anomalies_df[anomalies_df['IF_Anomaly']].sort_values(by='Sales', ascending=False)
anomaly_notes.append("Top 3 High Anomalies:")
for date, row in high_anomalies.head(3).iterrows():
    month_name = date.strftime('%B')
    year = date.year
    explanation = "Likely holiday season surge, Black Friday promotions, or year-end clearing." if month_name in ['November', 'December'] else "Likely a major corporate bulk order or mid-year promotional event."
    anomaly_notes.append(f"- Week ending {date.strftime('%Y-%m-%d')}: Sales = ${row['Sales']:,.2f} ({explanation})")

low_anomalies = anomalies_df[anomalies_df['IF_Anomaly']].sort_values(by='Sales', ascending=True)
anomaly_notes.append("\nTop 3 Low Anomalies:")
for date, row in low_anomalies.head(3).iterrows():
    month_name = date.strftime('%B')
    year = date.year
    explanation = "Likely post-holiday seasonal drop or early-year reset." if month_name in ['January', 'February'] else "Likely transition period between seasonal promotional campaigns."
    anomaly_notes.append(f"- Week ending {date.strftime('%Y-%m-%d')}: Sales = ${row['Sales']:,.2f} ({explanation})")

with open('Charts/anomaly_explanations.txt', 'w') as f:
    f.write('\n'.join(anomaly_notes))

# ==========================================================
# TASK 6: Product Demand Segmentation using Clustering
# ==========================================================
print("\n--- Running Task 6: Product Demand Segmentation ---")
# Aggregate at sub-category level
# YoY Growth rate calculation (Growth from 2017 to 2018)
sub_yearly = df.groupby(['Sub-Category', 'Year'])['Sales'].sum().unstack(level=1)
# Handle case if years are missing (fill with 0)
sub_yearly = sub_yearly.fillna(0)
sub_growth_rate = ((sub_yearly[2018] - sub_yearly[2017]) / sub_yearly[2017]) * 100

# Volume (total sales)
sub_volume = df.groupby('Sub-Category')['Sales'].sum()

# Volatility (std dev of monthly sales)
sub_monthly_sales = df.groupby(['Sub-Category', pd.Grouper(key='Order Date', freq='MS')])['Sales'].sum().unstack(level=0).fillna(0)
sub_volatility = sub_monthly_sales.std()

# Average order value
sub_avg_order = df.groupby('Sub-Category').apply(lambda x: x['Sales'].sum() / x['Order ID'].nunique())

# Combine features
clustering_df = pd.DataFrame({
    'Total Volume': sub_volume,
    'YoY Growth': sub_growth_rate,
    'Sales Volatility': sub_volatility,
    'Avg Order Value': sub_avg_order
}).fillna(0)

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(clustering_df)

# Elbow method to choose cluster number
inertia = []
K_range = range(1, 10)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(K_range, inertia, marker='o', color='#2ca02c')
plt.title('Elbow Method for Optimal Clusters (Product Clustering)', fontsize=14, fontweight='bold')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.tight_layout()
plt.savefig('Charts/elbow_method.png', dpi=300)
plt.close()

# We will fit 4 clusters as requested (to label High Volume Stable, Low Volume High Volatility, Growing, Declining)
kmeans = KMeans(n_clusters=4, random_state=42)
clustering_df['Cluster'] = kmeans.fit_predict(scaled_features)

# PCA to reduce to 2 dimensions for visualization
pca = PCA(n_components=2, random_state=42)
pca_features = pca.fit_transform(scaled_features)
clustering_df['PCA1'] = pca_features[:, 0]
clustering_df['PCA2'] = pca_features[:, 1]

# Analyze clusters to label them meaningfully
# Let's print cluster stats
cluster_means = clustering_df.groupby('Cluster').mean()
print("Cluster Centroid Means:\n", cluster_means)

# Formulate labels based on characteristics:
# 1. High Volume, Stable Demand: High Total Volume, Low Volatility / Growth ratio, moderate AOV
# 2. Low Volume, High Volatility: Low Volume, High Volatility relative to volume, lower growth
# 3. Growing Demand: High YoY Growth
# 4. Declining Demand: Low or negative YoY Growth, lower Volume
labels = {}
for cluster_id in range(4):
    c_df = clustering_df[clustering_df['Cluster'] == cluster_id]
    vol = c_df['Total Volume'].mean()
    growth = c_df['YoY Growth'].mean()
    volat = c_df['Sales Volatility'].mean()
    aov = c_df['Avg Order Value'].mean()
    
    # Simple rule-based tagging for labeling
    if vol > clustering_df['Total Volume'].median() * 1.5:
        labels[cluster_id] = 'High Volume, Stable Demand'
    elif growth > 20:
        labels[cluster_id] = 'Growing Demand'
    elif growth < 0:
        labels[cluster_id] = 'Declining Demand'
    else:
        labels[cluster_id] = 'Low Volume, High Volatility'

# Ensure labels are unique, if there's duplicate logic, assign fallback unique labels
used_labels = set()
final_labels = {}
for cid in range(4):
    label = labels.get(cid, 'Moderate Demand')
    if label in used_labels:
        # Fallback to make them unique
        fallback_labels = ['High Volume, Stable Demand', 'Low Volume, High Volatility', 'Growing Demand', 'Declining Demand']
        for fl in fallback_labels:
            if fl not in used_labels:
                label = fl
                break
    used_labels.add(label)
    final_labels[cid] = label

clustering_df['Cluster Label'] = clustering_df['Cluster'].map(final_labels)
clustering_df.to_csv('Charts/product_clusters.csv')

# Plot K-Means Clusters
plt.figure(figsize=(12, 8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for cid in range(4):
    c_data = clustering_df[clustering_df['Cluster'] == cid]
    plt.scatter(c_data['PCA1'], c_data['PCA2'], s=150, color=colors[cid], label=final_labels[cid], alpha=0.8, edgecolors='black')
    
    # Annotate subcategories
    for subcat, row in c_data.iterrows():
        plt.annotate(subcat, (row['PCA1'], row['PCA2']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, fontweight='bold')

plt.title('Product Demand Segmentation (2D PCA Projection)', fontsize=14, fontweight='bold')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(prop={'weight': 'bold'})
plt.tight_layout()
plt.savefig('Charts/product_demand_clusters.png', dpi=300)
plt.close()

print("Clustering labels assigned:")
for cid, label in final_labels.items():
    print(f"Cluster {cid}: {label} -> {list(clustering_df[clustering_df['Cluster'] == cid].index)}")

print("\n--- Pipeline Completed Successfully ---")
