import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Sales Forecasting Dashboard", page_icon="📈", layout="wide")

root = Path(__file__).resolve().parent
charts_dir = root / "Charts"


@st.cache_data
def load_sales_data():
    df = pd.read_csv(root / "train.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    return df


@st.cache_data
def load_model_metrics():
    return pd.read_csv(charts_dir / "model_comparison.csv")


@st.cache_data
def load_forecast(scope, selected):
    file_map = {
        ("Category", "Furniture"): charts_dir / "forecast_furniture_category.csv",
        ("Category", "Office Supplies"): charts_dir / "forecast_office_supplies_category.csv",
        ("Category", "Technology"): charts_dir / "forecast_technology_category.csv",
        ("Region", "East"): charts_dir / "forecast_east_region.csv",
        ("Region", "West"): charts_dir / "forecast_west_region.csv",
    }
    path = file_map.get((scope, selected))
    if path is None or not path.exists():
        return pd.DataFrame(columns=["Date", "Forecast"])
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


sales_df = load_sales_data()
model_metrics = load_model_metrics()

page = st.sidebar.radio(
    "Navigation",
    ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"],
)

if page == "Sales Overview":
    st.title("Sales Overview Dashboard")
    st.markdown("Interactive view of the historical sales performance for the business.")

    yearly_sales = sales_df.groupby("Year")["Sales"].sum().reset_index()
    monthly_sales = sales_df.set_index("Order Date").resample("MS")["Sales"].sum().reset_index()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Total Sales by Year")
        st.bar_chart(yearly_sales.set_index("Year"))
    with col2:
        st.subheader("Monthly Sales Trend")
        st.line_chart(monthly_sales.set_index("Order Date"))

    st.subheader("Sales by Region and Category")
    region_options = ["All"] + sorted(sales_df["Region"].astype(str).unique().tolist())
    category_options = ["All"] + sorted(sales_df["Category"].astype(str).unique().tolist())

    selected_region = st.multiselect("Region", options=region_options, default=region_options[1:])
    selected_category = st.multiselect("Category", options=category_options, default=category_options[1:])

    filtered_df = sales_df.copy()
    if "All" not in selected_region:
        filtered_df = filtered_df[filtered_df["Region"].isin(selected_region)]
    if "All" not in selected_category:
        filtered_df = filtered_df[filtered_df["Category"].isin(selected_category)]

    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        grouped = filtered_df.groupby(["Region", "Category"])["Sales"].sum().reset_index()
        pivot = grouped.pivot(index="Region", columns="Category", values="Sales").fillna(0)
        st.dataframe(grouped, use_container_width=True)
        st.bar_chart(pivot)

elif page == "Forecast Explorer":
    st.title("Forecast Explorer")
    st.markdown("Explore forecast outputs for selected categories or regions and compare model metrics.")

    scope = st.selectbox("Forecast level", ["Category", "Region"])
    if scope == "Category":
        target_options = ["Furniture", "Office Supplies", "Technology"]
    else:
        target_options = ["East", "West"]

    selected_target = st.selectbox("Select item", target_options)
    horizon = st.slider("Forecast horizon (months ahead)", min_value=1, max_value=3, value=3)

    forecast_df = load_forecast(scope, selected_target)
    if forecast_df.empty:
        st.warning("No forecast file was found for the selected input.")
    else:
        display_df = forecast_df.head(horizon).copy()
        display_df["Forecast"] = display_df["Forecast"].round(2)
        st.subheader(f"{selected_target} {scope} Forecast")
        st.line_chart(forecast_df.set_index("Date"))
        st.dataframe(display_df, use_container_width=True)

    best_model = model_metrics.sort_values("RMSE").iloc[0]
    st.subheader("Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Best Model", best_model["Model"])
    col2.metric("MAE", f"{best_model['MAE']:.2f}")
    col3.metric("RMSE", f"{best_model['RMSE']:.2f}")

elif page == "Anomaly Report":
    st.title("Anomaly Report")
    st.markdown("Detected anomalies from the time-series analysis and their sales values.")

    anomaly_image = charts_dir / "weekly_anomalies_comparison.png"
    if anomaly_image.exists():
        st.image(str(anomaly_image), caption="Anomaly detection chart", use_container_width=True)

    anomaly_df = pd.read_csv(charts_dir / "weekly_anomalies.csv")
    anomaly_df["Order Date"] = pd.to_datetime(anomaly_df["Order Date"])
    anomaly_mask = anomaly_df["IF_Anomaly"] | anomaly_df["Z_Anomaly"]
    anomaly_table = anomaly_df.loc[anomaly_mask, ["Order Date", "Sales", "IF_Anomaly", "Z_Anomaly"]].copy()
    anomaly_table = anomaly_table.sort_values("Order Date")
    st.subheader("Detected Anomaly Dates")
    st.dataframe(anomaly_table.reset_index(drop=True), use_container_width=True)

else:
    st.title("Product Demand Segments")
    st.markdown("Cluster view showing the demand segments for each sub-category.")

    cluster_image = charts_dir / "product_demand_clusters.png"
    if cluster_image.exists():
        st.image(str(cluster_image), caption="Product demand segmentation chart", use_container_width=True)

    cluster_df = pd.read_csv(charts_dir / "product_clusters.csv")
    st.subheader("Sub-categories by Demand Cluster")
    st.dataframe(cluster_df[["Sub-Category", "Cluster Label", "Cluster"]], use_container_width=True)
