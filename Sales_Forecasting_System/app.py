import pandas as pd
import streamlit as st
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING)

st.set_page_config(page_title="Sales Forecasting Dashboard", page_icon="📈", layout="wide")

root = Path(__file__).resolve().parent
charts_dir = root / "Charts"


@st.cache_data
def load_sales_data():
    try:
        df = pd.read_csv(root / "train.csv")
        df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
        df["Year"] = df["Order Date"].dt.year
        df["Month"] = df["Order Date"].dt.month
        return df
    except Exception as e:
        st.error(f"Error loading sales data: {e}")
        st.stop()


@st.cache_data
def load_model_metrics():
    if (charts_dir / "model_comparison.csv").exists():
        return pd.read_csv(charts_dir / "model_comparison.csv")
    return pd.DataFrame(columns=["Model", "MAE", "RMSE", "MAPE (%)"])


@st.cache_data
def load_forecast(scope, selected):
    try:
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
    except Exception as e:
        logging.error(f"Error loading forecast: {e}")
        return pd.DataFrame(columns=["Date", "Forecast"])


sales_df = load_sales_data()
if sales_df is None or sales_df.empty:
    st.error("No sales data available. Please ensure train.csv exists in the project directory.")
    st.stop()

model_metrics = load_model_metrics()

page = st.sidebar.radio(
    "Navigation",
    ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"],
)

if page == "Sales Overview":
    st.title("Sales Overview Dashboard")
    st.markdown("Explore historical sales performance with filters and summary metrics.")

    region_options = ["All"] + sorted(sales_df["Region"].astype(str).unique().tolist())
    category_options = ["All"] + sorted(sales_df["Category"].astype(str).unique().tolist())
    year_options = ["All"] + sorted(sales_df["Year"].astype(int).unique().tolist())

    with st.sidebar:
        st.subheader("Filters")
        selected_region = st.multiselect("Region", options=region_options, default=region_options[1:])
        selected_category = st.multiselect("Category", options=category_options, default=category_options[1:])
        selected_year = st.selectbox("Year", options=year_options)

    filtered_df = sales_df.copy()
    if selected_year != "All":
        filtered_df = filtered_df[filtered_df["Year"] == selected_year]
    if "All" not in selected_region:
        filtered_df = filtered_df[filtered_df["Region"].isin(selected_region)]
    if "All" not in selected_category:
        filtered_df = filtered_df[filtered_df["Category"].isin(selected_category)]

    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        total_sales = filtered_df["Sales"].sum()
        avg_sale = filtered_df["Sales"].mean()
        top_region = filtered_df.groupby("Region")["Sales"].sum().idxmax()
        top_category = filtered_df.groupby("Category")["Sales"].sum().idxmax()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Sales", f"${total_sales:,.2f}")
        col2.metric("Average Sale", f"${avg_sale:,.2f}")
        col3.metric("Top Region", top_region)
        col4.metric("Top Category", top_category)

        yearly_sales = filtered_df.groupby("Year")["Sales"].sum().reset_index()
        monthly_sales = filtered_df.set_index("Order Date").resample("MS")["Sales"].sum().reset_index()

        tab1, tab2 = st.tabs(["Trend View", "Breakdown Table"])
        with tab1:
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Sales by Year")
                st.bar_chart(yearly_sales.set_index("Year"))
            with col_b:
                st.subheader("Monthly Trend")
                st.line_chart(monthly_sales.set_index("Order Date"))
        with tab2:
            grouped = filtered_df.groupby(["Region", "Category"])["Sales"].sum().reset_index()
            pivot = grouped.pivot(index="Region", columns="Category", values="Sales").fillna(0)
            st.dataframe(grouped, use_container_width=True)
            st.bar_chart(pivot)

elif page == "Forecast Explorer":
    st.title("Forecast Explorer")
    st.markdown("Compare forecast outputs and model metrics for selected categories or regions.")

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
        tab1, tab2 = st.tabs(["Forecast Chart", "Forecast Table"])
        with tab1:
            st.subheader(f"{selected_target} {scope} Forecast")
            st.line_chart(forecast_df.set_index("Date"))
        with tab2:
            st.dataframe(display_df, use_container_width=True)

    if not model_metrics.empty:
        best_model = model_metrics.sort_values("RMSE").iloc[0]
        st.subheader("Model Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("Best Model", best_model["Model"])
        col2.metric("MAE", f"{best_model['MAE']:.2f}")
        col3.metric("RMSE", f"{best_model['RMSE']:.2f}")

elif page == "Anomaly Report":
    st.title("Anomaly Report")
    st.markdown("Inspect detected anomalies and the weeks where unusual sales behavior occurred.")

    anomaly_image = charts_dir / "weekly_anomalies_comparison.png"
    if anomaly_image.exists():
        try:
            st.image(str(anomaly_image), caption="Anomaly detection chart", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading anomaly chart: {e}")
    
    try:
        anomaly_df = pd.read_csv(charts_dir / "weekly_anomalies.csv")
    except FileNotFoundError:
        st.error("Anomaly data file not found. Please run the analysis notebook first.")
        st.stop()
    anomaly_df["Order Date"] = pd.to_datetime(anomaly_df["Order Date"])
    anomaly_mask = anomaly_df["IF_Anomaly"] | anomaly_df["Z_Anomaly"]
    anomaly_table = anomaly_df.loc[anomaly_mask, ["Order Date", "Sales", "IF_Anomaly", "Z_Anomaly"]].copy()
    anomaly_table = anomaly_table.sort_values("Order Date")

    st.subheader("Detected Anomaly Dates")
    st.dataframe(anomaly_table.reset_index(drop=True), use_container_width=True)

else:
    st.title("Product Demand Segments")
    st.markdown("View product sub-category clusters and segment labels.")

    cluster_image = charts_dir / "product_demand_clusters.png"
    if cluster_image.exists():
        try:
            st.image(str(cluster_image), caption="Product demand segmentation chart", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading cluster chart: {e}")

    try:
        cluster_df = pd.read_csv(charts_dir / "product_clusters.csv")
    except FileNotFoundError:
        st.error("Cluster data file not found. Please run the analysis notebook first.")
        st.stop()
    selected_cluster = st.selectbox("Filter by cluster", options=["All"] + sorted(cluster_df["Cluster Label"].astype(str).unique().tolist()))
    if selected_cluster != "All":
        cluster_df = cluster_df[cluster_df["Cluster Label"] == selected_cluster]

    st.subheader("Sub-categories by Demand Cluster")
    st.dataframe(cluster_df[["Sub-Category", "Cluster Label", "Cluster"]], use_container_width=True)
