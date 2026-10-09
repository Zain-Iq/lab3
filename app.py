from pathlib import Path
import math

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="EC2 Instance EDA Dashboard", layout="wide")
st.title("EC2 Instance EDA Dashboard")
st.caption("Lab 3 — EC2 Instance Cost & Specification Explorer")
st.info("Prices come from the supplied CSV. Monthly estimates use 730 hours.")

# 1. Load the original dataset.
csv_path = Path(__file__).with_name("ec2dataset.csv")
if not csv_path.exists():
    st.error("Place ec2dataset.csv in the same folder as app.py.")
    st.stop()
raw_df = pd.read_csv(csv_path)

st.header("1. Dataset Overview")
a, b, c = st.columns(3)
a.metric("Number of Records", len(raw_df))
b.metric("Original Columns", len(raw_df.columns))
c.metric("Raw Missing Values", int(raw_df.isna().sum().sum()))
with st.expander("Raw dataset, columns, and data types"):
    st.dataframe(raw_df, use_container_width=True)
    st.write("Columns:", raw_df.columns.tolist())
    st.dataframe(raw_df.dtypes.astype(str).rename("Data Type"))

# 2. Clean numeric specifications and prices.
df = raw_df.copy()
df["Memory_GiB"] = pd.to_numeric(
    df["Instance Memory"].str.replace(",", "", regex=False)
    .str.extract(r"([\d.]+)", expand=False), errors="coerce"
)
df["vCPU_Count"] = pd.to_numeric(
    df["vCPUs"].str.extract(r"(\d+)", expand=False), errors="coerce"
)


def clean_price(value):
    if pd.isna(value) or "unavailable" in str(value).lower():
        return float("nan")
    return pd.to_numeric(
        str(value).replace("$", "").replace(",", "")
        .replace(" hourly", "").strip(), errors="coerce"
    )


price_columns = [
    "On Demand", "Linux Reserved cost", "Linux Spot Minimum cost",
    "Windows On Demand cost", "Windows Reserved cost"
]
for column in price_columns:
    df[column + "_USD"] = df[column].apply(clean_price)

# Calculate ALL derived columns BEFORE filtering.
df["Monthly_On_Demand"] = df["On Demand_USD"] * 730
df["Cost_Per_GiB"] = df["On Demand_USD"] / df["Memory_GiB"]
df["Cost_Per_vCPU"] = df["On Demand_USD"] / df["vCPU_Count"]
df["Memory_per_vCPU"] = df["Memory_GiB"] / df["vCPU_Count"]

numeric_columns = ["Memory_GiB", "vCPU_Count"] + [
    column + "_USD" for column in price_columns
]
with st.expander("Cleaning results and summary statistics"):
    st.write("Unavailable prices become missing numeric values, not zero.")
    st.dataframe(df[numeric_columns].isna().sum().rename("Missing After Cleaning"))
    st.dataframe(df[numeric_columns].describe())

# 3. Interactive filters, including all five challenges.
st.sidebar.header("Filters")
max_memory = st.sidebar.slider(
    "Maximum Memory (GiB)", 0.5,
    float(df["Memory_GiB"].max()), float(df["Memory_GiB"].max()), step=0.5
)
cpu_values = sorted(df["vCPU_Count"].dropna().astype(int).unique().tolist())
selected_cpu = st.sidebar.multiselect("vCPU Count", cpu_values, default=cpu_values)
network_values = sorted(df["Network Performance"].dropna().unique().tolist())
selected_network = st.sidebar.multiselect(
    "Network Performance", network_values, default=network_values
)
storage_values = ["EBS", "SSD", "NVMe", "HDD"]
selected_storage = st.sidebar.multiselect(
    "Storage Type", storage_values, default=storage_values
)
st.sidebar.caption("Storage matches any selected label; NVMe can overlap SSD or HDD.")
hourly_limit = float(math.ceil(df["On Demand_USD"].max()))
max_hourly = st.sidebar.slider(
    "Maximum Hourly Price (USD)", 0.0, hourly_limit, hourly_limit, step=0.01
)
max_monthly = st.sidebar.number_input(
    "Maximum Monthly Cost (USD)", min_value=0.0,
    value=float(math.ceil(df["Monthly_On_Demand"].max())), step=5.0
)
st.sidebar.caption("Try monthly budgets of $10, $25, $50, $100, or $500.")
include_unknown = st.sidebar.checkbox("Include unavailable On-Demand prices", value=True)
st.sidebar.caption("Unpriced rows cannot be verified against either budget limit.")

storage_mask = df["Instance Storage"].str.contains(
    "|".join(selected_storage), case=False, na=False
) if selected_storage else pd.Series(False, index=df.index)
price_mask = (
    (df["On Demand_USD"] <= max_hourly)
    & (df["Monthly_On_Demand"] <= max_monthly)
)
if include_unknown:
    price_mask = price_mask | df["On Demand_USD"].isna()
filtered_df = df[
    (df["Memory_GiB"] <= max_memory)
    & df["vCPU_Count"].isin(selected_cpu)
    & df["Network Performance"].isin(selected_network)
    & storage_mask & price_mask
].copy()

st.header("2. Filtered EC2 Instances")
st.write(f"{len(filtered_df)} instances found")
st.dataframe(filtered_df, use_container_width=True)
st.download_button(
    "Download Filtered Dataset", filtered_df.to_csv(index=False),
    file_name="filtered_ec2_instances.csv", mime="text/csv"
)
if filtered_df.empty:
    st.warning("No matching instances. Increase limits or select more filter options.")
    st.stop()

# 4. KPI cards.
st.header("3. EC2 Summary")
a, b, c, d = st.columns(4)
a.metric("Instances", len(filtered_df))
b.metric("Avg Memory", f"{filtered_df['Memory_GiB'].mean():.2f} GiB")
c.metric("Avg vCPUs", f"{filtered_df['vCPU_Count'].mean():.1f}")
average_cost = filtered_df["On Demand_USD"].mean()
d.metric("Avg Hourly Cost", "N/A" if pd.isna(average_cost) else f"${average_cost:.4f}")

# 5. Four required EDA charts.
st.header("4. Exploratory Data Analysis")
st.plotly_chart(px.histogram(
    filtered_df, x="Memory_GiB", nbins=30,
    title="Memory Distribution", labels={"Memory_GiB": "Memory (GiB)"}
), use_container_width=True)
st.plotly_chart(px.histogram(
    filtered_df, x="vCPU_Count", title="vCPU Distribution",
    labels={"vCPU_Count": "vCPUs"}
), use_container_width=True)
with st.expander("Exact memory and vCPU frequencies"):
    st.dataframe(filtered_df["Memory_GiB"].value_counts().rename("Instances"))
    st.dataframe(filtered_df["vCPU_Count"].value_counts().rename("Instances"))
    st.write("Instances with 1–4 vCPUs:", int(filtered_df["vCPU_Count"].between(1, 4).sum()))
st.plotly_chart(px.scatter(
    filtered_df, x="vCPU_Count", y="Memory_GiB", hover_name="API Name",
    hover_data=["On Demand_USD"], title="Memory vs vCPUs"
), use_container_width=True)
priced_df = filtered_df.dropna(subset=["On Demand_USD"]).copy()
st.plotly_chart(px.scatter(
    priced_df, x="Memory_GiB", y="On Demand_USD", size="vCPU_Count",
    hover_name="API Name", title="Memory vs On-Demand Cost"
), use_container_width=True)

# 6. Hourly/monthly costs and resource efficiency.
st.header("5. EC2 Cost Analysis")
cost_columns = [
    "Name", "API Name", "Memory_GiB", "vCPU_Count", "On Demand_USD",
    "Monthly_On_Demand", "Cost_Per_GiB", "Cost_Per_vCPU"
]
st.subheader("Hourly and Monthly On-Demand Costs")
st.dataframe(filtered_df[cost_columns], use_container_width=True)
st.subheader("10 Lowest-Cost EC2 Instances")
st.dataframe(priced_df.sort_values("On Demand_USD")[cost_columns].head(10), use_container_width=True)
st.subheader("10 Highest-Cost EC2 Instances")
st.dataframe(priced_df.sort_values("On Demand_USD", ascending=False)[cost_columns].head(10), use_container_width=True)
st.subheader("15 Lowest Costs per GiB of Memory")
st.dataframe(priced_df.sort_values("Cost_Per_GiB")[cost_columns].head(15), use_container_width=True)
st.subheader("15 Lowest Costs per vCPU")
st.dataframe(priced_df.sort_values("Cost_Per_vCPU")[cost_columns].head(15), use_container_width=True)
st.subheader("15 Highest Memory-per-vCPU Ratios")
st.dataframe(
    filtered_df.sort_values("Memory_per_vCPU", ascending=False)[
        ["API Name", "Memory_GiB", "vCPU_Count", "Memory_per_vCPU"]
    ].head(15), use_container_width=True
)
st.caption("Cost per GiB and cost per vCPU are USD per hour per resource unit. "
           "Workload, architecture, network, storage, and commitments also matter.")

# 7. Pricing comparison and interactive instance selection.
st.header("6. Pricing Comparison")
st.dataframe(filtered_df[
    ["Name", "API Name"] + [column + "_USD" for column in price_columns]
], use_container_width=True)
selected_instance = st.selectbox(
    "Select an EC2 Instance", sorted(filtered_df["API Name"].unique())
)
instance = filtered_df[filtered_df["API Name"] == selected_instance].iloc[0]
st.write(f"Selected instance: **{instance['Name']} ({selected_instance})**")
pricing_data = pd.DataFrame({
    "Pricing Model": ["On Demand", "Linux Reserved", "Linux Spot"],
    "Hourly Cost": [instance["On Demand_USD"],
                    instance["Linux Reserved cost_USD"],
                    instance["Linux Spot Minimum cost_USD"]]
})
pricing_data["Monthly Cost"] = pricing_data["Hourly Cost"] * 730
st.dataframe(pricing_data, use_container_width=True)
available_prices = pricing_data.dropna(subset=["Hourly Cost"])
if available_prices.empty:
    st.info("No Linux pricing is available for this instance in the CSV.")
else:
    st.plotly_chart(px.bar(
        available_prices, x="Pricing Model", y="Hourly Cost",
        title=f"Pricing Comparison: {selected_instance}"
    ), use_container_width=True)
st.caption("Missing prices are unavailable, not free. Spot uses the CSV's minimum "
           "hourly price; multiplying by 730 is an estimate, not a guaranteed bill.")
