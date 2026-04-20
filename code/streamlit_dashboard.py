# -*- coding: utf-8 -*-
"""
Enhanced Streamlit Dashboard for YouTube Trending Analysis
More interactive controls in the sidebar
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="YouTube Trending Dashboard", layout="wide")

# -------------------------------
# Load data
# -------------------------------
DATA_PATH = r"D:/homework/youtube/youtube_trending_processed_enhanced.parquet"
df = pd.read_parquet(DATA_PATH)

# -------------------------------
# Basic derived fields
# -------------------------------
df = df.copy()
df["channel_size"] = pd.qcut(
    df["channel_total_views"],
    q=4,
    labels=["Small", "Medium", "Large", "Top"],
    duplicates="drop"
)

# safer category string
df["category_id"] = df["category_id"].astype(str)
df["region_code"] = df["region_code"].astype(str)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("Dashboard Controls")

# Section 1: country/category filters
st.sidebar.markdown("### Geographic & Content Filters")

all_countries = sorted(df["region_code"].dropna().unique().tolist())
all_categories = sorted(df["category_id"].dropna().unique().tolist())

selected_countries = st.sidebar.multiselect(
    "Select Countries",
    options=all_countries,
    default=all_countries[:5] if len(all_countries) >= 5 else all_countries
)

selected_categories = st.sidebar.multiselect(
    "Select Categories",
    options=all_categories,
    default=all_categories[:5] if len(all_categories) >= 5 else all_categories
)

# Section 2: numeric filters
st.sidebar.markdown("### Numeric Filters")

lag_min = float(df["trending_lag_hours"].quantile(0.01))
lag_max = float(df["trending_lag_hours"].quantile(0.99))
selected_lag = st.sidebar.slider(
    "Trending Lag Range (hours)",
    min_value=float(df["trending_lag_hours"].min()),
    max_value=float(df["trending_lag_hours"].max()),
    value=(lag_min, lag_max)
)

dur_min = float(df["video_duration_sec"].quantile(0.01))
dur_max = float(df["video_duration_sec"].quantile(0.99))
selected_duration = st.sidebar.slider(
    "Video Duration Range (seconds)",
    min_value=float(df["video_duration_sec"].min()),
    max_value=float(df["video_duration_sec"].max()),
    value=(dur_min, dur_max)
)

sub_min = float(df["log_subscriber_count"].quantile(0.01))
sub_max = float(df["log_subscriber_count"].quantile(0.99))
selected_subscribers = st.sidebar.slider(
    "Log Subscriber Count Range",
    min_value=float(df["log_subscriber_count"].min()),
    max_value=float(df["log_subscriber_count"].max()),
    value=(sub_min, sub_max)
)

# Section 3: channel filters
st.sidebar.markdown("### Channel Filters")

selected_channel_sizes = st.sidebar.multiselect(
    "Select Channel Size Groups",
    options=["Small", "Medium", "Large", "Top"],
    default=["Small", "Medium", "Large", "Top"]
)

# Section 4: interaction / chart controls
st.sidebar.markdown("### Visualization Controls")

engagement_option = st.sidebar.selectbox(
    "Select Engagement Metric",
    ["like_to_view", "comment_to_view"]
)

partial_feature_option = st.sidebar.selectbox(
    "Select Feature for Partial Effect",
    ["log_subscriber_count", "video_duration_sec", "like_to_view", "comment_to_view"]
)

interaction_metric = st.sidebar.selectbox(
    "Select Interaction Metric",
    ["like_to_view", "comment_to_view"]
)

country_metric = st.sidebar.selectbox(
    "Country Ranking Metric",
    ["mean_trending_lag", "mean_like_to_view", "mean_comment_to_view", "video_count"]
)

top_n_countries = st.sidebar.slider(
    "Top N Countries to Display",
    min_value=5,
    max_value=30,
    value=10
)

sample_size = st.sidebar.slider(
    "Sample Size for Scatter / Partial Effect",
    min_value=500,
    max_value=10000,
    step=500,
    value=3000
)

show_outliers = st.sidebar.checkbox("Show Outliers in Boxplots", value=False)

concentration_mode = st.sidebar.radio(
    "Concentration Curve Mode",
    ["Cumulative Share", "Top 1%-100% Share"]
)

# -------------------------------
# Apply filters
# -------------------------------
filtered_df = df.copy()

if selected_countries:
    filtered_df = filtered_df[filtered_df["region_code"].isin(selected_countries)]

if selected_categories:
    filtered_df = filtered_df[filtered_df["category_id"].isin(selected_categories)]

filtered_df = filtered_df[
    (filtered_df["trending_lag_hours"] >= selected_lag[0]) &
    (filtered_df["trending_lag_hours"] <= selected_lag[1]) &
    (filtered_df["video_duration_sec"] >= selected_duration[0]) &
    (filtered_df["video_duration_sec"] <= selected_duration[1]) &
    (filtered_df["log_subscriber_count"] >= selected_subscribers[0]) &
    (filtered_df["log_subscriber_count"] <= selected_subscribers[1]) &
    (filtered_df["channel_size"].astype(str).isin(selected_channel_sizes))
]

# -------------------------------
# Header
# -------------------------------
st.title("YouTube Trending Analysis Dashboard")
st.markdown("Interactive dashboard for trending lag, engagement, concentration, and regression relationships.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Filtered Rows", f"{len(filtered_df):,}")
col2.metric("Countries", filtered_df["region_code"].nunique())
col3.metric("Categories", filtered_df["category_id"].nunique())
col4.metric("Channels", filtered_df["channel_id"].nunique())

if filtered_df.empty:
    st.warning("No data available for the current filter selection.")
    st.stop()

# -------------------------------
# Sampled data for heavy charts
# -------------------------------
plot_df = filtered_df.dropna(subset=[
    "trending_lag_hours",
    "log_subscriber_count",
    "video_duration_sec",
    "like_to_view",
    "comment_to_view"
]).sample(min(sample_size, len(filtered_df)), random_state=42)

# -------------------------------
# 1. Trending lag distribution
# -------------------------------
st.subheader("1. Trending Lag Distribution")
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(filtered_df["trending_lag_hours"], bins=50, kde=True, ax=ax)
ax.set_xlabel("Trending Lag (hours)")
ax.set_ylabel("Count")
st.pyplot(fig)

# -------------------------------
# 2. Trending lag by country
# -------------------------------
st.subheader("2. Trending Lag by Country")
country_order = (
    filtered_df.groupby("region_code")["trending_lag_hours"]
    .median()
    .sort_values()
    .index
)

fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(
    data=filtered_df,
    x="region_code",
    y="trending_lag_hours",
    order=country_order,
    showfliers=show_outliers,
    ax=ax
)
ax.set_xlabel("Country")
ax.set_ylabel("Trending Lag (hours)")
plt.xticks(rotation=90)
st.pyplot(fig)

# -------------------------------
# 3. Trending lag by category
# -------------------------------
st.subheader("3. Trending Lag by Video Category")
category_order = (
    filtered_df.groupby("category_id")["trending_lag_hours"]
    .median()
    .sort_values()
    .index
)

fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(
    data=filtered_df,
    x="category_id",
    y="trending_lag_hours",
    order=category_order,
    showfliers=show_outliers,
    ax=ax
)
ax.set_xlabel("Video Category")
ax.set_ylabel("Trending Lag (hours)")
plt.xticks(rotation=90)
st.pyplot(fig)

# -------------------------------
# 4. Channel size vs trending speed
# -------------------------------
st.subheader("4. Channel Size vs Trending Speed")
channel_group = (
    filtered_df.groupby("channel_size", observed=False)["trending_lag_hours"]
    .agg(["count", "mean", "median"])
    .reset_index()
)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=channel_group, x="channel_size", y="mean", ax=ax)
ax.set_xlabel("Channel Size")
ax.set_ylabel("Mean Trending Lag (hours)")
st.pyplot(fig)

st.dataframe(channel_group, use_container_width=True)

# -------------------------------
# 5. Engagement by country
# -------------------------------
st.subheader("5. Cross-country Engagement Intensity")

country_engagement = (
    filtered_df.groupby("region_code")[["like_to_view", "comment_to_view"]]
    .mean()
    .reset_index()
)

if country_metric == "mean_trending_lag":
    ranked_countries = (
        filtered_df.groupby("region_code")["trending_lag_hours"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n_countries)
        .index
    )
elif country_metric == "mean_like_to_view":
    ranked_countries = (
        filtered_df.groupby("region_code")["like_to_view"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n_countries)
        .index
    )
elif country_metric == "mean_comment_to_view":
    ranked_countries = (
        filtered_df.groupby("region_code")["comment_to_view"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n_countries)
        .index
    )
else:
    ranked_countries = (
        filtered_df.groupby("region_code")["video_id"]
        .count()
        .sort_values(ascending=False)
        .head(top_n_countries)
        .index
    )

country_engagement_top = country_engagement[country_engagement["region_code"].isin(ranked_countries)]

fig, ax = plt.subplots(figsize=(14, 6))
melted = country_engagement_top.melt(
    id_vars="region_code",
    value_vars=["like_to_view", "comment_to_view"]
)
sns.barplot(data=melted, x="region_code", y="value", hue="variable", ax=ax)
ax.set_xlabel("Country")
ax.set_ylabel("Average Engagement Ratio")
plt.xticks(rotation=90)
st.pyplot(fig)

# -------------------------------
# 6. Interaction analysis
# -------------------------------
st.subheader("6. Interaction Effect: Subscriber × Engagement")

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(
    plot_df["log_subscriber_count"],
    plot_df["trending_lag_hours"],
    c=plot_df[interaction_metric],
    alpha=0.5
)
cbar = plt.colorbar(scatter)
cbar.set_label(interaction_metric)
ax.set_xlabel("Log Subscriber Count")
ax.set_ylabel("Trending Lag (hours)")
ax.set_title(f"Interaction: Subscriber × {interaction_metric}")
st.pyplot(fig)

# optional interaction regression
interaction_formula = f"trending_lag_hours ~ log_subscriber_count * {interaction_metric}"
interaction_model = smf.ols(interaction_formula, data=plot_df).fit()

with st.expander("Show Interaction Regression Summary"):
    st.text(interaction_model.summary())

# -------------------------------
# 7. Partial effect plot
# -------------------------------
st.subheader("7. Partial Effect Plot")

features = [
    "log_subscriber_count",
    "video_duration_sec",
    "like_to_view",
    "comment_to_view"
]

others = [f for f in features if f != partial_feature_option]

# manual partial regression for speed and cleaner plot
y_model = sm.OLS(
    plot_df["trending_lag_hours"],
    sm.add_constant(plot_df[others])
).fit()
y_resid = y_model.resid

x_model = sm.OLS(
    plot_df[partial_feature_option],
    sm.add_constant(plot_df[others])
).fit()
x_resid = x_model.resid

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(x_resid, y_resid, alpha=0.4)

line_model = sm.OLS(y_resid, sm.add_constant(x_resid)).fit()
x_sorted = pd.Series(x_resid).sort_values()
y_pred = line_model.predict(sm.add_constant(x_sorted))
ax.plot(x_sorted, y_pred)

ax.set_xlabel(f"Residualized {partial_feature_option}")
ax.set_ylabel("Residualized trending_lag_hours")
ax.set_title(f"Partial Effect of {partial_feature_option}")
st.pyplot(fig)

# -------------------------------
# 8. Top channels concentration curve
# -------------------------------
st.subheader("8. Top Channels Concentration Curve")

channel_counts = filtered_df.groupby("channel_id").size().sort_values(ascending=False)

if len(channel_counts) > 0:
    total_videos = channel_counts.sum()
    sorted_counts = channel_counts.values
    cumulative_videos = np.cumsum(sorted_counts)

    if concentration_mode == "Cumulative Share":
        y_values = cumulative_videos / total_videos * 100
        y_label = "Cumulative Percent of Trending Videos"
        title = "Top Channels Concentration Curve"
    else:
        y_values = sorted_counts / total_videos * 100
        y_label = "Percent of Trending Videos"
        title = "Top 1%-100% Channel Share Curve"

    x_values = np.linspace(0, 100, len(sorted_counts))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x_values, y_values, linewidth=2, label="Observed")
    if concentration_mode == "Cumulative Share":
        ax.plot([0, 100], [0, 100], linestyle="--", label="Equality Line")
    ax.set_xlabel("Percent of Channels")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    top_1pct = max(1, int(len(channel_counts) * 0.01))
    top_5pct = max(1, int(len(channel_counts) * 0.05))

    st.write(f"Top 1% channels account for {channel_counts.head(top_1pct).sum() / total_videos:.2%} of trending videos")
    st.write(f"Top 5% channels account for {channel_counts.head(top_5pct).sum() / total_videos:.2%} of trending videos")
else:
    st.info("No channels available for concentration analysis.")


# -------------------------------
# 9.1 Gini / HHI by Country
# -------------------------------
st.subheader("Gini / HHI by Country")

def gini(array):
    array = np.array(array, dtype=np.float64)
    if len(array) == 0 or np.sum(array) == 0:
        return 0.0
    if np.amin(array) < 0:
        array -= np.amin(array)
    array = np.sort(array)
    n = len(array)
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))

def hhi(array):
    array = np.array(array, dtype=np.float64)
    if len(array) == 0 or np.sum(array) == 0:
        return 0.0
    shares = array / np.sum(array)
    return np.sum(shares ** 2)

country_metrics = []

for country in sorted(filtered_df["region_code"].dropna().unique()):
    country_df = filtered_df[filtered_df["region_code"] == country]
    channel_counts_country = country_df.groupby("channel_id").size().values

    country_metrics.append({
        "region_code": country,
        "Gini": gini(channel_counts_country),
        "HHI": hhi(channel_counts_country),
        "video_count": len(country_df),
        "channel_count": country_df["channel_id"].nunique()
    })

country_metrics_df = pd.DataFrame(country_metrics)

if not country_metrics_df.empty:
    country_sort_metric = st.selectbox(
        "Sort Countries By",
        ["Gini", "HHI", "video_count", "channel_count"],
        index=0,
        key="country_metric_sort"
    )

    ascending_flag = st.checkbox(
        "Sort Ascending",
        value=False,
        key="country_metric_sort_order"
    )

    country_metrics_df = country_metrics_df.sort_values(
        by=country_sort_metric,
        ascending=ascending_flag
    )

    st.dataframe(country_metrics_df, use_container_width=True)

    metric_plot_mode = st.radio(
        "Country Concentration Plot Mode",
        ["Bar Chart", "Line Chart"],
        horizontal=True,
        key="country_metric_plot_mode"
    )

    metric_y_scale = st.radio(
        "Y-axis Scale",
        ["Linear", "Log"],
        horizontal=True,
        key="country_metric_y_scale"
    )

    plot_df = country_metrics_df.copy()
    plot_df["HHI_plot"] = plot_df["HHI"].clip(lower=1e-6)
    plot_df["Gini_plot"] = plot_df["Gini"].clip(lower=1e-6)

    if metric_plot_mode == "Bar Chart":
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(plot_df))
        width = 0.38

        ax.bar(x - width/2, plot_df["Gini_plot"], width=width, label="Gini")
        ax.bar(x + width/2, plot_df["HHI_plot"], width=width, label="HHI")

        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["region_code"], rotation=90)
        ax.set_xlabel("Country")
        ax.set_ylabel("Metric Value")
        ax.set_title("Gini / HHI by Country")
        ax.legend()

        if metric_y_scale == "Log":
            ax.set_yscale("log")

        st.pyplot(fig)

    else:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(plot_df["region_code"], plot_df["Gini_plot"], marker="o", label="Gini")
        ax.plot(plot_df["region_code"], plot_df["HHI_plot"], marker="o", label="HHI")

        ax.set_xlabel("Country")
        ax.set_ylabel("Metric Value")
        ax.set_title("Gini / HHI by Country")
        ax.legend()
        plt.xticks(rotation=90)

        if metric_y_scale == "Log":
            ax.set_yscale("log")

        st.pyplot(fig)
else:
    st.info("No country-level concentration metrics available.")

# -------------------------------
# 9.2 Concentration Metrics Trend Under Filters
# -------------------------------
st.subheader("Concentration Metrics Trend Under Filters")

trend_variable = st.selectbox(
    "Select Variable for Trend Analysis",
    ["log_subscriber_count", "video_duration_sec", "trending_lag_hours"],
    key="trend_variable"
)

num_bins = st.slider(
    "Number of Bins",
    min_value=4,
    max_value=12,
    value=6,
    step=1,
    key="trend_bins"
)

trend_df = filtered_df.dropna(subset=[trend_variable, "channel_id"]).copy()

if len(trend_df) > 0:
    trend_df["trend_bin"] = pd.qcut(
        trend_df[trend_variable],
        q=num_bins,
        duplicates="drop"
    )

    trend_metrics = []

    for bin_name, sub_df in trend_df.groupby("trend_bin", observed=False):
        channel_counts_bin = sub_df.groupby("channel_id").size().values

        trend_metrics.append({
            "bin": str(bin_name),
            "midpoint": sub_df[trend_variable].median(),
            "Gini": gini(channel_counts_bin),
            "HHI": hhi(channel_counts_bin),
            "video_count": len(sub_df),
            "channel_count": sub_df["channel_id"].nunique()
        })

    trend_metrics_df = pd.DataFrame(trend_metrics).sort_values("midpoint")

    st.dataframe(trend_metrics_df, use_container_width=True)

    trend_y_scale = st.radio(
        "Trend Plot Y-axis Scale",
        ["Linear", "Log"],
        horizontal=True,
        key="trend_metric_y_scale"
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    gini_plot = trend_metrics_df["Gini"].clip(lower=1e-6)
    hhi_plot = trend_metrics_df["HHI"].clip(lower=1e-6)

    ax.plot(
        trend_metrics_df["bin"],
        gini_plot,
        marker="o",
        linewidth=2,
        label="Gini"
    )
    ax.plot(
        trend_metrics_df["bin"],
        hhi_plot,
        marker="o",
        linewidth=2,
        label="HHI"
    )

    ax.set_xlabel(f"Binned {trend_variable}")
    ax.set_ylabel("Concentration Metric")
    ax.set_title(f"Concentration Metrics Across {trend_variable} Bins")
    ax.legend()
    plt.xticks(rotation=45, ha="right")

    if trend_y_scale == "Log":
        ax.set_yscale("log")

    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(
        trend_metrics_df["bin"],
        trend_metrics_df["video_count"],
        marker="o"
    )
    ax2.set_xlabel(f"Binned {trend_variable}")
    ax2.set_ylabel("Video Count")
    ax2.set_title(f"Video Count Across {trend_variable} Bins")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig2)
else:
    st.info("No data available for trend analysis.")

# -------------------------------
# 10. Cross-country category dominance
# -------------------------------
st.subheader("10. Cross-country Category Dominance")

country_cat_counts = (
    filtered_df.groupby(["region_code", "category_id"])["video_id"]
    .count()
    .reset_index()
)
country_totals = (
    filtered_df.groupby("region_code")["video_id"]
    .count()
    .reset_index()
    .rename(columns={"video_id": "total_videos"})
)
country_cat_counts = country_cat_counts.merge(country_totals, on="region_code")
country_cat_counts["percentage"] = (
    country_cat_counts["video_id"] / country_cat_counts["total_videos"]
)

pivot_df = country_cat_counts.pivot(
    index="region_code",
    columns="category_id",
    values="percentage"
).fillna(0)

if len(pivot_df) > 0:
    fig, ax = plt.subplots(figsize=(16, 6))
    pivot_df.plot(kind="bar", stacked=True, ax=ax)
    ax.set_xlabel("Country")
    ax.set_ylabel("Share of Trending Videos")
    ax.set_title("Cross-country Video Category Dominance")
    plt.xticks(rotation=90)
    plt.legend(title="Category ID", bbox_to_anchor=(1.05, 1), loc="upper left")
    st.pyplot(fig)
else:
    st.info("No category dominance data available.")

st.markdown("Use the sidebar controls to explore different countries, categories, channel groups, and interaction effects.")

# -------------------------------
# Interaction Analysis: Subscriber × Engagement
# -------------------------------
st.subheader("Interaction Effect: Subscriber x Engagement")

engagement_option = st.selectbox(
    "Select Engagement Metric:",
    ["like_to_view", "comment_to_view"]
)

sample_df = filtered_df.sample(min(5000, len(filtered_df)), random_state=42)

fig, ax = plt.subplots(figsize=(8,6))
scatter = ax.scatter(
    sample_df["log_subscriber_count"],
    sample_df["trending_lag_hours"],
    c=sample_df[engagement_option],
    cmap="viridis",
    alpha=0.5
)

cbar = plt.colorbar(scatter)
cbar.set_label(engagement_option)

ax.set_xlabel("Log Subscriber Count")
ax.set_ylabel("Trending Lag (hours)")
ax.set_title(f"Interaction: Subscriber × {engagement_option}")

st.pyplot(fig)

# -------------------------------
# Interaction Analysis: Subscriber × Engagement
# -------------------------------

import statsmodels.formula.api as smf

st.subheader("Regression with Interaction Term")

formula = f"trending_lag_hours ~ log_subscriber_count * {engagement_option}"

interaction_model = smf.ols(formula=formula, data=sample_df).fit()

st.text(interaction_model.summary())


# -------------------------------
# Partial Effect Plot (Interactive)
# -------------------------------
import statsmodels.api as sm

st.subheader("Partial Effect Plot")

feature_option = st.selectbox(
    "Select Feature for Partial Effect:",
    ["log_subscriber_count", "video_duration_sec", "like_to_view", "comment_to_view"]
)

sample_df = filtered_df.sample(min(1000, len(filtered_df)), random_state=42)

fig, ax = plt.subplots(figsize=(6,5))

sm.graphics.plot_partregress(
    endog="trending_lag_hours",
    exog_i=feature_option,
    exog_others=[f for f in ["log_subscriber_count", "video_duration_sec", "like_to_view", "comment_to_view"] if f != feature_option],
    data=sample_df,
    ax=ax,
    obs_labels=False
)

plt.title(f"Partial Effect of {feature_option}")
st.pyplot(fig)