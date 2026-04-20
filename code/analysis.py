# -*- coding: utf-8 -*-
"""
YouTube Trending Final Analysis Script (Enhanced + Cross-country Engagement + Top Channel Concentration + Gini/HHI)
- Based on enhanced preprocessed Parquet file
- Includes video duration, channel authority, channel age
- Performs correlation, regression, country/category comparison, channel size analysis
- Adds cross-country engagement intensity, top channel concentration
- Adds Gini coefficient and Herfindahl-Hirschman Index (HHI) for trending distribution
- Saves all plots for reporting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import os
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
import scipy.stats as stats
# -------------------------------
# 1. Load enhanced preprocessed data
# -------------------------------
parquet_path = r"D:/homework/youtube/youtube_trending_processed_enhanced.parquet"
if not os.path.exists(parquet_path):
    raise FileNotFoundError(f"{parquet_path} not found. Please run the enhanced preprocessing script first.")

df = pd.read_parquet(parquet_path)
print("Enhanced data loaded:", df.shape)

# -------------------------------
# 2. Correlation Analysis
# -------------------------------
numeric_cols = [
    "view_count","comment_count","like_count","log_views","log_comments",
    "comment_to_view","like_to_view","video_duration_sec","log_subscriber_count",
    "channel_age_years","trending_lag_hours"
]

corr_matrix = df[numeric_cols].corr()
print("Correlation matrix (trending_lag_hours):")
print(corr_matrix["trending_lag_hours"].sort_values(ascending=False))

plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix of Numerical Features")
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=300)
plt.close()

# -------------------------------
# 3. Regression Analysis
# -------------------------------
formula = (
    "trending_lag_hours ~ video_duration_sec + log_subscriber_count + "
    "log_views + log_comments + comment_to_view + like_to_view + "
    "C(category_id) + C(region_code)"
)
model = smf.ols(formula=formula, data=df).fit()
print("Regression Summary:")
print(model.summary())

# -------------------------------
# 3.1 Regression Diagnostics
# -------------------------------

print("\n--- Regression Diagnostics ---")

# Prepare X matrix (remove categorical for VIF)
X = df[[
    "video_duration_sec",
    "log_subscriber_count",
    "log_views",
    "log_comments",
    "comment_to_view",
    "like_to_view"
]].dropna()

X = sm.add_constant(X)

# -------------------------------
# 3.2. Multicollinearity (VIF)
# -------------------------------
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("\nVIF (Multicollinearity check):")
print(vif_data)

# -------------------------------
# 3.3. Heteroskedasticity (Breusch–Pagan)
# -------------------------------
y = df.loc[X.index, "trending_lag_hours"]

model_diag = sm.OLS(y, X).fit()

bp_test = het_breuschpagan(model_diag.resid, model_diag.model.exog)

bp_labels = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
print("\nBreusch–Pagan test:")
for name, val in zip(bp_labels, bp_test):
    print(f"{name}: {val}")

# -------------------------------
# 3.4 Residual Plot
# -------------------------------
plt.figure(figsize=(8,6))
plt.scatter(model_diag.fittedvalues, model_diag.resid, alpha=0.3)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig("residual_plot.png", dpi=300)
plt.close()

# -------------------------------
# 3.5. QQ Plot
# -------------------------------
plt.figure(figsize=(6,6))
stats.probplot(model_diag.resid, dist="norm", plot=plt)
plt.title("QQ Plot of Residuals")
plt.tight_layout()
plt.savefig("qq_plot.png", dpi=300)
plt.close()

# -------------------------------
# 3.6 Partial Effect Plot
# -------------------------------
import statsmodels.api as sm
import matplotlib.pyplot as plt

print("\nGenerating Partial Effect Plots...")

features = [
    "log_subscriber_count",
    "video_duration_sec",
    "like_to_view",
    "comment_to_view"
]

plot_df = df[
    ["trending_lag_hours"] + features
].dropna().sample(min(1000, len(df)), random_state=42)

for feature in features:
    fig, ax = plt.subplots(figsize=(6, 5))

    sm.graphics.plot_partregress(
        endog="trending_lag_hours",
        exog_i=feature,
        exog_others=[f for f in features if f != feature],
        data=plot_df,
        ax=ax,
        obs_labels=False
    )

    ax.set_title(f"Partial Effect of {feature}")
    plt.tight_layout()
    plt.savefig(f"partial_effect_{feature}.png", dpi=300)
    plt.close()


# -------------------------------
# 4. Country / Video Category Comparison
# -------------------------------
print("\nCountry / Video Category Comparison")

plt.figure(figsize=(18,6))
sns.boxplot(data=df, x="region_code", y="trending_lag_hours")
plt.xticks(rotation=90)
plt.ylabel("Trending Lag (hours)")
plt.xlabel("Country")
plt.title("Trending Lag Distribution by Country")
plt.tight_layout()
plt.savefig("trending_lag_by_country.png", dpi=300)
plt.close()

plt.figure(figsize=(12,6))
sns.boxplot(data=df, x="category_id", y="trending_lag_hours")
plt.xticks(rotation=90)
plt.ylabel("Trending Lag (hours)")
plt.xlabel("Video Category")
plt.title("Trending Lag Distribution by Video Category")
plt.tight_layout()
plt.savefig("trending_lag_by_category.png", dpi=300)
plt.close()

# -------------------------------
# 5. Channel Size Analysis (Traffic Concentration)
# -------------------------------
df["channel_size"] = pd.qcut(df["channel_total_views"], q=4, labels=["Small","Medium","Large","Top"])
grouped = df.groupby("channel_size")["trending_lag_hours"].agg(["count","mean","median"]).reset_index()
print("Average Trending Lag by Channel Size:")
print(grouped)

plt.figure(figsize=(8,6))
sns.barplot(data=grouped, x="channel_size", y="mean", palette="Blues_d")
plt.ylabel("Mean Trending Lag (hours)")
plt.xlabel("Channel Size (by total views)")
plt.title("Mean Trending Lag by Channel Size")
plt.tight_layout()
plt.savefig("trending_lag_by_channel_size.png", dpi=300)
plt.close()

# -------------------------------
# 6. Cross-country engagement intensity
# -------------------------------
country_engagement = df.groupby("region_code")[["comment_to_view","like_to_view"]].mean().reset_index()
print("Cross-country engagement intensity (like/view, comment/view):")
print(country_engagement)

plt.figure(figsize=(16,6))
sns.barplot(data=country_engagement.melt(id_vars="region_code", value_vars=["like_to_view","comment_to_view"]),
            x="region_code", y="value", hue="variable")
plt.xticks(rotation=90)
plt.ylabel("Engagement ratio")
plt.xlabel("Country")
plt.title("Average Engagement Ratio by Country")
plt.tight_layout()
plt.savefig("cross_country_engagement.png", dpi=300)
plt.close()

# -------------------------------
# 7. Top Channels Concentration (Line plot)
# -------------------------------
channel_trending_count = df.groupby("channel_id")["video_id"].count().sort_values(ascending=False)
total_trending_videos = channel_trending_count.sum()

# Compute cumulative percentage for Lorenz-style line
sorted_counts = channel_trending_count.values
cumulative_videos = np.cumsum(sorted_counts)
cumulative_percentage = cumulative_videos / total_trending_videos * 100  # Y-axis: cumulative trending videos percentage
percent_channels = np.linspace(0, 100, len(sorted_counts))                 # X-axis: percent of channels

print(f"Top 1% channels account for {channel_trending_count.head(int(len(channel_trending_count)*0.01)).sum()/total_trending_videos:.2%} of trending videos")
print(f"Top 5% channels account for {channel_trending_count.head(int(len(channel_trending_count)*0.05)).sum()/total_trending_videos:.2%} of trending videos")

# Plot cumulative line
plt.figure(figsize=(10,6))
plt.plot(percent_channels, cumulative_percentage, color="blue", linewidth=2)
plt.plot([0,100],[0,100], color="red", linestyle="--")  # 45-degree equality line
plt.xlabel("Percent of Channels")
plt.ylabel("Cumulative Percent of Trending Videos")
plt.title("Top Channels Concentration Curve (1%-100%)")
plt.grid(True)
plt.tight_layout()
plt.savefig("top_channels_concentration_line.png", dpi=300)
plt.close()

# -------------------------------
# 8. Cross-country category dominance
# -------------------------------
country_category_counts = df.groupby(["region_code","category_id"])["video_id"].count().reset_index()
country_totals = df.groupby("region_code")["video_id"].count().reset_index().rename(columns={"video_id":"total_videos"})
country_category_counts = country_category_counts.merge(country_totals, on="region_code")
country_category_counts["percentage"] = country_category_counts["video_id"] / country_category_counts["total_videos"]

print("Cross-country category dominance (sample):")
print(country_category_counts.head(20))

pivot_df = country_category_counts.pivot(index="region_code", columns="category_id", values="percentage").fillna(0)
pivot_df.plot(kind="bar", stacked=True, figsize=(18,6), width=0.8)
plt.ylabel("Share of Trending Videos")
plt.xlabel("Country")
plt.title("Cross-country Video Category Dominance")
plt.xticks(rotation=90)
plt.legend(title="Category ID", bbox_to_anchor=(1.05,1), loc="upper left")
plt.tight_layout()
plt.savefig("cross_country_category_dominance.png", dpi=300)
plt.close()

# -------------------------------
# 9. Gini coefficient and HHI
# -------------------------------
def gini(array):
    """Calculate Gini coefficient of array of values"""
    array = np.array(array, dtype=np.float64)
    if np.amin(array) < 0:
        array -= np.amin(array)  # values cannot be negative
    array = np.sort(array)
    n = len(array)
    index = np.arange(1, n+1)
    return (np.sum((2*index - n - 1)*array)) / (n * np.sum(array))

def hhi(array):
    """Calculate Herfindahl-Hirschman Index"""
    shares = array / np.sum(array)
    return np.sum(shares**2)

# Use channel_trending_count to calculate Gini and HHI
channel_counts = channel_trending_count.values
gini_value = gini(channel_counts)
hhi_value = hhi(channel_counts)

print(f"Gini coefficient of trending distribution: {gini_value:.4f}")
print(f"Herfindahl-Hirschman Index (HHI): {hhi_value:.4f}")

# Plot Gini & HHI as bar chart

plt.figure(figsize=(6,6))

values = [gini_value, hhi_value]

# 防止 log(0) 报错
values = [max(v, 1e-6) for v in values]

plt.bar(["Gini", "HHI"], values, color=["skyblue", "orange"])

plt.yscale("log")  # 👈 关键：对数坐标

plt.ylabel("Log Scale (Concentration Metric)")
plt.title("Trending Video Concentration Metrics (Log Scale)")

plt.tight_layout()
plt.savefig("gini_hhi_concentration_log.png", dpi=300)
plt.close()

# -------------------------------
# 10. Completion message
# -------------------------------
print("Full enhanced analysis completed. All plots have been saved in the current directory, including Gini and HHI metrics.")