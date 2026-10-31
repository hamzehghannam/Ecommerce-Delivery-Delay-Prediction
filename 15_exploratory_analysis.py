
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load eligible modeling data
df = pd.read_csv(
    PROCESSED_DIR / "modeling_data.csv"
)

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("Total eligible orders:", len(df))

# Count on-time and late deliveries
target_counts = (
    df["is_late"]
    .value_counts()
    .reindex([0, 1], fill_value=0)
)

print("\nDelivery status distribution:")
print(target_counts.to_string())

print("\nDelivery percentages:")
print(
    (target_counts / len(df) * 100)
    .round(2)
    .to_string()
)

# Create the first chart
fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    ["On Time", "Late"],
    target_counts.values
)

ax.set_title("Delivery Delay Distribution")
ax.set_xlabel("Delivery Status")
ax.set_ylabel("Number of Orders")

for bar, count in zip(bars, target_counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{count:,}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

output_path = FIGURES_DIR / "01_delivery_distribution.png"

plt.savefig(output_path, dpi=300)
plt.close()

print("\nFigure saved:", output_path)
print("\nEXPLORATORY ANALYSIS COMPLETED")
# ============================================================
# QUESTION 2: How does delay rate change over time?
# ============================================================

df["order_approved_at"] = pd.to_datetime(
    df["order_approved_at"],
    errors="coerce"
)

df["approval_month_period"] = (
    df["order_approved_at"]
    .dt.to_period("M")
)

monthly_delay = (
    df.groupby("approval_month_period")
    .agg(
        total_orders=("order_id", "count"),
        late_orders=("is_late", "sum"),
    )
    .reset_index()
)

monthly_delay["delay_rate"] = (
    monthly_delay["late_orders"]
    / monthly_delay["total_orders"]
) * 100

monthly_delay["approval_month"] = (
    monthly_delay["approval_month_period"]
    .astype(str)
)

print("\nMONTHLY DELAY ANALYSIS:")
print(
    monthly_delay[
        ["approval_month", "total_orders", "late_orders", "delay_rate"]
    ]
    .head(12)
    .to_string(index=False)
)

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    monthly_delay["approval_month"],
    monthly_delay["delay_rate"],
    marker="o"
)

ax.set_title("Monthly Delivery Delay Rate")
ax.set_xlabel("Approval Month")
ax.set_ylabel("Delay Rate (%)")
ax.tick_params(axis="x", rotation=45)

plt.tight_layout()

output_path = FIGURES_DIR / "02_monthly_delay_rate.png"
plt.savefig(output_path, dpi=300)
plt.close()

print("\nFigure saved:", output_path)
# ============================================================
# QUESTION 3: Which customer states have the highest delay rate?
# ============================================================

state_delay = (
    df.groupby("customer_state")
    .agg(
        total_orders=("order_id", "count"),
        late_orders=("is_late", "sum"),
    )
    .reset_index()
)

state_delay["delay_rate"] = (
    state_delay["late_orders"]
    / state_delay["total_orders"]
) * 100

# Keep only states with a reasonable number of orders
state_delay_filtered = state_delay[
    state_delay["total_orders"] >= 500
].copy()

top_delay_states = (
    state_delay_filtered
    .sort_values("delay_rate", ascending=False)
    .head(10)
    .sort_values("delay_rate", ascending=True)
)

print("\nTOP STATES BY DELAY RATE:")
print(
    top_delay_states[
        ["customer_state", "total_orders", "late_orders", "delay_rate"]
    ].to_string(index=False)
)

fig, ax = plt.subplots(figsize=(9, 6))

bars = ax.barh(
    top_delay_states["customer_state"],
    top_delay_states["delay_rate"]
)

ax.set_title("Top 10 Customer States by Delay Rate")
ax.set_xlabel("Delay Rate (%)")
ax.set_ylabel("Customer State")

for bar, rate in zip(bars, top_delay_states["delay_rate"]):
    ax.text(
        bar.get_width(),
        bar.get_y() + bar.get_height() / 2,
        f"{rate:.2f}%",
        va="center"
    )

plt.tight_layout()

output_path = FIGURES_DIR / "03_top_states_delay_rate.png"
plt.savefig(output_path, dpi=300)
plt.close()

print("\nFigure saved:", output_path)

# ============================================================
# QUESTION 4: Does order size affect delivery delays?
# ============================================================

df["order_size_group"] = pd.cut(
    df["item_count"],
    bins=[0, 1, 2, 5, float("inf")],
    labels=["1 item", "2 items", "3-5 items", "6+ items"],
    include_lowest=True
)

size_delay = (
    df.groupby("order_size_group", observed=True)
    .agg(
        total_orders=("order_id", "count"),
        late_orders=("is_late", "sum")
    )
    .reset_index()
)

size_delay["delay_rate"] = (
    size_delay["late_orders"]
    / size_delay["total_orders"] * 100
)

print("\nORDER SIZE DELAY ANALYSIS:")
print(size_delay.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.bar(
    size_delay["order_size_group"].astype(str),
    size_delay["delay_rate"]
)

ax.set_title("Delivery Delay Rate by Order Size")
ax.set_xlabel("Number of Items in Order")
ax.set_ylabel("Delay Rate (%)")

for bar, rate in zip(bars, size_delay["delay_rate"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{rate:.2f}%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

output_path = FIGURES_DIR / "04_order_size_delay.png"
plt.savefig(output_path, dpi=300)
plt.close()

print("\nFigure saved:", output_path)

# ============================================================
# QUESTION 5: Does order value differ by delivery status?
# ============================================================

value_analysis = df[
    ["is_late", "order_total_value"]
].dropna().copy()

print("\nORDER VALUE ANALYSIS:")

value_summary = (
    value_analysis.groupby("is_late")["order_total_value"]
    .agg(["count", "mean", "median", "std"])
    .round(2)
)

print(value_summary.to_string())

# Limit the plot to the 99th percentile for readability
upper_limit = value_analysis["order_total_value"].quantile(0.99)

plot_data = value_analysis[
    value_analysis["order_total_value"] <= upper_limit
]

fig, ax = plt.subplots(figsize=(9, 5))

ax.boxplot(
    [
        plot_data.loc[
            plot_data["is_late"] == 0,
            "order_total_value"
        ],
        plot_data.loc[
            plot_data["is_late"] == 1,
            "order_total_value"
        ]
    ],
    tick_labels=["On Time", "Late"],
    showfliers=False
)

ax.set_title("Order Value by Delivery Status")
ax.set_xlabel("Delivery Status")
ax.set_ylabel("Order Value")

plt.tight_layout()

output_path = FIGURES_DIR / "05_order_value_boxplot.png"
plt.savefig(output_path, dpi=300)
plt.close()

print("\nFigure saved:", output_path)
print("99th percentile visualization limit:", round(upper_limit, 2))

# ============================================================
# QUESTION 6: Planned delivery duration vs. delay status
# ============================================================

duration_analysis = df[
    ["is_late", "planned_delivery_days"]
].dropna().copy()

print("\nPLANNED DELIVERY DURATION ANALYSIS:")

duration_summary = (
    duration_analysis
    .groupby("is_late")["planned_delivery_days"]
    .agg(["count", "mean", "median", "std"])
    .round(2)
)

print(duration_summary.to_string())

# Histogram with the same bins for both groups
bins = range(0, 91, 5)

fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(
    duration_analysis.loc[
        duration_analysis["is_late"] == 0,
        "planned_delivery_days"
    ],
    bins=bins,
    alpha=0.6,
    density=True,
    label="On Time"
)

ax.hist(
    duration_analysis.loc[
        duration_analysis["is_late"] == 1,
        "planned_delivery_days"
    ],
    bins=bins,
    alpha=0.6,
    density=True,
    label="Late"
)

ax.set_title("Planned Delivery Duration by Delivery Status")
ax.set_xlabel("Planned Delivery Duration (Days)")
ax.set_ylabel("Probability Density")
ax.legend()

plt.tight_layout()

output_path = FIGURES_DIR / "06_planned_duration_histogram.png"
plt.savefig(output_path, dpi=300)
plt.close()

outside_range = (
    (duration_analysis["planned_delivery_days"] < 0)
    | (duration_analysis["planned_delivery_days"] >= 90)
).sum()

print("\nFigure saved:", output_path)
print("Orders outside plotted range:", outside_range)

# ============================================================
# QUESTION 7: Delay rate by approval day and hour
# ============================================================

import seaborn as sns

day_names = [
    "Mon", "Tue", "Wed", "Thu",
    "Fri", "Sat", "Sun"
]

heatmap_data = df[
    ["approval_day_of_week", "approval_hour", "is_late"]
].dropna().copy()

heatmap_data["approval_day_of_week"] = (
    heatmap_data["approval_day_of_week"].astype(int)
)

heatmap_data["approval_hour"] = (
    heatmap_data["approval_hour"].astype(int)
)

# Number of orders in each day-hour combination
heatmap_counts = pd.pivot_table(
    heatmap_data,
    index="approval_day_of_week",
    columns="approval_hour",
    values="is_late",
    aggfunc="count",
    observed=True
)

# Average of binary target = proportion of late orders
heatmap_rates = pd.pivot_table(
    heatmap_data,
    index="approval_day_of_week",
    columns="approval_hour",
    values="is_late",
    aggfunc="mean",
    observed=True
) * 100

heatmap_rates = heatmap_rates.reindex(
    index=range(7),
    columns=range(24)
)

heatmap_counts = heatmap_counts.reindex(
    index=range(7),
    columns=range(24)
)

# Hide cells containing fewer than 30 orders
heatmap_rates = heatmap_rates.mask(
    heatmap_counts < 30
)

print("\nAPPROVAL DAY-HOUR ANALYSIS:")
print("Orders analyzed:", len(heatmap_data))
print(
    "Day-hour groups with at least 30 orders:",
    int((heatmap_counts >= 30).sum().sum())
)

fig, ax = plt.subplots(figsize=(14, 6))

sns.heatmap(
    heatmap_rates,
    cmap="YlOrRd",
    ax=ax,
    cbar_kws={"label": "Delay Rate (%)"}
)

ax.set_title("Delivery Delay Rate by Approval Day and Hour")
ax.set_xlabel("Approval Hour")
ax.set_ylabel("Approval Day")
ax.set_yticklabels(day_names, rotation=0)

plt.tight_layout()

output_path = FIGURES_DIR / "07_approval_day_hour_heatmap.png"
plt.savefig(output_path, dpi=300)
plt.close()

print("\nFigure saved:", output_path)

# ============================================================
# QUESTION 8: Delivery status by number of sellers
# ============================================================

seller_analysis = df[
    ["seller_count", "is_late"]
].dropna().copy()

seller_analysis["seller_group"] = seller_analysis[
    "seller_count"
].apply(
    lambda x: "1 seller" if x == 1 else "2+ sellers"
)

seller_summary = pd.crosstab(
    seller_analysis["seller_group"],
    seller_analysis["is_late"]
).reindex(
    index=["1 seller", "2+ sellers"],
    columns=[0, 1],
    fill_value=0
)

seller_summary.columns = ["On Time", "Late"]

seller_percent = (
    seller_summary.div(
        seller_summary.sum(axis=1),
        axis=0
    ) * 100
)

print("\nSELLER COUNT DELAY ANALYSIS:")
print(seller_summary.to_string())

print("\nSELLER COUNT DELAY PERCENTAGES:")
print(seller_percent.round(2).to_string())

fig, ax = plt.subplots(figsize=(9, 5))

seller_percent.plot(
    kind="bar",
    stacked=True,
    ax=ax
)

ax.set_title("Delivery Status by Number of Sellers")
ax.set_xlabel("Number of Sellers")
ax.set_ylabel("Percentage of Orders (%)")
ax.set_ylim(0, 100)
ax.legend(title="Delivery Status")

plt.xticks(rotation=0)
plt.tight_layout()

output_path = FIGURES_DIR / "08_seller_count_stacked_bar.png"
plt.savefig(output_path, dpi=300)
plt.close()

print("\nFigure saved:", output_path)
print("\nALL 8 EXPLORATORY FIGURES GENERATED")

# ============================================================
# STEP 50: Save reproducible analytical tables
# ============================================================

TABLES_DIR = BASE_DIR / "outputs" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

monthly_delay.to_csv(
    TABLES_DIR / "monthly_delay_analysis.csv",
    index=False
)

state_delay.to_csv(
    TABLES_DIR / "state_delay_analysis.csv",
    index=False
)

size_delay.to_csv(
    TABLES_DIR / "order_size_delay_analysis.csv",
    index=False
)

value_summary.to_csv(
    TABLES_DIR / "order_value_summary.csv"
)

duration_summary.to_csv(
    TABLES_DIR / "planned_duration_summary.csv"
)

heatmap_rates.to_csv(
    TABLES_DIR / "approval_day_hour_delay_rates.csv"
)

heatmap_counts.to_csv(
    TABLES_DIR / "approval_day_hour_counts.csv"
)

seller_summary.to_csv(
    TABLES_DIR / "seller_count_summary.csv"
)

seller_percent.to_csv(
    TABLES_DIR / "seller_count_percentages.csv"
)

print("\nANALYTICAL TABLES SAVED SUCCESSFULLY")
print("Tables directory:", TABLES_DIR)
print("Total tables saved: 9")
