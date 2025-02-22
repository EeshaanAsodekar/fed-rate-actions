import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load raw macroeconomic data
macro_df = pd.read_csv(r"data/raw/raw_macro_market_data.csv")
macro_df["date"] = pd.to_datetime(macro_df.iloc[:, 0])

# Load FOMC rate moves
fomc_df = pd.read_csv(r"data/processed/fomc_meeting_rate_moves.csv")
fomc_df["date"] = pd.to_datetime(fomc_df["date"])

# **Step 1: Extract Required Inflation Columns**
macro_columns_subset = {
    "CPI (All Urban Consumers)": "CPIAUCSL",
    "Core CPI (Ex Food & Energy)": "CPILFESL",
    "PCE Inflation": "PCEPI",
    "Core PCE Inflation (Ex Food & Energy)": "PCEPILFE",
    "Trimmed Mean PCE Inflation Rate": "PCETRIM12M159SFRBDAL",
    "Unemployment Rate": "UNRATE",
}

# Keep only relevant columns
macro_df = macro_df[["date"] + list(macro_columns_subset.keys())]

# **Step 2: Drop NaN Rows (Keep Only Real Data Releases)**
macro_clean = macro_df.dropna()

# **Step 3: Compute YoY Inflation for Each Measure (Except Trimmed Mean PCE)**
for col in macro_columns_subset.keys():
    if col == "Trimmed Mean PCE Inflation Rate" or col == "Unemployment Rate":
        macro_clean[f"{col} YoY"] = macro_clean[col]  # Already annualized
    else:
        macro_clean[f"{col} YoY"] = macro_clean[col].pct_change(12) * 100  # Compute YoY change

# **Step 4: Outer Join with FOMC Rate Moves on `date`**
final_df = pd.merge(fomc_df, macro_clean, on="date", how="outer")

# **Step 5: Sort by Date to Maintain Chronology**
final_df.sort_values("date", inplace=True)

# **Plot Fed Funds Rate, Rate Hikes, and All Inflation Measures**
fig, ax1 = plt.subplots(figsize=(12, 6))

# **Plot Fed Funds Target Rate (Line + Markers, Left Y-Axis)**
sns.lineplot(data=final_df, x="date", y="tgt_rate", label="Fed Funds Target Rate", 
             color="midnightblue", marker="o", markersize=5, ax=ax1)

# **Plot Rate Changes as Bar Chart (Left Y-Axis, centered at 0)**
ax1.bar(final_df["date"], final_df["rate_change"], 
        color=["red" if x > 0 else "green" for x in final_df["rate_change"]], 
        alpha=0.6, width=30, label="Rate Hikes (+) / Cuts (-)")

# **Create a secondary Y-axis for Inflation Measures**
ax2 = ax1.twinx()

# **Plot All YoY Inflation Measures (Right Y-Axis)**
colors = ["red", "blue", "purple", "orange", "green", "brown"]  # Color map for inflation lines
for idx, (label, col) in enumerate(macro_columns_subset.items()):
    sns.lineplot(data=final_df, x="date", y=f"{label} YoY", label=f"{label} YoY", 
                 color=colors[idx], marker=".", markersize=3, ax=ax2)

# **Align the Zero Lines of Both Axes**
ax1_min, ax1_max = ax1.get_ylim()  # Get current y-limits for left axis
ax2_min, ax2_max = ax2.get_ylim()  # Get current y-limits for right axis

# Find the range for each axis
ax1_range = max(abs(ax1_min), abs(ax1_max))  # Max absolute range for left axis
ax2_range = max(abs(ax2_min), abs(ax2_max))  # Max absolute range for right axis

lower_limit = max(abs(ax1_min), abs(ax2_min))
# Set symmetric limits so both axes have the same zero alignment
ax1.set_ylim(-lower_limit, ax1_range)
ax2.set_ylim(-lower_limit, ax2_range)

# **Formatting the plot**
ax1.axhline(0, linestyle="--", color="black", alpha=0.7)
ax1.set_xlabel("Date")
ax1.set_ylabel("Fed Funds Rate & Rate Changes (bps)")
ax2.set_ylabel("Year-over-Year Economic Indicators (%)")

ax1.set_title("Fed Funds Rate vs. Rate Hikes vs. All YoY Inflation Measures")

# **Adjust legends**
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

ax1.grid(True)

# **Show the plot**
plt.show()
