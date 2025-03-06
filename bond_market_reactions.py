import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fredapi import Fred
import datetime

# Replace with your FRED API Key
FRED_API_KEY = "efe3da4f00a3fac72acd1e0dbe68901d"

# Initialize FRED API
fred = Fred(api_key=FRED_API_KEY)

# Define bond market series (DO NOT fetch Fed rate from FRED)
macro_series = {
    "3-Month Treasury Bill Rate": "TB3MS",  # Added TB3MS (3-Month T-Bill)
    "2-Year Treasury Yield": "DGS2",
    "10-Year Treasury Yield": "DGS10",
}

# Fetch bond market data from FRED
start_date = "2011-01-01"
end_date = datetime.datetime.now().strftime("%Y-%m-%d")

macro_data = {}
for name, series_id in macro_series.items():
    macro_data[name] = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)

# Convert to DataFrame
df = pd.DataFrame(macro_data)

# Ensure 'date' is a column, not an index
df = df.reset_index()
df.rename(columns={"index": "date"}, inplace=True)

# Convert 'date' column to datetime format
df["date"] = pd.to_datetime(df["date"])

# Calculate the 2-10 Spread
df["2-10 Spread"] = df["10-Year Treasury Yield"] - df["2-Year Treasury Yield"]

# Drop NaN values
df.dropna(inplace=True)

# **Load Fed Funds Rate & Rate Changes from CSV (instead of FRED)**
fomc_df = pd.read_csv(r"data/processed/fomc_meeting_rate_moves.csv")

# Ensure 'date' is in the right format
fomc_df["date"] = pd.to_datetime(fomc_df["date"])

# Debugging check to confirm 'date' exists in both DataFrames
print("FOMC DF Columns:", fomc_df.columns)
print("Macro DF Columns:", df.columns)

# **Merge FOMC data with bond market data**
df = pd.merge(fomc_df, df, on="date", how="outer")

# Sort by date to maintain chronology
df.sort_values("date", inplace=True)

# Set seaborn style
sns.set_style("whitegrid")

# Create the plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# **Plot Fed Funds Target Rate (Line + Markers, Left Y-Axis)**
sns.lineplot(
    data=df, 
    x="date", y="tgt_rate", 
    label="Fed Funds Target Rate", 
    color="midnightblue", 
    marker="o", markersize=5, 
    ax=ax1, linestyle=":"
)

# **Plot Rate Changes as a Bar Chart (Fatter Bars)**
bar_width = 40  # Increased bar width
ax1.bar(
    df["date"], 
    df["rate_change"], 
    color=["red" if x > 0 else "green" for x in df["rate_change"]], 
    alpha=0.6, width=bar_width, 
    label="Rate Hikes (+) / Cuts (-)"
)

# **Create a secondary Y-axis for Bond Market Data**
ax2 = ax1.twinx()

# **Plot 3-Month Treasury Bill Rate (Right Y-Axis)**
sns.lineplot(
    data=df, 
    x="date", y="3-Month Treasury Bill Rate", 
    label="3-Month Treasury Bill Rate", 
    color="purple", 
    markersize=5, 
    ax=ax2
)

# **Plot 2-Year Treasury Yield (Right Y-Axis)**
sns.lineplot(
    data=df, 
    x="date", y="2-Year Treasury Yield", 
    label="2-Year Treasury Yield", 
    color="red", 
    markersize=5, 
    ax=ax2
)

# **Plot 10-Year Treasury Yield (Right Y-Axis)**
sns.lineplot(
    data=df, 
    x="date", y="10-Year Treasury Yield", 
    label="10-Year Treasury Yield", 
    color="blue", 
    markersize=5, 
    ax=ax2
)

# **Plot 2-10 Spread (Right Y-Axis)**
sns.lineplot(
    data=df, 
    x="date", y="2-10 Spread", 
    label="2-10 Spread", 
    color="black", 
    linestyle="--", 
    ax=ax2
)

# **Force both y-axes to align at zero** 
ax1_min, ax1_max = ax1.get_ylim()
ax2_min, ax2_max = ax2.get_ylim()
max_range = max(abs(ax1_max), abs(ax2_max))
min_range = max(abs(ax1_min), abs(ax2_min))
ax1.set_ylim(-min_range, max_range*1.2)
ax2.set_ylim(-min_range, max_range*1.2)

# **Formatting the plot**
ax1.axhline(0, linestyle="--", color="black", alpha=0.7)
ax1.set_xlabel("Date")
ax1.set_ylabel("Fed Funds Rate & Rate Changes (bps)")
ax2.set_ylabel("Bond Market Rates & Spread (%)")

# **Title Formatting**
ax1.set_title(
    "Fed Funds Rate vs. Rate Hikes vs. 3M, 2Y, 10Y, and 2-10 Spread"
)

# **Adjust legends**
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

ax1.grid(True)

# **Show the plot**     
plt.show()
