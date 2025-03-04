"""
Script to get insightful plot of:
        1. unemployment gap (Core PCE inflation Ex. Food and Energy based)
        2. inflation gap (Unemployment Rate based)
        3. fomc target rate
        4. fomc meeting rate hikes/cuts

"""

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

# **Step 1: Extract Only Required Macro Columns**
macro_df = macro_df[["date", "Unemployment Rate", "Core PCE Inflation (Ex Food & Energy)"]]

# **Step 2: Drop NaN Rows (Keep Only Real Data Releases)**
macro_clean = macro_df.dropna()

# **Step 3: Compute Inflation Gap (YoY % Change in Core PCE - Fed’s 2% target)**
macro_clean["inflation_gap"] = (macro_clean["Core PCE Inflation (Ex Food & Energy)"].pct_change(12) * 100) - 2

# **Step 4: Compute Unemployment Gap (Deviation from 4% neutral level)**
macro_clean["unemployment_gap"] = macro_clean["Unemployment Rate"] - 4

# **Step 5: Outer Join with FOMC Rate Moves on `date`**
final_df = pd.merge(fomc_df, macro_clean, on="date", how="outer")

# **Step 6: Sort by Date to Maintain Chronology**
final_df.sort_values("date", inplace=True)


# **Plot Fed Funds Rate, Rate Hikes, Inflation Gap, and Unemployment Gap**
fig, ax1 = plt.subplots(figsize=(12, 6))

# **Plot Fed Funds Target Rate (Line + Markers, Left Y-Axis)**
sns.lineplot(data=final_df, x="date", y="tgt_rate", label="Fed Funds Target Rate", color="midnightblue", marker="o", markersize=5, ax=ax1)

# **Plot Rate Changes as Bar Chart (Left Y-Axis, centered at 0)**
ax1.bar(final_df["date"], final_df["rate_change"], 
        color=["red" if x > 0 else "green" for x in final_df["rate_change"]], 
        alpha=0.6, width=30, label="Rate Hikes (+) / Cuts (-)")

# **Create a secondary Y-axis for Inflation & Unemployment Gaps**
ax2 = ax1.twinx()

# **Plot Inflation Gap (Right Y-Axis)**
sns.lineplot(data=final_df, x="date", y="inflation_gap", label="Inflation Gap", color="red", marker="o", markersize=5, ax=ax2)

# **Plot Unemployment Gap (Right Y-Axis)**
sns.lineplot(data=final_df, x="date", y="unemployment_gap", label="Unemployment Gap", color="blue", marker="o", markersize=5, ax=ax2)

# **Formatting the plot**
ax1.axhline(0, linestyle="--", color="black", alpha=0.7)
ax1.set_xlabel("Date")
ax1.set_ylabel("Fed Funds Rate & Rate Changes (bps)")
ax2.set_ylabel("Inflation & Unemployment Gaps")

ax1.set_title("Fed Funds Rate vs. Rate Hikes vs. Inflation & Unemployment Gaps")

# **Adjust legends**
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

ax1.grid(True)

# **Show the plot**     
plt.show()
