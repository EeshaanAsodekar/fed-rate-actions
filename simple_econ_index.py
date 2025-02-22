import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load macroeconomic data
macro_df = pd.read_csv(r"data\processed\interpolated_macro_market_data.csv")  # Update with actual file path
macro_df["date"] = pd.to_datetime(macro_df.iloc[:, 0])

# Load FOMC rate moves
fomc_df = pd.read_csv(r"data\processed\fomc_meeting_rate_moves.csv")
fomc_df["date"] = pd.to_datetime(fomc_df["date"])

# Compute Inflation Gap (Core PCE YoY - Fed’s 2% target)
macro_df["inflation_gap"] = (macro_df["Core PCE Inflation (Ex Food & Energy)"].pct_change(12) * 100) - 2

# Compute Unemployment Gap (Difference from assumed Fed's neutral unemployment rate of 4.5%)
macro_df["unemployment_gap"] = macro_df["Unemployment Rate"] - 4

# Normalize the gaps using Z-score
macro_df["inflation_gap_z"] = (macro_df["inflation_gap"] - macro_df["inflation_gap"].mean()) / macro_df["inflation_gap"].std()
macro_df["unemployment_gap_z"] = (macro_df["unemployment_gap"] - macro_df["unemployment_gap"].mean()) / macro_df["unemployment_gap"].std()

# Construct Mandate Index (Equal Weighted)
macro_df["mandate_index"] = 0.5 * macro_df["inflation_gap_z"] + 0.5 * macro_df["unemployment_gap_z"]

# Merge with FOMC rate data
final_df = pd.merge(fomc_df, macro_df, on="date", how="inner")

# Plot Mandate Index, Fed Funds Rate, and Rate Change on the Same Plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot Fed Funds Target Rate (Line, Left Y-Axis)
sns.lineplot(data=final_df, x="date", y="tgt_rate", label="Fed Funds Target Rate", color="midnightblue", ax=ax1)

# Plot Rate Changes as Bar Chart (Left Y-Axis, centered at 0)
ax1.bar(final_df["date"], final_df["rate_change"], color=["red" if x > 0 else "green" for x in final_df["rate_change"]], 
        alpha=0.6, width=30, label="Rate Hikes (+) / Cuts (-)")

# Create a secondary Y-axis for Mandate Index
ax2 = ax1.twinx()

# Plot Mandate Index (Right Y-Axis)
sns.lineplot(data=final_df, x="date", y="mandate_index", label="Fed Mandate- Economic Indicators Composite", color="blue", ax=ax2)

# Formatting the plot
ax1.axhline(0, linestyle="--", color="black", alpha=0.7)
ax1.set_xlabel("Date")
ax1.set_ylabel("Fed Funds Rate & Rate Changes (bps)")
ax2.set_ylabel("Mandate Index (Z-score)")

ax1.set_title("Fed Funds Rate vs. Rate Hikes vs. Mandate Index")

# Adjust legends
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

ax1.grid(True)

# Show the plot
plt.show()
