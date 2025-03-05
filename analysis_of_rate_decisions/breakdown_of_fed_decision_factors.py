"""
studying interaction only on the rate_change variable (Y) and the
inflation macros and unemployement YOY Rates/ Rates (X)
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def create_distance_from_target(macro_df,
                                inflation_col='CPI (All Urban Consumers)',
                                unemp_col='Unemployment Rate',
                                infl_target=2.0,
                                unemp_target=4.0):
    """
    Adds two new columns to macro_df:
      - 'inflation_gap' = inflation_col - infl_target
      - 'unemployment_gap' = unemp_col - unemp_target
    """
    macro_df['inflation_gap'] = macro_df[inflation_col] - infl_target
    macro_df['unemployment_gap'] = macro_df[unemp_col] - unemp_target
    
    return macro_df

def merge_fomc_with_macro(fomc_df, macro_df, date_col='date', macro_date_col='date',
                          lag_months=0):
    """
    Merges macro data onto FOMC meetings based on a specified lag in months.
    
    - For each FOMC meeting date, we look up the macro data 'lag_months' 
      before the meeting date.
    - If lag_months=0, we use the same month (or same day if daily data).
    - This function assumes both dataframes have date columns in datetime format.
      If not, we convert them inside the function.
    
    Returns a merged dataframe with one row per FOMC meeting and
    columns of macro data as features.
    """
    # Ensure date columns are datetime
    fomc_df[date_col] = pd.to_datetime(fomc_df[date_col])
    macro_df[macro_date_col] = pd.to_datetime(macro_df[macro_date_col])
    
    # Sort macro data by date (important for asof-merge)
    macro_df = macro_df.sort_values(by=macro_date_col).reset_index(drop=True)
    
    # If your macro data is monthly, you can do a month-based offset.
    # If it's daily, you can do a day-based offset. 
    # For demonstration, let's assume monthly frequency. 
    # We'll shift by approx 30 * lag_months days:
    
    shift_days = 30 * lag_months
    
    # Create a new column in FOMC for "lookup date"
    # i.e. the date we want to use in macro data
    fomc_df['lookup_date'] = fomc_df[date_col] - pd.to_timedelta(shift_days, unit='days')
    
    # We can do a merge_asof if macro data is daily or monthly
    # We want the row in macro_df at or before 'lookup_date' 
    # (the nearest date not after the lookup_date).
    merged_df = pd.merge_asof(
        fomc_df.sort_values('lookup_date'),
        macro_df.sort_values(macro_date_col),
        left_on='lookup_date',
        right_on=macro_date_col,
        direction='backward'  # get the macro data on or before the lookup_date
    )
    
    # Re-sort back by the actual FOMC meeting date, if desired
    merged_df = merged_df.sort_values(date_col).reset_index(drop=True)
    
    return merged_df

def run_linear_regression(merged_df, target_col='rate_change', 
                          drop_cols=None):
    """
    Runs a simple Linear Regression on the merged dataframe:
      - Y = target_col (rate_change)
      - X = all numeric columns except drop_cols and the target column.

    Returns the fitted model and a DataFrame of coefficients.
    """
    if drop_cols is None:
        drop_cols = ['Unnamed: 0', 'date', 'lookup_date', 'tgt_rate',
                     'rate_change']  # add or remove as needed
    
    # Select features
    features_df = merged_df.select_dtypes(include=[np.number]).drop(columns=drop_cols, errors='ignore')
    X = features_df.values
    y = merged_df[target_col].values
    
    # Fit linear regression
    linreg = LinearRegression()
    linreg.fit(X, y)
    
    # Get coefficients
    coeffs = pd.DataFrame({
        'feature': features_df.columns,
        'coefficient': linreg.coef_
    }).sort_values(by='coefficient', ascending=False)
    
    intercept = linreg.intercept_
    return linreg, coeffs, intercept

def run_random_forest(merged_df, target_col='rate_change', 
                      drop_cols=None, n_estimators=100, max_depth=5):
    """
    Runs a RandomForestRegressor on the merged dataframe:
      - Y = target_col
      - X = all numeric columns except drop_cols and the target column.

    Returns the fitted RandomForest model and a DataFrame of feature importances.
    """
    if drop_cols is None:
        drop_cols = ['Unnamed: 0', 'date', 'lookup_date', 'tgt_rate',
                     'rate_change']  # add or remove as needed
    
    features_df = merged_df.select_dtypes(include=[np.number]).drop(columns=drop_cols, errors='ignore')
    X = features_df.values
    y = merged_df[target_col].values
    
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(X, y)
    
    importances = pd.DataFrame({
        'feature': features_df.columns,
        'importance': rf.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    return rf, importances

# -----------------------------------------------
# Example Usage (Assuming you already have the dataframes):
# -----------------------------------------------

if __name__ == "__main__":
    # 1. Load Data
    fomc_df = pd.read_csv(r"data\processed\fomc_meeting_rate_moves.csv")
    macro_df = pd.read_csv(r"data\processed\interpolated_macro_market_data.csv")

    # 2. Create distance-from-target features
    macro_df = create_distance_from_target(macro_df,
                                           inflation_col='CPI (All Urban Consumers)',
                                           unemp_col='Unemployment Rate',
                                           infl_target=2.0,
                                           unemp_target=4.0)

    # 3. Merge FOMC + Macro with a certain lag (e.g., 1 month before the meeting)
    merged_df = merge_fomc_with_macro(fomc_df, macro_df,
                                      date_col='date', 
                                      macro_date_col='Unnamed: 0',  # or 'date' if your macro_df has a date column named 'date'
                                      lag_months=1)

    # 4. Run Linear Regression to get interpretable coefficients
    linreg_model, linreg_coeffs, linreg_intercept = run_linear_regression(merged_df,
                                                                          target_col='rate_change',
                                                                          drop_cols=['Unnamed: 0','date','lookup_date','tgt_rate','rate_change'])
    print("Linear Regression Coefficients:")
    print(linreg_coeffs)
    print(f"Intercept: {linreg_intercept}\n")

    # 5. Run a Random Forest to see non-linear feature importances
    rf_model, rf_importances = run_random_forest(merged_df,
                                                 target_col='rate_change',
                                                 drop_cols=['Unnamed: 0','date','lookup_date','tgt_rate','rate_change'],
                                                 n_estimators=100,
                                                 max_depth=5)
    print("Random Forest Feature Importances:")
    print(rf_importances)
