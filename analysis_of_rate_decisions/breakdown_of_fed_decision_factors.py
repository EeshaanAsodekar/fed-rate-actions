import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def yoy_on_daily_single_col(df, date_col, col):
    """
    For a single column of daily data, compute the YoY % change by looking up
    the value from ~1 year ago using a merge_asof approach (nearest date on or before).
    
    - df: DataFrame with columns [date_col, col] plus potentially others.
    - date_col: Name of the date column (assumed to be in datetime form).
    - col: The column for which we want to compute a YOY rate.
    
    Returns a DataFrame that:
      1) Has a new column  col+'_yoy'
      2) Does NOT drop the original column (we'll handle dropping outside).
    """
    # Ensure sorted by date
    df = df.sort_values(date_col).copy()

    # Create a helper DataFrame for the '1 year ago' lookup
    helper_df = df[[date_col, col]].copy()
    helper_df = helper_df.rename(
        columns={date_col: 'helper_date', col: 'helper_value'}
    )
    helper_df = helper_df.sort_values('helper_date')

    # For each row, define date_1y_ago = current_date - 1 year
    df['date_1y_ago'] = df[date_col] - pd.DateOffset(years=1)

    # Merge using asof to find the row in helper_df whose 'helper_date' is
    # closest to but not after 'date_1y_ago'
    merged = pd.merge_asof(
        df.sort_values('date_1y_ago'),
        helper_df,
        left_on='date_1y_ago',
        right_on='helper_date',
        direction='backward'
    ).sort_values(date_col)

    yoy_col = col + '_yoy'
    merged[yoy_col] = (
        (merged[col] - merged['helper_value']) / merged['helper_value']
    ) * 100

    # Merge the new yoy_col back into the original df structure
    df[yoy_col] = merged[yoy_col].values

    # Clean up
    df.drop(columns=['date_1y_ago'], inplace=True, errors='ignore')
    return df

def create_yoy_inflation_daily_precise(df, inflation_cols, date_col='date'):
    """
    For each inflation column in inflation_cols, compute a YOY % change
    by looking up the actual data from ~1 year prior. Drop the original
    absolute columns afterwards, keeping only the new _yoy columns.
    """
    # Sort the DF by date first
    df = df.sort_values(date_col).copy()

    for col in inflation_cols:
        # 1) Compute yoy col
        df = yoy_on_daily_single_col(df, date_col, col)
        # 2) Drop the original column after yoy is computed
        df.drop(columns=[col], inplace=True)

    return df

def merge_fomc_with_macro(fomc_df, macro_df, fomc_date_col='date', macro_date_col='date',
                          lag_months=0):
    """
    Merges macro data onto FOMC meetings with a specified lag in months,
    using a merge_asof approach (nearest date on or before the 'lookup_date').
    """
    fomc_df = fomc_df.copy()
    macro_df = macro_df.copy()

    # Ensure datetime
    fomc_df[fomc_date_col] = pd.to_datetime(fomc_df[fomc_date_col])
    macro_df[macro_date_col] = pd.to_datetime(macro_df[macro_date_col])

    # Sort
    fomc_df.sort_values(fomc_date_col, inplace=True)
    macro_df.sort_values(macro_date_col, inplace=True)

    # If we want data X months before the meeting, shift by ~30 * X days
    shift_days = 30 * lag_months
    fomc_df['lookup_date'] = fomc_df[fomc_date_col] - pd.to_timedelta(shift_days, unit='days')

    merged_df = pd.merge_asof(
        fomc_df, 
        macro_df,
        left_on='lookup_date',
        right_on=macro_date_col,
        direction='backward'
    )

    return merged_df

def run_linear_regression(merged_df, target_col='rate_change', drop_cols=None):
    """
    Performs OLS regression to compute coefficient significance.

    - merged_df: DataFrame containing the data
    - target_col: The dependent variable (Y)
    - drop_cols: Columns to exclude from the independent variables (X)

    Returns:
    - results: The fitted OLS model
    - summary_df: DataFrame containing coefficients, p-values, and confidence intervals
    - intercept: The model intercept
    """
    if drop_cols is None:
        drop_cols = ['date', 'lookup_date', 'tgt_rate', 'rate_change']

    # Selecting numeric columns only
    features_df = merged_df.select_dtypes(include=[np.number]).drop(columns=drop_cols, errors='ignore')

    # Define X (independent variables) and Y (dependent variable)
    X = features_df.copy()
    X = sm.add_constant(X)  # Adds an intercept term
    y = merged_df[target_col].values

    # Fit OLS Regression
    model = sm.OLS(y, X).fit()

    # Extract coefficients, p-values, and confidence intervals
    summary_df = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.params.values,
        'p_value': model.pvalues.values,
        'conf_lower': model.conf_int()[0].values,
        'conf_upper': model.conf_int()[1].values
    }).sort_values(by='p_value')

    intercept = model.params['const']
    
    return model, summary_df, intercept

def run_random_forest(merged_df, target_col='rate_change', drop_cols=None,
                      n_estimators=100, max_depth=5):
    """
    Random Forest regression for non-linear relationships.
    Returns the fitted model, and a DataFrame of feature importances.
    """
    if drop_cols is None:
        drop_cols = ['date','lookup_date','tgt_rate','rate_change']

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

if __name__ == "__main__":

    # 1. Load data
    fomc_df = pd.read_csv(r"data\processed\fomc_meeting_rate_moves.csv")
    macro_df = pd.read_csv(r"data\processed\interpolated_macro_market_data.csv")
    macro_df.rename(columns={"Unnamed: 0": "date"}, inplace=True)
    macro_df['date'] = pd.to_datetime(macro_df['date'])

    # 2. Focus only on inflation macros, 2–10 spread, unemployment rate,
    #    GDP growth rate, and 10-year yield.
    inflation_cols = [
        'CPI (All Urban Consumers)',
        'Core CPI (Ex Food & Energy)',
        'PCE Inflation',
        'Core PCE Inflation (Ex Food & Energy)',
        # Add more inflation columns here if desired (e.g., 'PCE Inflation')
    ]
    keep_cols = [
        'date',
        '2-10 Spread',
        'Unemployment Rate',
        'GDP Growth Rate',
        '2-Year Treasury Yield',
        '10-Year Treasury Yield',
        '5-Year Breakeven Inflation Rate',
    ] + inflation_cols

    macro_df = macro_df[keep_cols].copy()

    # 3. Convert the daily inflation data to YOY by looking up the exact day ~1 year prior
    macro_df = create_yoy_inflation_daily_precise(macro_df, inflation_cols, date_col='date')

    # 4. Merge with FOMC data. We use lag_months=0 for the same day; adjust if needed.
    merged_df = merge_fomc_with_macro(
        fomc_df,
        macro_df,
        fomc_date_col='date',
        macro_date_col='date',
        lag_months=0
    )

    merged_df.drop(columns=["Unnamed: 0"], inplace=True)
    print(merged_df.columns)
    print(merged_df.head())

    # 5. Run Linear Regression
    linreg_model, linreg_summary, linreg_intercept = run_linear_regression(
        merged_df,
        target_col='rate_change',
        drop_cols=['date','lookup_date','tgt_rate','rate_change']
    )

    print("Linear Regression Coefficients, P-values, and Confidence Intervals:")
    print(linreg_summary)
    print(f"Intercept: {linreg_intercept}\n")

    # 6. Run Random Forest
    rf_model, rf_importances = run_random_forest(
        merged_df,
        target_col='rate_change',
        drop_cols=['date','lookup_date','tgt_rate','rate_change'],
        n_estimators=100,
        max_depth=5
    )
    print("Random Forest Feature Importances:")
    print(rf_importances)
