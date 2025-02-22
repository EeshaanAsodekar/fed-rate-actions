from fredapi import Fred
import pandas as pd
import datetime
import yfinance as yf

def get_raw_macro_market_data():
    # Replace with your FRED API Key
    FRED_API_KEY = "efe3da4f00a3fac72acd1e0dbe68901d"

    # Initialize the FRED API
    fred = Fred(api_key=FRED_API_KEY)

    # Define the key macro variables and their FRED series IDs
    macro_variables = {
        "Unemployment Rate": "UNRATE",
        "CPI (All Urban Consumers)": "CPIAUCSL",
        "Core CPI (Ex Food & Energy)": "CPILFESL",
        "PCE Inflation": "PCEPI",
        "Core PCE Inflation (Ex Food & Energy)": "PCEPILFE",
        "Trimmed Mean PCE": "PCETRIM12M159SFRBDAL",
        "Median CPI": "MEDCPIM158SFRBCLE",
        "Real GDP": "A191RL1Q225SBEA",
        "GDP Growth Rate": "GDPC1",
        "10-Year Treasury Yield": "DGS10",
        "2-Year Treasury Yield": "DGS2",
        "Industrial Production Index": "INDPRO",
        "Total Nonfarm Payrolls": "PAYEMS",
        "Housing Starts": "HOUST",
        "Building Permits": "PERMIT",
        "Consumer Sentiment Index": "UMCSENT",
        "ISM Manufacturing PMI": "MANEMP",
        "ISM Non-Manufacturing PMI": "NMFSL",
        "Trade Balance": "BOPGTB",
        "M2 Money Supply": "M2SL",
        "30-Year Treasury Yield": "DGS30",
        "Initial Jobless Claims": "ICSA",
        "West Texas Intermediate":"DCOILWTICO",
        "Real Broad Dollar Index":"RTWEXBGS",
        "NASDAQ Composite Index":"NASDAQCOM",
    }

    # Specify the time range
    start_date = "2011-01-01"
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")

    # Fetch macroeconomic data from FRED
    macro_data = {}
    for name, series_id in macro_variables.items():
        try:
            print(f"Downloading macro data for {name}...")
            macro_data[name] = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
        except ValueError as e:
            print(f"Error fetching {name} (Series ID: {series_id}): {e}")
    
    # Combine the macro data into a single DataFrame
    macro_df = pd.DataFrame(macro_data)

    # After fetching the data for all variables
    macro_df["2-10 Spread"] = macro_df["10-Year Treasury Yield"] - macro_df["2-Year Treasury Yield"]

    macro_df.to_csv("data/raw/raw_macro_market_data.csv")





def interpolate_macro_market_data(file_path="data/raw/raw_macro_market_data.csv"):
    """
    Reads the raw macro market data from a CSV file and applies linear interpolation
    to fill missing values in all columns.
    
    :param file_path: Path to the CSV file containing raw macro market data
    :return: DataFrame with interpolated values
    """
    # Load the dataset
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    # Perform linear interpolation on all columns
    df_interpolated = df.interpolate(method="linear")

    # Save the interpolated data
    df_interpolated.to_csv("data/processed/interpolated_macro_market_data.csv")

    return df_interpolated





if __name__ =="__main__":
    get_raw_macro_market_data()
    interpolate_macro_market_data()