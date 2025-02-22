### get the fed funds rate and rate moves
import requests
import pandas as pd





### get the Federal Funds Effective Rate
def get_fed_effective_rate():
    # FRED API key
    FRED_API_KEY = "efe3da4f00a3fac72acd1e0dbe68901d"

    # Endpoint for Federal Funds Effective Rate (FEDFUNDS)
    url = "https://api.stlouisfed.org/fred/series/observations"

    # Request data from FRED API
    params = {
        "series_id": "FEDFUNDS",
        "observation_start": "2012-01-01",
        "api_key": FRED_API_KEY,
        "file_type": "json"
    }
    response = requests.get(url, params=params)
    data = response.json()

    # Extract relevant data
    observations = data.get('observations', [])
    fed_funds_rate = [{"date": obs["date"], "rate": float(obs["value"])} for obs in observations if obs["value"] != "."]

    # Convert to DataFrame
    fed_funds_df = pd.DataFrame(fed_funds_rate)

    # Save to CSV or process further
    fed_funds_df.to_csv("data/raw/fed_funds_rate.csv", index=False)

    print(fed_funds_df.head())





### get the Federal Funds Target Rate (Upper Bound)
def get_fed_utarget_rate():
    # FRED API key
    FRED_API_KEY = "efe3da4f00a3fac72acd1e0dbe68901d"

    # Endpoint for Federal Funds Target Rate (upper bound)
    url = "https://api.stlouisfed.org/fred/series/observations"

    # Request data from FRED API
    params = {
        "series_id": "DFEDTARU",  # Federal Funds Target Rate (Upper Bound)
        "observation_start": "2012-01-01",
        "api_key": FRED_API_KEY,
        "file_type": "json"
    }
    response = requests.get(url, params=params)
    data = response.json()

    # Extract relevant data
    observations = data.get('observations', [])
    fed_funds_target_rate = [{"date": obs["date"], "rate": float(obs["value"])} for obs in observations if obs["value"] != "."]

    # Convert to DataFrame
    fed_funds_df = pd.DataFrame(fed_funds_target_rate)

    # Save to CSV or process further
    fed_funds_df.to_csv("data/raw/fed_funds_target_rate.csv", index=False)

    print(fed_funds_df.head())





### Function to get Fed rate moves by meeting
def get_rate_moves_by_meeting():
    # Load the FOMC meeting dates
    fomc_meeting_dates_path = "data/processed/fomc_meeting_dates.csv"
    fomc_dates_df = pd.read_csv(fomc_meeting_dates_path)
    fomc_dates_df["date"] = pd.to_datetime(fomc_dates_df["date"])
    print("FOMC Meeting Dates:")
    print(fomc_dates_df.tail())
    
    # Load the Fed funds target rate data
    fed_funds_rate_path = "data/raw/fed_funds_target_rate.csv"
    fed_funds_df = pd.read_csv(fed_funds_rate_path)
    fed_funds_df["date"] = pd.to_datetime(fed_funds_df["date"])
    print("Fed Funds Target Rate:")
    print(fed_funds_df.tail())

    # Initialize a list to store the results
    rate_changes = []

    # Iterate over each FOMC meeting date
    for i in range(len(fomc_dates_df)):
        meeting_date = fomc_dates_df.loc[i, "date"]
        print(f"Processing meeting date: {meeting_date}")

        # Get the rate on the meeting date
        rate_on_meeting_date = fed_funds_df.loc[fed_funds_df["date"] == meeting_date, "rate"]
        print(f"Rate on meeting date: {rate_on_meeting_date.values}")

        # Initialize tgt_rate as None
        tgt_rate = None

        # Check if there is a rate change on the meeting day itself
        if not rate_on_meeting_date.empty:
            rate_before_meeting = fed_funds_df.loc[fed_funds_df["date"] < meeting_date, "rate"]
            if not rate_before_meeting.empty:
                previous_rate = rate_before_meeting.iloc[-1]
                if rate_on_meeting_date.iloc[0] != previous_rate:
                    rate_change = rate_on_meeting_date.iloc[0] - previous_rate
                    tgt_rate = rate_on_meeting_date.iloc[0]
                    rate_changes.append({"date": meeting_date, "rate_change": rate_change, "tgt_rate": tgt_rate})
                    print(f"Rate changed on meeting day: {rate_change}, Target rate: {tgt_rate}")
                    continue  # Skip to the next meeting

        # Get the rate for the day after the meeting
        next_day_rates = fed_funds_df.loc[fed_funds_df["date"] > meeting_date, "rate"]
        rate_after_meeting = next_day_rates.iloc[0] if not next_day_rates.empty else None
        print(f"Rate after meeting: {rate_after_meeting}")

        # Check and calculate the rate change for the next day
        if not rate_on_meeting_date.empty and rate_after_meeting is not None:
            rate_change = rate_after_meeting - rate_on_meeting_date.iloc[0]
            tgt_rate = rate_after_meeting
            rate_changes.append({"date": meeting_date, "rate_change": rate_change, "tgt_rate": tgt_rate})
            print(f"Rate changed the next day: {rate_change}, Target rate: {tgt_rate}")
        else:
            print(f"Issue with meeting date {meeting_date}: Missing data.")

    # Create a DataFrame from the results
    rate_changes_df = pd.DataFrame(rate_changes)

    # Display the DataFrame
    print("Rate Changes:")
    print(rate_changes_df.head())

    rate_changes_df.to_csv("data/processed/fomc_meeting_rate_moves.csv")

    return rate_changes_df


if __name__ == "__main__":
    get_fed_effective_rate()
    get_fed_utarget_rate()
    get_rate_moves_by_meeting()