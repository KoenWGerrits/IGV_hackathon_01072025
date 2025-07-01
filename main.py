# -*- coding: utf-8 -*-
"""
Do not spread this Python script without the author's approval
Created on Tue Jun  3 14:31:37 2025
@author: Koen Gerrits
for more information contact: k.gerrits@vnog.nl or +31 6 14210001

"""

import knmi
from datetime import datetime, timedelta
from fastapi.responses import JSONResponse
import joblib
import numpy as np
import pandas as pd
import logging
import requests
import sqlite3
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
import os

path = "" # << Vul hier het path in waar het script en het joblib bestand komen te staan
logging.basicConfig(filename=path+'hackathon_1_juni.log',
                    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

security = HTTPBasic()

# Define your username and password here
VALID_USERNAME = "IGV_hackathon_01072025"
VALID_PASSWORD = "KGbqfxmh3mpzhxlCehD7pjqWnJyWJyRG2mzUx7Is"

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    is_correct_username = secrets.compare_digest(credentials.username, VALID_USERNAME)
    is_correct_password = secrets.compare_digest(credentials.password, VALID_PASSWORD)
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


stations = {
    210: (52.171, 4.430), 215: (52.141, 4.437), 235: (52.928, 4.781),
    240: (52.318, 4.790), 249: (52.644, 4.979), 251: (53.392, 5.346),
    260: (52.100, 5.180), 265: (52.130, 5.274), 267: (52.898, 5.384),
    270: (53.224, 5.752), 269: (52.458, 5.520), 273: (52.703, 5.888),
    275: (52.056, 5.873), 277: (53.413, 6.200), 278: (52.435, 6.259),
    279: (52.750, 6.574), 280: (53.125, 6.585), 283: (52.069, 6.657),
    286: (53.196, 7.150), 290: (52.274, 6.891), 310: (51.442, 3.596),
    311: (51.379, 3.672), 319: (51.226, 3.861), 323: (51.527, 3.884),
    330: (51.992, 4.122), 340: (51.449, 4.342), 344: (51.962, 4.447),
    348: (51.970, 4.926), 350: (51.566, 4.936), 356: (51.859, 5.146),
    370: (51.451, 5.377), 375: (51.659, 5.707), 377: (51.198, 5.763),
    380: (50.906, 5.762), 391: (51.498, 6.197)}


def retrieve_knmi_data(knmi_station, variables, start, end):
    """Retrieving data from KNMI API"""
    dataframe = knmi.get_day_data_dataframe(stations=[knmi_station], start=start,
                                            end=end, variables=variables)
    # Reset index
    dataframe.reset_index(inplace=True)
    return dataframe

def save_to_sqlite(final_df: pd.DataFrame, station_number: int, db_path: str, db_name: str = "weather_data.db"):
    """
    Inserts generated data into database
    """
    # Add a simple identifier for date + station
    current_date = datetime.now().strftime("%d-%m-%Y")
    final_df["date_station"] = f"{current_date}_{station_number}"


    # Open connection
    conn = sqlite3.connect(os.path.join(db_path, db_name))

    try:
        # Automatically creates the table if it doesn't exist
        # Appends if it does
        final_df.to_sql("weather_data", conn, if_exists="append", index=False)
    finally:
        # Explicit close
        conn.close()

def fill_missing_days(data, columns):
    """Identifying and filling missing rows. Usually the current/previous day is missing
    from KNMI and meteoserver API. """
    # Identify missing rows
    all_dates = pd.date_range(start=data['Datum'].min(), end=data['Datum'].max())
    missing_dates = all_dates[~all_dates.isin(data['Datum'])]
    # Create new DataFrame for missing dates and copy values from the surrounding days
    new_rows = []
    for date in missing_dates:
        # Find the nearest available dates
        previous_date = data['Datum'][data['Datum'] < date].max()
        next_date = data['Datum'][data['Datum'] > date].min()

        # Get the mean of the surrounding dates
        previous_value = data.loc[data['Datum'] == previous_date, columns].mean()
        next_value = data.loc[data['Datum'] == next_date, columns].mean()
        mean_values = (previous_value + next_value) / 2

        new_row = {'Datum': date}
        for col in columns:
            new_row[col] = mean_values[col]
        new_rows.append(new_row)

    new_data = pd.concat([data, pd.DataFrame(new_rows)], ignore_index=True)
    # Sort DataFrame by date
    new_data.sort_values(by='Datum', inplace=True)
    new_data.reset_index(inplace=True, drop=True)
    return new_data

def data_type_transformer(data, date_variable, numeric_variables=list):
    """Function to transform data types"""
    for item in numeric_variables:
        data[item] = pd.to_numeric(data[item], errors= "coerce")
    # Create proper datetime format
    data[date_variable] = data[date_variable].dt.strftime('%d-%m-%Y')
    data[date_variable] = pd.to_datetime(data[date_variable], format='%d-%m-%Y')
    return data

def calculate_hdw(df):
    """
    Calculates the Saturated Vapor Pressure (Es) and Vapor Pressure (E)
    from a DataFrame containing columns for Mean Temperature and Relative Humidity.

    Args:
    df (pd.DataFrame): DataFrame with 'Mean Temperature' (°C) and 'Relative Humidity' (%).

    Returns:
    pd.DataFrame: DataFrame with added columns for 'Saturated Vapor Pressure (Es)'
                  and 'Vapor Pressure (E)'.
    """
    # Function to calculate Es (Saturated Vapor Pressure)
    def calculate_Es(T):
        return 6.112 * np.exp((17.62 * T) / (T + 243.12))

    # Apply the function to the 'Mean Temperature' column to get Es
    df['Saturated Vapor Pressure (Es)'] = df['temp_gem'].apply(calculate_Es)

    # Calculate the Vapor Pressure (E) using the Relative Humidity
    df['Vapor Pressure (E)'] = (df['vocht_gem'] / 100) * df['Saturated Vapor Pressure (Es)']

    df["Vapor Pressure Deficit"] = df['Saturated Vapor Pressure (Es)'] - df["Vapor Pressure (E)"]
    df["HDWI"] = df["Vapor Pressure Deficit"] * df["wind_gem"]
    df["HDWI_gust"] = df["Vapor Pressure Deficit"] * df["wind_max"]
    return df

def check_if_data_exists(station_number: int, db_path: str, db_name: str = "weather_data.db") -> pd.DataFrame | None:
    """
    Searches database if the requested data is already present
    Returns None if data is not present
    """
    current_date = datetime.now().strftime("%d-%m-%Y")
    date_station = f"{current_date}_{station_number}"

    # Connect and check if data exists
    conn = sqlite3.connect(os.path.join(db_path, db_name))
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='weather_data';")
        if not cursor.fetchone():
            # Table doesn't exist yet
            return None

        # Try to fetch existing data
        query = "SELECT * FROM weather_data WHERE date_station = ?"
        result_df = pd.read_sql_query(query, conn, params=(date_station,))

        if not result_df.empty:
            return result_df  # Data exists
        else:
            return None       # No match
    finally:
        conn.close()

@app.get("/natuurbrand_model/{weerstation}", response_description="Generate and download data from KNMI")
def natuurbrand_model(weerstation: int, username: str = Depends(authenticate)):
    available_weerstations = [215, 235, 240, 249, 251, 260, 267, 270, 269,
                              273, 275, 277, 278, 279, 280, 283, 286, 290,
                              310, 319, 323, 330, 340, 344, 348, 350, 356,
                              370, 375, 377, 380, 391]

    if weerstation not in available_weerstations:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid station: {weerstation}\nAvailable stations are {available_weerstations}."
        )

    # Check if data is already requested
    requested_data = check_if_data_exists(weerstation, path)

    if requested_data is not None:
        # Drop date station column
        requested_data = requested_data.drop('date_station', axis=1)
        return JSONResponse(content=requested_data.to_dict(orient="records"))
    else:
        # Retrieve location requested weather station
        lat, lon = stations.get(weerstation)

        # Formulate requested objects
        end_date = datetime.today()
        start_date = end_date - timedelta(weeks=4)
        desired_variables = ["FXX", "FG", "TG", "TN", "TX", "UG", "UX", "UN", "RH"]

        # Retrieve historic knmi data
        df = retrieve_knmi_data(weerstation, desired_variables, start_date, end_date)

        # Rename columns
        df.rename(columns={
            "YYYYMMDD": "Datum",
            "FXX": "wind_max",
            "FG": "wind_gem",
            "RH": "neersl_som",
            "TG": "temp_gem",
            "TX": "temp_max",
            "TN": "temp_min",
            "UG": "vocht_gem",
            "UX": "vocht_max",
            "UN": "vocht_min"
        }, inplace=True)

        if df.isna().any(axis=1).sum() > 5:
            raise HTTPException(
                status_code=400,
                detail=f"Weather station {weerstation} has too many missing values, please select another weather station."
            )

        df = df.fillna(df.mean())
        df.reset_index(drop=True, inplace=True)

        # Change data into proper units rather than 0,1 C
        df.loc[df['neersl_som'] == -1, 'neersl_som'] = 0
        df["neersl_som"] = df["neersl_som"]/10
        df["temp_gem"] = df["temp_gem"]/10
        df["temp_min"] = df["temp_min"]/10
        df["temp_max"] = df["temp_max"]/10
        df["wind_gem"] = df["wind_gem"]/10
        df["wind_max"] = df["wind_max"]/10

        df = data_type_transformer(df, "Datum", [])

        try:
            # Formulate url
            GFS_URL = f"https://data.meteoserver.nl/api/uurverwachting_gfs.php?lat={lat}&long={lon}&key="
            API_KEY = "7dbd6236a5"
            GFS_API = GFS_URL + API_KEY
            data_gfs = pd.DataFrame()\
            # Get request
            response = requests.get(GFS_API, timeout=30)
            # Raise HTTPError for bad status codes (4xx, 5xx)
            response.raise_for_status()
            # load data
            data_gfs = response.json()["data"]
            data_gfs = pd.DataFrame(data_gfs)
            response.close()
            logging.info("Loaded GFS data")
        except requests.HTTPError as e:
            logging.error("Failed to load GFS data. Status code: %s", e.response.status_code)
        except requests.RequestException as e:
            logging.error("An error occurred during the request to load GFS data: %s", e)
        except Exception as e:
            logging.error("An unexpected error occurred while loading GFS data: %s", e)

        # Make copy of dataframe
        data_harmonie = data_gfs.copy()

        try:
            # Make sure that get request was not copied from previous request
            while len(data_harmonie) == len(data_gfs):
                # Formulate url
                HARMONIE_URL = f"https://data.meteoserver.nl/api/uurverwachting.php?lat={lat}&long={lon}&key="
                HARMONIE_API = HARMONIE_URL + API_KEY
                data_harmonie = data_gfs.copy()
                # Get request
                response = requests.get(HARMONIE_API, timeout=30)
                # Raise HTTPError for bad status codes (4xx, 5xx)
                response.raise_for_status()
                data_harmonie = response.json()["data"]
                data_harmonie = pd.DataFrame(data_harmonie)
                response.close()
                logging.info("Loaded Harmonie data")
        except requests.HTTPError as e:
            logging.error("Failed to load Harmonie data. Status code: %s", e.response.status_code)
        except requests.RequestException as e:
            logging.error("An error occurred during the request to load Harmonie data: %s", e)
        except Exception as e:
            logging.error("An unexpected error occurred while loading Harmonie data: %s", e)

        # Change to proper Datetime columns
        data_gfs["Datum"] = pd.to_datetime(data_gfs["tijd_nl"], format='%d-%m-%Y %H:%M')

        # Change to proper datetime column
        data_harmonie["Datum"] = pd.to_datetime(data_harmonie["tijd_nl"], format='%d-%m-%Y %H:%M')
        # Replace first rows of GFS dataset with HARMONIE model
        data_gfs.loc[:len(data_harmonie)-1, ['temp', 'rv', 'neersl']] =  data_harmonie[['temp', 'rv',
                                                                                        'neersl']].values
        # Select only desired variables
        data_gfs = data_gfs[["Datum", "rv", "temp", "neersl", "winds"]]
        # Transform datasets to numeric / categorical
        data_gfs = data_type_transformer(data_gfs, "Datum", ["rv", "temp", "neersl", "winds"])


        # calculating mean, min, max values per day
        mean_temp_rv_winds = data_gfs.groupby('Datum')[['rv', "temp", "winds"]].mean().reset_index()
        max_temp_rv_winds = data_gfs.groupby('Datum')[['rv', "temp", "winds"]].max().reset_index()
        min_temp_rv = data_gfs.groupby('Datum')[['rv', "temp"]].min().reset_index()
        neerslag = data_gfs.groupby('Datum')["neersl"].sum().reset_index()

        # Drop date columsn from each selection
        max_temp_rv_winds.drop(columns=["Datum"], inplace=True)
        min_temp_rv.drop(columns=["Datum"], inplace=True)
        neerslag.drop(columns=["Datum"], inplace=True)

        # Rename columns
        mean_temp_rv_winds.rename(columns={"temp":"temp_gem", "rv":"vocht_gem", "winds": "wind_gem"}, inplace=True)
        max_temp_rv_winds.rename(columns={"temp":"temp_max", "rv":"vocht_max", "winds":"wind_max"}, inplace=True)
        min_temp_rv.rename(columns={"temp":"temp_min", "rv":"vocht_min"}, inplace=True)
        neerslag.rename(columns={"neersl":"neersl_som"}, inplace=True)

        # Concat all selections into a single dataframe
        data_per_day = pd.concat([mean_temp_rv_winds, neerslag, max_temp_rv_winds, min_temp_rv], axis=1)

        # Combine historic data to expected data
        final_df = pd.concat([df, data_per_day])
        final_df.reset_index(inplace=True, drop=True)

        # KNMI has a delay with historic data << fill data based on surrounding data
        fill_columns = ['wind_max', 'wind_gem', 'temp_gem', 'temp_min', 'temp_max',
                        'neersl_som', 'vocht_gem', 'vocht_max', 'vocht_min']
        final_df = fill_missing_days(final_df, fill_columns)

        # Calculate Hot-dry-windy index
        final_df = calculate_hdw(final_df)

        final_df.reset_index(inplace=True, drop=True)

        # Initiate new calculated columns
        final_df["neersl_som_3"] = ""
        final_df["neersl_som_7"] = ""
        final_df["temp_gem_som_3"] = ""
        final_df["temp_gem_som_7"] = ""
        final_df["vocht_gem_som_3"] = ""
        final_df["vocht_gem_som_7"] = ""

        # Input features for model
        features = ['wind_max', 'temp_gem', 'temp_min', 'temp_max', 'neersl_som',
                    'vocht_gem', 'vocht_max', 'vocht_min', 'neersl_som_3',
                    'neersl_som_7', 'temp_gem_som_3', 'temp_gem_som_7', 'vocht_gem_som_3',
                    'vocht_gem_som_7', 'HDWI', 'HDWI_gust']

        # Set all data columns to numeric
        for column in features:
            final_df[column] = pd.to_numeric(final_df[column])

        # Sum all data from neersl, temp & vocht
        for row, item in final_df.iterrows():
            final_df.at[row, "neersl_som_3"] = sum(final_df.loc[max(0, row-3):row, "neersl_som"])
            final_df.at[row, "neersl_som_7"] = sum(final_df.loc[max(0, row-7):row, "neersl_som"])
            final_df.at[row, "temp_gem_som_3"] = sum(final_df.loc[max(0, row-3):row, "temp_gem"])
            final_df.at[row, "temp_gem_som_7"] = sum(final_df.loc[max(0, row-7):row, "temp_gem"])
            final_df.at[row, "vocht_gem_som_3"] = sum(final_df.loc[max(0, row-3):row, "vocht_gem"])
            final_df.at[row, "vocht_gem_som_7"] = sum(final_df.loc[max(0, row-7):row, "vocht_gem"])

        # Load logistic model and scaler
        loaded_data = joblib.load(path+"log_model_and_scaler.joblib")
        model = loaded_data['model']
        scaler = loaded_data['scaler']

        # Scale data
        scaled_data = scaler.transform(final_df[features])
        # Use model to calculated predicted probability
        final_df["predicted_probability"] = model.predict_proba(scaled_data)[:,1]
        final_df["STN"] = weerstation

        final_df["Datum"] = final_df["Datum"].dt.strftime("%d-%m-%Y")
        # Save data in database file
        save_to_sqlite(final_df, weerstation, path)
        # Convert DataFrame to JSON and return
        final_df = final_df.drop('date_station', axis=1)
        return JSONResponse(content=final_df.to_dict(orient="records"))
