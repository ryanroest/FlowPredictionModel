import pandas as pd
import numpy as np
import utility
import datetime

def clean_mes_data(df, convert_timestamp=True, sort_timestamp=True, remove_duplicates=True, select_quality=True):
    '''
    This function convert the timestamp column to timestamp, sort on the timestamp column,
    removes duplicates and saves only the data with quality equal to 1
    '''
    if convert_timestamp:
        if df["TimeStamp"].dtype != "<M8[ns]":
            df["TimeStamp"] = pd.to_datetime(df["TimeStamp"])

    if sort_timestamp:
        df.sort_values("TimeStamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

    if remove_duplicates:
        df = df.loc[~df["TimeStamp"].duplicated()].reset_index(drop=True)

    if select_quality:
        df = df.loc[df["DataQuality"] == 1].reset_index(drop=True)

    return df

def merge_flow_level(flow_data, level_data):
    # INTERPOLATION OF MISSING MEASUREMENTS
    # Get all timestamps
    unique_timestamps = pd.concat([flow_data["TimeStamp"], level_data["TimeStamp"]]).unique()

    # Add timestamps to index
    flow_data = flow_data.set_index("TimeStamp").reindex(unique_timestamps)\
                         .reset_index(drop=False).sort_values("TimeStamp").reset_index(drop=True)
    level_data = level_data.set_index("TimeStamp").reindex(unique_timestamps)\
                           .reset_index(drop=False).sort_values("TimeStamp").reset_index(drop=True)

    return flow_data, level_data

def fill_flow(flow_data):
    '''
    Fill in missing flow data
    '''
    flow_data.loc[flow_data["Value"].isna() &\
                  (~flow_data["Value"].isna()).shift(1) &\
                  (~flow_data["Value"].isna()).shift(-1), "Value"] = flow_data["Value"].shift()\
                                                                            [flow_data["Value"].isna() &\
                                                                             (~flow_data["Value"].isna()).shift(1) &\
                                                                             (~flow_data["Value"].isna()).shift(-1)]
    flow_data["Value"] = flow_data["Value"].fillna(0)

    return flow_data

def fill_level(level_data):
    '''
    Fill missing level data
    '''
    na_indices = level_data.index[level_data["Value"].isna()]
    non_na_indices = level_data.index[~level_data["Value"].isna()]

    prior_indices = utility.search_prior_indices(na_indices, non_na_indices).reset_index(drop=True)
    posterior_indices = utility.search_posterior_indices(na_indices, non_na_indices).reset_index(drop=True)

    # TimeStamps of prior and posterior indices
    ts_prior = level_data[level_data.index.isin(prior_indices)]['TimeStamp']\
                         .apply(lambda i: (i - datetime.datetime(2017,1,1)).total_seconds())\
                         .reset_index(drop=True)
    ts_posterior = level_data.loc[posterior_indices, "TimeStamp"]\
                             .apply(lambda i: (i - datetime.datetime(2017,1,1)).total_seconds())\
                             .reset_index(drop=True)
    ts_actual = level_data.loc[level_data["Value"].isna(), "TimeStamp"]\
                          .apply(lambda i: (i - datetime.datetime(2017,1,1)).total_seconds())\
                          .reset_index(drop=True)

    # Levels of prior and posterior indices
    level_prior = level_data[level_data.index.isin(prior_indices)]['Value'].reset_index(drop=True)
    level_posterior = level_data.loc[posterior_indices, "Value"].reset_index(drop=True)

    # Calculating weighted level values
    fill_values = (level_prior*(ts_posterior-ts_actual) + level_posterior*(ts_actual-ts_prior)) /\
                  (ts_posterior - ts_prior)
    fill_values.index = level_data.index[level_data["Value"].isna()]
    level_data["Value"] = level_data["Value"].fillna(fill_values)

    return level_data

def flow_by_hour(df, impute_range=False):
    '''
    Calculates the total amount of flow per hour
    '''
    flow_data = df.copy()
    flow_data["TimeStamp"] = pd.to_datetime(flow_data["TimeStamp"])
    flow_data["TimeSpan"] = flow_data["TimeStamp"].diff(1).apply(lambda i: i.seconds).fillna(5)
    flow_data["TimeHour"] = flow_data["TimeStamp"].apply(lambda i: i.replace(minute=0, second=0))
    flow_data["Flow"] = flow_data["Value"] / 3600 * flow_data["TimeSpan"]

    flow_data = flow_data.groupby("TimeHour").aggregate({"Flow": np.sum, "DataQuality": np.mean, "TimeSpan": np.sum})
    
    if impute_range:
        dt_range = pd.date_range(flow_data.index[0].floor('h'), flow_data.index[-1].floor('h'), freq='h')
        flow_data = flow_data.reindex(dt_range)
        flow_data["Flow"] = flow_data["Flow"].fillna(0)
        flow_data["DataQuality"] = flow_data["DataQuality"].fillna(0)
    
    return flow_data.reset_index(drop=False).rename(columns={"index": "TimeHour"})