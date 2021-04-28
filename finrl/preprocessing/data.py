from __future__ import division, absolute_import, print_function

from datetime import datetime

import pandas as pd


def load_dataset(*, file_name: str) -> pd.DataFrame:
    """
    load csv dataset from path
    :return: (df) pandas dataframe
    """
    # _data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
    _data = pd.read_csv(file_name)
    return _data


def calculate_split(df, start=None, end=None):
    start, end = start or df.date.min(), end or df.date.max()
    start = convert_to_datetime(start)
    end = convert_to_datetime(end)
    trade = start + (end - start) * 0.66
    assert start < trade < end
    return (
        start.strftime("%Y-%m-%d"),
        trade.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d")
    )


def data_split(df, start, end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df.date >= start) & (df.date < end)]
    data = data.sort_values(["date", "tic"], ignore_index=True)
    data.index = data.date.factorize()[0]
    return data


def convert_to_datetime(time):
    if isinstance(time, str):
        try:
            return datetime.strptime(time, "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return datetime.strptime(time, "%Y-%m-%d")
