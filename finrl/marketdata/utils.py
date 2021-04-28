from datetime import datetime
import glob
import os

from pandas import read_csv

from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader


def fetch_and_store(
    start_date=config.START_DATE,
    end_date=None,
    interval=None,
    ticker_list=config.CRYPTO_TICKER
):
    print("==============Start Fetching Data===========")

    df = YahooDownloader(
        start_date=start_date,
        end_date=end_date or datetime.utcnow().strftime("%Y-%m-%d"),
        ticker_list=ticker_list
    ).fetch_data()
    now = datetime.now().strftime(config.DATETIME_FMT)
    filename = f'./{config.DATA_SAVE_DIR}/{now}.csv'
    df.to_csv(filename)
    return df


def load():
    print("==============Loading Data===========")

    list_of_files = glob.glob(f'./{config.DATA_SAVE_DIR}/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    df = read_csv(latest_file)
    print(f'Shape of Dataframe: {df.shape}')
    print(df.head())
    print(df.tail())
    return df
