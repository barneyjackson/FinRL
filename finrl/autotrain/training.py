import datetime

import matplotlib
import numpy as np
import pandas as pd
import pytz

from finrl.config import config
from finrl.marketdata.utils import fetch_and_store, load
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import calculate_split, data_split
from finrl.env.env_stocktrading import StockTradingEnv
from finrl.model.models import DRLAgent
from finrl.trade.backtest import backtest_stats, backtest_plot

matplotlib.use("Agg")


def train_one(fetch=False):
    """
    train an agent
    """
    if fetch:
        df = fetch_and_store()
    else:
        df = load()

    counts = df[['date', 'tic']].groupby(['date']).count().tic
    assert counts.min() == counts.max()

    print("==============Start Feature Engineering===========")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
        use_turbulence=True,
        # use_turbulence=False,
        user_defined_feature=False,
    )

    processed = fe.preprocess_data(df)

    # Training & Trading data split
    start_date, trade_date, end_date = calculate_split(df, start=config.START_DATE)
    print(start_date, trade_date, end_date)
    train = data_split(processed, start_date, trade_date)
    trade = data_split(processed, trade_date, end_date)

    print(f'\n******\nRunning from {start_date} to {end_date} for:\n{", ".join(config.CRYPTO_TICKER)}\n******\n')

    # calculate state action space
    stock_dimension = len(train.tic.unique())
    state_space = (1 + (2 * stock_dimension) + (len(config.TECHNICAL_INDICATORS_LIST) * stock_dimension))

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 100000,
        "buy_cost_pct": 0.0026,
        "sell_cost_pct": 0.0026,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }

    e_train_gym = StockTradingEnv(df=train, **env_kwargs)

    e_trade_gym = StockTradingEnv(df=trade, turbulence_threshold=250, make_plots=True, **env_kwargs)

    env_train, _ = e_train_gym.get_sb_env()
    env_trade, obs_trade = e_trade_gym.get_sb_env()

    agent = DRLAgent(env=env_train)

    print("==============Model Training===========")
    now = datetime.datetime.now().strftime(config.DATETIME_FMT)

    model_sac = agent.get_model("sac")
    trained_sac = agent.train_model(
        model=model_sac,
        tb_log_name="sac",
        # total_timesteps=100
        total_timesteps=80000
    )

    print("==============Start Trading===========")
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        # model=trained_sac, test_data=trade, test_env=env_trade, test_obs=obs_trade
        trained_sac,
        e_trade_gym)
    df_account_value.to_csv(f"./{config.RESULTS_DIR}/df_account_value_{now}.csv")
    df_actions.to_csv(f"./{config.RESULTS_DIR}/df_actions_{now}.csv")

    df_txns = pd.DataFrame(e_trade_gym.transactions, columns=['date', 'amount', 'price', 'symbol'])
    df_txns = df_txns.set_index(pd.DatetimeIndex(df_txns['date'], tz=pytz.utc))
    df_txns.to_csv(f'./{config.RESULTS_DIR}/df_txns_{now}.csv')

    df_positions = pd.DataFrame(e_trade_gym.positions, columns=['date', 'cash'] + config.CRYPTO_TICKER)
    df_positions = df_positions.set_index(pd.DatetimeIndex(df_positions['date'], tz=pytz.utc)).drop(columns=['date'])
    df_positions['cash'] = df_positions.astype({col: np.float64 for col in df_positions.columns})
    df_positions.to_csv(f'./{config.RESULTS_DIR}/df_positions_{now}.csv')

    print("==============Get Backtest Results===========")
    perf_stats_all = backtest_stats(df_account_value, transactions=df_txns, positions=df_positions)
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv(f"./{config.RESULTS_DIR}/perf_stats_all_{now}.csv")

    backtest_plot(
        df_account_value,
        baseline_start=trade_date,
        baseline_end=end_date,
        positions=df_positions,
        transactions=df_txns
    )
