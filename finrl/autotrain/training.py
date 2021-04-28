import datetime

import pandas as pd
import matplotlib
# import matplotlib.pyplot as plt
# from sklearn import preprocessing

from finrl.config import config
from finrl.marketdata.utils import fetch_and_store, load
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import calculate_split, data_split
from finrl.env.env_stocktrading import StockTradingEnv
from finrl.model.models import DRLAgent
from finrl.trade.backtest import backtest_stats  # , backtest_plot, get_daily_return, get_baseline

matplotlib.use("Agg")


def train_one(fetch=False):
    """
    train an agent
    """
    if fetch:
        df = fetch_and_store()
    else:
        df = load()

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
    start_date, trade_date, end_date = calculate_split(df,
                                                       start=config.START_DATE)
    print(start_date, trade_date, end_date)
    train = data_split(processed, start_date, trade_date)
    trade = data_split(processed, trade_date, end_date)

    # calculate state action space
    stock_dimension = len(train.tic.unique())
    state_space = (
        1 + (2 * stock_dimension) +
        (len(config.TECHNICAL_INDICATORS_LIST) * stock_dimension)
    )

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

    e_trade_gym = StockTradingEnv(
        df=trade,
        turbulence_threshold=250,
        **env_kwargs
    )

    env_train, _ = e_train_gym.get_sb_env()
    env_trade, obs_trade = e_trade_gym.get_sb_env()

    agent = DRLAgent(env=env_train)

    print("==============Model Training===========")
    now = datetime.datetime.now().strftime(config.DATETIME_FMT)

    model_sac = agent.get_model("sac")
    trained_sac = agent.train_model(
        model=model_sac,
        tb_log_name="sac",
        total_timesteps=100
        # total_timesteps=80000
    )

    print("==============Start Trading===========")
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        # model=trained_sac, test_data=trade, test_env=env_trade, test_obs=obs_trade
        trained_sac,
        env_trade)
    df_account_value.to_csv(
        "./" + config.RESULTS_DIR + "/df_account_value_" + now + ".csv"
    )
    df_actions.to_csv("./" + config.RESULTS_DIR + "/df_actions_" + now + ".csv")

    print("==============Get Backtest Results===========")
    perf_stats_all = backtest_stats(df_account_value)
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv(
        "./" + config.RESULTS_DIR + "/perf_stats_all_" + now + ".csv"
    )
