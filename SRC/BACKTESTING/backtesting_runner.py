import multiprocessing
from multiprocessing import Pool
import os

from SRC.CORE._CONSTANTS import _UTC_TIMESTAMP, _DISCRETIZATION
from SRC.CORE.debug_utils import SET_CONSOLE_LOGLEVEL
from SRC.CORE.debug_utils import is_running_under_pycharm, SET_SYMBOL
from SRC.LIBRARIES.new_data_utils import fetch_featurize_group_iterate
from SRC.LIBRARIES.new_utils import floor, __uuid4_12 as uuid4_12, produce_net_folder, INITIALIZE_NETFOLDER, parse_net_folder_hashed, write_datetime_price_candle
from SRC.NN.IModelBase import produce_model
from SRC.TESTS.test_margin_dashboard_trades_executor_new import produce_mock_margin_isolated_binance_client_singleton
from SRC.WEBAPP.libs.BinanceMarginIsolatedTrader import trade_utils_producer


def perform_backtesting(data):
    symbol = data["symbol"]
    model_name = data["model_name"]
    start_dt = data["start_dt"]
    end_dt = data["end_dt"]
    fee = data["fee"]
    plr = data["plr"]

    model = produce_model(model_name)
    segments = model.segments_count()
    discretization_s = model.discretization_s()
    discretization_feature_s = model.discretization_feature_s()
    inference_discretization = discretization_s[0]

    SET_SYMBOL(symbol)
    SET_CONSOLE_LOGLEVEL()

    market_type = 'MARGIN'
    data['market'] = f"{market_type}__REGIME_MOCK__INF_DISCR_{inference_discretization}__FEE_{fee}"
    net_folder = produce_net_folder(data)
    net_data = parse_net_folder_hashed(net_folder)

    os.environ[_DISCRETIZATION] = inference_discretization
    os.environ['INITIAL_USDT_DEPOSIT'] = '1000'
    os.environ['TRANSFER_CROSS_ISOLATED'] = 'True'
    os.environ['MAX_ORDERS_COUNT'] = '5'                    # Max 5 open orders at the same time
    os.environ['STOP_ON_JUMP_UP_RATIO'] = '5'               # Stop if balance explodes more than 5x
    os.environ['STOP_ON_DROP_DOWN_RATIO'] = '0.2'           # Stop if balance drops more than 5x
    os.environ['OVER_TIMEOUT'] = '25'                       # Stop if order is not filled after 25 hours (to avoid hanging orders in backtesting)
    os.environ['NO_TRADES_TIMEOUT'] = '10'                  # Stop if no trades timeout is more than threshold
    os.environ['RUN_TO_END'] = 'True'                       # Run to the end of backtesting even if balance drops or explodes (ignore STOP_ON_JUMP_UP_RATIO and STOP_ON_DROP_DOWN_RATIO)
    os.environ['DASHBOARD_SEGMENT'] = 'BACKTESTING'
    os.environ['BINANCE_FEE'] = fee
    os.environ['NET_FOLDER'] = net_folder

    _CLEAR_FILES = INITIALIZE_NETFOLDER('BACKTESTING', reset=True)
    _CLEAR_FILES()

    client = produce_mock_margin_isolated_binance_client_singleton()
    trade_utils = trade_utils_producer(client, net_data, ext_log_s=None)

    executors = trade_utils['executors_producer'](None, hash)
    executorBacktestingSync = executors[0]  
    group_iterator = fetch_featurize_group_iterate(market_type, symbol, segments, discretization_s, discretization_feature_s, start_dt, end_dt)

    first_group = next(group_iterator)
    write_datetime_price_candle(first_group)

    start_dt_utc = first_group[0].iloc[0][_UTC_TIMESTAMP]
    executorBacktestingSync.initialize(start_dt_utc, symbol)

    correlation_id = uuid4_12()

    present_total_net_USDT = trade_utils['get_present_total_net_USDT']()
    print(f"STARTED [{net_data['hash']}|{correlation_id}]:\r\n{net_folder}\r\n{str(data)}\r\nINITIAL {present_total_net_USDT}")

    for group in group_iterator:
        write_datetime_price_candle(group)
        prediction = model.predict_signal(group)

        correlation_id = prediction['correlation_id'] if 'correlation_id' in prediction else correlation_id
        transaction_id = prediction['transaction_id'] if 'transaction_id' in prediction else uuid4_12()
        signal = prediction['signal']
        take_profit_ratio = prediction['take_profit_ratio']
        profit_loss_ratio = prediction['profit_loss_ratio'] if 'profit_loss_ratio' in prediction else plr

        stop_loss_ratio = floor(1 + (take_profit_ratio - 1) / profit_loss_ratio, 4)

        executorBacktestingSync.execute_oco_orders()

        if signal == 'LONG' and executorBacktestingSync.can_execute_long_order(transaction_id=transaction_id, correlation_id=correlation_id):
            executorBacktestingSync.execute_long_order(take_profit_ratio, stop_loss_ratio, transaction_id=transaction_id, correlation_id=correlation_id)

            continue

        if signal == 'SHORT' and executorBacktestingSync.can_execute_short_order(transaction_id=transaction_id, correlation_id=correlation_id):
            executorBacktestingSync.execute_short_order(take_profit_ratio, stop_loss_ratio, transaction_id=transaction_id, correlation_id=correlation_id)

            continue

        executorBacktestingSync.ignore(transaction_id=transaction_id, correlation_id=correlation_id)

    present_total_net_USDT = trade_utils['get_present_total_net_USDT']()
    print(f"FINISHED [{net_data['hash']}|{correlation_id}]:\r\n{net_folder}\r\nFINAL {present_total_net_USDT}")


def perform_backtesting_bulk(start_data):
    num_workers = start_data["num_workers"]

    args_s = []
    for symbol in start_data["symbol_s"]:
        for model_name in start_data["model_name_s"]:
            for fee in start_data["fee_s"]:
                args_s.append({
                    'symbol': symbol,
                    'model_name': model_name,
                    'plr': start_data["plr"],
                    'fee': fee,
                    'start_dt': start_data["backtesting_start_dt"],
                    'end_dt': start_data["backtesting_end_dt"],
                })

    if num_workers > 1 and not is_running_under_pycharm():
        multiprocessing.set_start_method("spawn", force=True)
        try:
            with Pool(num_workers) as pool:
                print("Running backtesting in parallel with multiprocessing...")
                for _ in pool.imap_unordered(perform_backtesting, args_s):
                    pass
        finally:
            pool.terminate()
            pool.join()
    else:
        for args in args_s:
            perform_backtesting(args)


if __name__ == "__main__":
    num_workers = 3

    start_data = {
        "num_workers": num_workers,
        "symbol_s": ["ZECUSDT"],
        'plr': 3,
        "model_name_s": ["GradExplosionTest_7", "GradExplosionTest_71", "GradExplosionTest_72"],
        "backtesting_start_dt": "2026-03-15 00:00:00",
        "backtesting_end_dt": "2027-01-01 00:00:00",

        "fee_s": ['0.0-0.0', '0.036-0.036'],
        # "fee_s": ['0.0-0.0'],
    }

    print(f"Starting BULK backtesting: {start_data}")

    perform_backtesting_bulk(start_data)

    print(f"Finished BULK backtesting: {start_data}")
