import os
from datetime import timedelta, datetime

import dateutil
import numpy as np

from SRC.CORE._CONSTANTS import CLASSES, POWER_DEGREE, UNBALANCED_CENTER_RATIO, CACHED_FOLDER_PATH, \
    TRAIN_FOLDER_PATH, DISCRETIZATION, PROCESSED_FOLDER_PATH, FEATURE_DIFF, FEATURE_2, CUT_OFFSET_KEY, FEATURE_KEY, \
    SPACE_PRODUCER_KEY, FEATURE_2_NORM, START_TRAIN_KEY, \
    FEATURE_NORM_KEY, SEGMENT_OFFSET_KEY, FEATURE_DIFF_NORM, SPACE_PRODUCER_NORM_KEY, FEATURE_3, FEATURE_3_NORM, \
    END_TRAIN_KEY, AUC_ROC_DOWN_SAMPLING_LIMIT, NON_LINEARITY_TOP, META_FOLDER_PATH, \
    MODEL_FOLDER_PATH, SEGMENT_LENGTH, ROLLING_GRAD_WINDOW_FREQs, SYMBOL_TRADING_KEY, TRADE_FOLDER_PATH, TARGET_FEATURE, \
    EPOCHS, \
    LEARNING_RATE, WEIGHTS_BOOSTER_COEF, MODEL_SUFFIX, TRAIN_SYMBOLS, NOTEBOOK_NAME_KEY, _REGIME, \
    _DASHBOARD_SEGMENT_BACKTESTING, PROCESS_SYMBOL, EMPTY_NETWORK_KEY, CLASS_KEY, TARGET_FEATURE_WINDOW_KEY, \
    BATCH_SIZE, BATCH_SIZE_KEY, NETWORK_KEY, MODEL_SUFFIX_KEY, META_SUFFIX_KEY, META_SUFFIX, IS_CONVOLUTIONAL_KEY, \
    CONV_KEY, START_CACHE_PROCESS_KEY, END_CACHE_PROCESS_KEY, \
    SIMULATION__START_DATE_KEY, SIMULATION__END_DATE_KEY, FORCE_DISCRETIZATION_KEY, FORCE_DISCRETIZATION, \
    _CONFIGS_SUFFIX, _MODEL_SUFFIX, _DASHBOARD_SEGMENT, _AUTOMATION_TYPE, _BACKTESTING
from SRC.CORE.debug_utils import printmd
from SRC.CORE.utils import calc_symmetric_pow_space, process_format_precision_order, datetime_h_m__d_m, filter_pairs

_float = lambda val: process_format_precision_order(val)

PRODUCE_CACHED_FILE_NAME = lambda pair: f'{CACHED_FOLDER_PATH}/{pair}-{FORCE_DISCRETIZATION()}-cache.csv'
PRODUCE_PROCESSED_FILE_NAME = lambda pair: lambda feature: f'{PROCESSED_FOLDER_PATH}/{pair}-{DISCRETIZATION()}-{feature}-processed.csv'
PRODUCE_TRAIN_FILE_NAME = lambda feature_norm, suffix=None: f"{TRAIN_FOLDER_PATH}/{DISCRETIZATION()}__{feature_norm}__CL_{CLASSES()}__PD_{_float(POWER_DEGREE())}__NLT_{_float(NON_LINEARITY_TOP())}{f'__MS_{suffix}' if suffix is not None else ''}-train.csv"
PRODUCE_META_FILE_NAME = lambda suffix=None: f"{META_FOLDER_PATH}/{DISCRETIZATION()}__CL_{CLASSES()}__PD_{_float(POWER_DEGREE())}__NLT_{_float(NON_LINEARITY_TOP())}{f'__MS_{suffix}' if suffix is not None else ''}-meta.json"
PRODUCE_MODEL_FILE_NAME = lambda meta_suffix, model_suffix=None: f"{MODEL_FOLDER_PATH}/{DISCRETIZATION()}__{TARGET_FEATURE()}__CL_{CLASSES()}__PD_{_float(POWER_DEGREE())}__NLT_{_float(NON_LINEARITY_TOP())}__EP_{EPOCHS()}__BS_{BATCH_SIZE()}__LR_{_float(LEARNING_RATE())}__WBC_{_float(WEIGHTS_BOOSTER_COEF())}__MS_{meta_suffix}|||{model_suffix}-model.pt"
PRODUCE_TRADE_FILE_PATH = lambda is_simulation_, save_symbol_, min_date_, max_date_, suffix_: f"{TRADE_FOLDER_PATH(is_simulation_)}/{DISCRETIZATION()}__{TARGET_FEATURE()}__{CLASSES()}__{save_symbol_}__{min_date_.strftime('%Y_%m_%d__%H:%M:%S')}-{max_date_.strftime('%Y_%m_%d__%H:%M:%S')}__{suffix_}.json".replace('"', "")
PRODUCE_OUT_FOLDER_PATH = lambda: f"{DISCRETIZATION()}__{TARGET_FEATURE()}__CL_{CLASSES()}__PD_{POWER_DEGREE()}__NLT_{NON_LINEARITY_TOP()}__EP_{EPOCHS()}__BS_{BATCH_SIZE()}__LR_{LEARNING_RATE()}__WBC_{WEIGHTS_BOOSTER_COEF()}{f'__MS_{MODEL_SUFFIX()}' if MODEL_SUFFIX() is not None else ''}"

# PROCUCE_PAIRS_BY_TRADE_SYMBOLS = lambda trade_symbols: list(filter(lambda p: p[SYMBOL_TRADING_KEY] in trade_symbols, PAIRS()))
PROCUCE_PAIRS_BY_TRADE_SYMBOLS = lambda trade_symbols: filter_pairs(PAIRS(), trade_symbols, SYMBOL_TRADING_KEY)

PRODUCE_PAIR_BY_TRADE_SYMBOL = lambda trade_symbol: PROCUCE_PAIRS_BY_TRADE_SYMBOLS([trade_symbol])[0]

PAIRS = (lambda: fill_pairs_missed_keys([
#STABLE TO STABLE COIN | NO FEE
    {
        SYMBOL_TRADING_KEY: 'TUSD/USDT',
        START_TRAIN_KEY: dateutil.parser.parse("2020-01-01T00:00:00Z"),
        END_TRAIN_KEY: dateutil.parser.parse("2024-01-01T00:00:00Z"),
    },
    {
        SYMBOL_TRADING_KEY: 'TUSD/BUSD',
        START_TRAIN_KEY: dateutil.parser.parse("2023-05-01T00:00:00Z"),
        END_TRAIN_KEY: dateutil.parser.parse("2024-01-01T00:00:00Z"),
    },
    {
        SYMBOL_TRADING_KEY: 'FDUSD/USDT',
        START_TRAIN_KEY: dateutil.parser.parse("2023-07-01T00:00:00Z"),
        END_TRAIN_KEY: dateutil.parser.parse("2024-01-01T00:00:00Z"),
    },
    {
        SYMBOL_TRADING_KEY: 'FDUSD/BUSD',
        START_TRAIN_KEY: dateutil.parser.parse("2023-07-01T00:00:00Z"),
        END_TRAIN_KEY: dateutil.parser.parse("2024-01-01T00:00:00Z"),
    },
#JOUNG TIMER | HIGH VOLATILITY | SMALL VOLUME
    {
        SYMBOL_TRADING_KEY: 'BETA/USDT',
        START_TRAIN_KEY: dateutil.parser.parse("2023-10-01T00:00:00Z"),
        END_TRAIN_KEY: dateutil.parser.parse("2024-10-15T00:00:00Z")
    },
    {
        SYMBOL_TRADING_KEY: 'MULTI/USDT',
        START_TRAIN_KEY: dateutil.parser.parse("2023-10-01T00:00:00Z"),
        END_TRAIN_KEY: dateutil.parser.parse("2024-10-20T00:00:00Z"),
    },
    {
        SYMBOL_TRADING_KEY: 'VIB/USDT',
        START_TRAIN_KEY: dateutil.parser.parse("2023-10-01T00:00:00Z"),
        END_TRAIN_KEY: dateutil.parser.parse("2024-10-20T00:00:00Z"),
    },
    {
        SYMBOL_TRADING_KEY: 'DOCK/USDT',
        START_TRAIN_KEY: dateutil.parser.parse("2023-10-01T00:00:00Z"),
        END_TRAIN_KEY: dateutil.parser.parse("2024-10-20T00:00:00Z"),
    },
    {
        SYMBOL_TRADING_KEY: 'HOT/USDT',
        START_TRAIN_KEY: dateutil.parser.parse("2023-10-01T00:00:00Z"),
        END_TRAIN_KEY: dateutil.parser.parse("2024-10-20T00:00:00Z"),
    },
    {
        SYMBOL_TRADING_KEY: 'OOKI/USDT',
        START_TRAIN_KEY: dateutil.parser.parse("2023-10-01T00:00:00Z"),
        END_TRAIN_KEY: dateutil.parser.parse("2024-10-20T00:00:00Z"),
    },
    {
        SYMBOL_TRADING_KEY: 'FRONT/USDT',
        START_TRAIN_KEY: dateutil.parser.parse("2023-08-01T00:00:00Z"),
        END_TRAIN_KEY: dateutil.parser.parse("2024-01-01T00:00:00Z")
    },
    {
        SYMBOL_TRADING_KEY: 'BNX/USDT',
        START_TRAIN_KEY: dateutil.parser.parse("2023-09-10T00:00:00Z"),
        END_TRAIN_KEY: dateutil.parser.parse("2023-10-12T00:00:00Z")
    },
    {
        SYMBOL_TRADING_KEY: 'UNFI/USDT',
        START_TRAIN_KEY: dateutil.parser.parse("2023-07-01T00:00:00Z"),
        END_TRAIN_KEY: dateutil.parser.parse("2024-01-01T00:00:00Z")
    },
    {
        SYMBOL_TRADING_KEY: 'PERL/USDT',
        START_TRAIN_KEY: dateutil.parser.parse("2023-10-17T00:00:00Z"),
        END_TRAIN_KEY: dateutil.parser.parse("2024-10-20T00:00:00Z"),
    },
    {
        SYMBOL_TRADING_KEY: 'AUCTION/USDT',
        START_TRAIN_KEY: dateutil.parser.parse("2023-10-17T00:00:00Z"),
        END_TRAIN_KEY: dateutil.parser.parse("2024-10-20T00:00:00Z"),
    },
    {
        SYMBOL_TRADING_KEY: 'REQ/USDT',
        START_TRAIN_KEY: dateutil.parser.parse("2023-10-10T00:00:00Z"),
        END_TRAIN_KEY: dateutil.parser.parse("2024-10-12T00:00:00Z")
    },
    {
        SYMBOL_TRADING_KEY: 'T/USDT',
        START_TRAIN_KEY: dateutil.parser.parse("2023-10-06T00:00:00Z"),
        END_TRAIN_KEY: dateutil.parser.parse("2024-10-12T00:00:00Z")
    },
    {
        SYMBOL_TRADING_KEY: 'LOOM/USDT',
        START_TRAIN_KEY: dateutil.parser.parse("2023-10-06T00:00:00Z"),
        END_TRAIN_KEY: dateutil.parser.parse("2024-10-12T00:00:00Z")
    },
    {
        SYMBOL_TRADING_KEY: 'DOGE/TUSD',
        START_TRAIN_KEY: dateutil.parser.parse("2023-04-28T00:00:00Z"),
        END_TRAIN_KEY: dateutil.parser.parse("2024-10-12T00:00:00Z")
    },
    {
        SYMBOL_TRADING_KEY: 'DOGE/USDT',
        START_TRAIN_KEY: dateutil.parser.parse("2023-04-28T00:00:00Z"),
        END_TRAIN_KEY: dateutil.parser.parse("2024-10-12T00:00:00Z")
    },
    {
        SYMBOL_TRADING_KEY: 'CYBER/USDT',
        START_TRAIN_KEY: dateutil.parser.parse("2023-08-16T00:00:00Z"),
        END_TRAIN_KEY: dateutil.parser.parse("2024-01-01T00:00:00Z")
    },
    {
        SYMBOL_TRADING_KEY: 'BNT/USDT',
        START_TRAIN_KEY: dateutil.parser.parse("2020-03-26T00:00:00Z"),
        END_TRAIN_KEY: dateutil.parser.parse("2024-06-25T00:00:00Z")
    },
    {
        SYMBOL_TRADING_KEY: 'LUNA/USDT',
        START_TRAIN_KEY: dateutil.parser.parse("2021-01-10T00:00:00Z"),
        END_TRAIN_KEY: dateutil.parser.parse("2024-01-01T00:00:00Z")
    },
#OLD TIMER | HUGE VOLUME | LOW VOLATILITY
    {
        SYMBOL_TRADING_KEY: 'XRP/USDT',
        START_TRAIN_KEY: dateutil.parser.parse("2019-01-10T00:00:00Z"),
        END_TRAIN_KEY: dateutil.parser.parse("2024-01-01T00:00:00Z"),
        START_CACHE_PROCESS_KEY: dateutil.parser.parse("2019-01-01T00:00:00Z"),
        END_CACHE_PROCESS_KEY: dateutil.parser.parse("2024-01-01T00:00:00Z"),
    },
    {
        SYMBOL_TRADING_KEY: 'ETH/USDT',
        SIMULATION__START_DATE_KEY: dateutil.parser.parse("2023-10-17T00:00:00Z"),
        SIMULATION__END_DATE_KEY: dateutil.parser.parse("2023-10-19T00:00:00Z"),
        START_TRAIN_KEY: dateutil.parser.parse("2019-01-10T00:00:00Z"),
        END_TRAIN_KEY: dateutil.parser.parse("2024-01-01T00:00:00Z"),
        START_CACHE_PROCESS_KEY: dateutil.parser.parse("2019-01-01T00:00:00Z"),
        END_CACHE_PROCESS_KEY: dateutil.parser.parse("2024-01-01T00:00:00Z"),
    },
    {
        SYMBOL_TRADING_KEY: 'BNB/USDT',
        SIMULATION__START_DATE_KEY: dateutil.parser.parse("2023-10-17T00:00:00Z"),
        SIMULATION__END_DATE_KEY: dateutil.parser.parse("2023-10-19T00:00:00Z"),
        START_TRAIN_KEY: dateutil.parser.parse("2019-01-10T00:00:00Z"),
        END_TRAIN_KEY: dateutil.parser.parse("2024-01-01T00:00:00Z"),
        START_CACHE_PROCESS_KEY: dateutil.parser.parse("2019-01-01T00:00:00Z"),
        END_CACHE_PROCESS_KEY: dateutil.parser.parse("2024-01-01T00:00:00Z"),
    },
    {
        SYMBOL_TRADING_KEY: 'BTC/USDT',
        START_TRAIN_KEY: dateutil.parser.parse("2019-01-10T00:00:00Z"),
        END_TRAIN_KEY: dateutil.parser.parse("2024-01-01T00:00:00Z"),
        START_CACHE_PROCESS_KEY: dateutil.parser.parse("2019-01-01T00:00:00Z"),
        END_CACHE_PROCESS_KEY: dateutil.parser.parse("2024-01-01T00:00:00Z"),
    },
]))
TRAIN_PAIRS = lambda: filter_pairs(PAIRS(), TRAIN_SYMBOLS(), SYMBOL_TRADING_KEY)


FEATURES_CONFIG = (lambda:
[
    {
        FEATURE_KEY: FEATURE_DIFF,
        FEATURE_NORM_KEY: FEATURE_DIFF_NORM,
        SEGMENT_OFFSET_KEY: 1,

        # CUT_OFFSET_KEY: 0.1,
        CUT_OFFSET_KEY: 1,

        SPACE_PRODUCER_KEY: lambda extremums: calc_symmetric_pow_space(classes=CLASSES(), power_degree=POWER_DEGREE(), non_linearity_top=get_feature_abs_max(FEATURE_DIFF, extremums) * NON_LINEARITY_TOP(), space_top=get_feature_abs_max(FEATURE_DIFF, extremums), unbalanced_center_ratio=UNBALANCED_CENTER_RATIO),
        SPACE_PRODUCER_NORM_KEY: lambda extremums: calc_symmetric_pow_space(classes=CLASSES(), power_degree=POWER_DEGREE(), non_linearity_top=get_feature_abs_max(FEATURE_DIFF_NORM, extremums) * NON_LINEARITY_TOP(), space_top=get_feature_abs_max(FEATURE_DIFF_NORM, extremums), unbalanced_center_ratio=UNBALANCED_CENTER_RATIO)
    },
    {
        FEATURE_KEY: FEATURE_2,
        FEATURE_NORM_KEY: FEATURE_2_NORM,
        SEGMENT_OFFSET_KEY: 2,

        # CUT_OFFSET_KEY: 0.1,
        CUT_OFFSET_KEY: 1,

        SPACE_PRODUCER_KEY: lambda extremums: calc_symmetric_pow_space(classes=CLASSES(), power_degree=POWER_DEGREE(), non_linearity_top=get_feature_abs_max(FEATURE_2, extremums) * NON_LINEARITY_TOP(), space_top=get_feature_abs_max(FEATURE_2, extremums), unbalanced_center_ratio=UNBALANCED_CENTER_RATIO),
        SPACE_PRODUCER_NORM_KEY: lambda extremums: calc_symmetric_pow_space(classes=CLASSES(), power_degree=POWER_DEGREE(), non_linearity_top=get_feature_abs_max(FEATURE_2_NORM, extremums) * NON_LINEARITY_TOP(), space_top=get_feature_abs_max(FEATURE_2_NORM, extremums), unbalanced_center_ratio=UNBALANCED_CENTER_RATIO)
    },
    {
        FEATURE_KEY: FEATURE_3,
        FEATURE_NORM_KEY: FEATURE_3_NORM,
        SEGMENT_OFFSET_KEY: 3,

        # CUT_OFFSET_KEY: 0.1,
        CUT_OFFSET_KEY: 1,

        SPACE_PRODUCER_KEY: lambda extremums: calc_symmetric_pow_space(classes=CLASSES(), power_degree=POWER_DEGREE(), non_linearity_top=get_feature_abs_max(FEATURE_3, extremums) * NON_LINEARITY_TOP(), space_top=get_feature_abs_max(FEATURE_3, extremums), unbalanced_center_ratio=UNBALANCED_CENTER_RATIO),
        SPACE_PRODUCER_NORM_KEY: lambda extremums: calc_symmetric_pow_space(classes=CLASSES(), power_degree=POWER_DEGREE(), non_linearity_top=get_feature_abs_max(FEATURE_3_NORM, extremums) * NON_LINEARITY_TOP(), space_top=get_feature_abs_max(FEATURE_3_NORM, extremums), unbalanced_center_ratio=UNBALANCED_CENTER_RATIO)
    }
])


def SEGMENT_TIME_DELTA_EXCEPT_LAST(feature, interval, partitioning):
    if partitioning.upper() == 'M':
        return timedelta(minutes=interval * (SEGMENT_LENGTH() - list(filter(lambda item: item[FEATURE_KEY] == feature, FEATURES_CONFIG()))[0][SEGMENT_OFFSET_KEY] + max(ROLLING_GRAD_WINDOW_FREQs)))

    if partitioning.upper() == 'H':
        return timedelta(hours=interval * (SEGMENT_LENGTH() - list(filter(lambda item: item[FEATURE_KEY] == feature, FEATURES_CONFIG()))[0][SEGMENT_OFFSET_KEY] + max(ROLLING_GRAD_WINDOW_FREQs)))

AUC_ROC_DOWN_SAMPLING_RATIO_PRODUCER = lambda count: int(count / AUC_ROC_DOWN_SAMPLING_LIMIT()) if count / AUC_ROC_DOWN_SAMPLING_LIMIT() > 1 else 1

get_exclude_extremums_configs_present = lambda: ' | '.join([f"{feature_config[FEATURE_KEY]} = {feature_config[CUT_OFFSET_KEY]}" for feature_config in FEATURES_CONFIG() if CUT_OFFSET_KEY in feature_config])
get_normalize_configs_present = lambda: ' | '.join([f"{exclude_extremum_config[FEATURE_KEY]} > {exclude_extremum_config[FEATURE_NORM_KEY]}" for exclude_extremum_config in FEATURES_CONFIG()])

get_filtered_pairs = lambda df: [pair.split('-')[0] for pair in df.keys() if pair.split('-')[0] in [PROCESS_SYMBOL(pair[SYMBOL_TRADING_KEY]) for pair in PAIRS()]]
get_exclude_dates_train_configs_present = lambda pairs: " | ".join([f"{pair[SYMBOL_TRADING_KEY]} ({pair[START_TRAIN_KEY].strftime('%Y-%m-%d')} : {pair[END_TRAIN_KEY].strftime('%Y-%m-%d')})" for pair in pairs if START_TRAIN_KEY in pair and END_TRAIN_KEY in pair])
get_exclude_dates_cache_process_configs_present = lambda pairs: " | ".join([f"{pair[SYMBOL_TRADING_KEY]} ({pair[START_CACHE_PROCESS_KEY].strftime('%Y-%m-%d')} : {pair[END_CACHE_PROCESS_KEY].strftime('%Y-%m-%d')})" for pair in pairs if START_CACHE_PROCESS_KEY in pair and END_CACHE_PROCESS_KEY in pair])


def get_date_train_constraints(symbol):
    pair = list(filter(lambda _pair: PROCESS_SYMBOL(_pair[SYMBOL_TRADING_KEY]) == symbol, PAIRS()))[0]
    start_date = dateutil.parser.parse("2017-01-01T00:00:00Z")
    end_date = dateutil.parser.parse("2026-01-01T00:00:00Z")
    if START_TRAIN_KEY in pair:
        start_date = pair[START_TRAIN_KEY]
    if END_TRAIN_KEY in pair:
        end_date = pair[END_TRAIN_KEY]

    return start_date, end_date


def get_date_cache_process_constraints(symbol):
    pair = list(filter(lambda _pair: PROCESS_SYMBOL(_pair[SYMBOL_TRADING_KEY]) == symbol, PAIRS()))[0]
    start_date = dateutil.parser.parse("2017-01-01T00:00:00Z")
    end_date = dateutil.parser.parse("2026-01-01T00:00:00Z")
    if START_CACHE_PROCESS_KEY in pair:
        start_date = pair[START_CACHE_PROCESS_KEY]
    if END_CACHE_PROCESS_KEY in pair:
        end_date = pair[END_CACHE_PROCESS_KEY]

    return start_date, end_date


def get_feature_abs_max(feature, extremums):
    abs_max = np.maximum(np.abs(extremums[f'{feature}_max']), np.abs(extremums[f'{feature}_min']))

    return abs_max

# make_dir(CACHED_FOLDER_PATH, PROCESSED_FOLDER_PATH, TRAIN_FOLDER_PATH, META_FOLDER_PATH, MODEL_FOLDER_PATH, TRADE_FOLDER_PATH(True), TRADE_FOLDER_PATH(False))

CURRENT_REGIME = lambda: os.environ[_REGIME]
IS_EMPTY_NETWORK = lambda: NETWORK_KEY in os.environ and os.environ[NETWORK_KEY] == EMPTY_NETWORK_KEY


def VALIDATE_CONFIGS(symbol=None):
    notebook = os.environ[NOTEBOOK_NAME_KEY]
    if 'pretrain' in notebook:
        for train_pair in TRAIN_SYMBOLS():
            assert PRODUCE_PAIR_BY_TRADE_SYMBOL(train_pair)

        return

    if 'train' in notebook:
        assert os.path.exists(PRODUCE_META_FILE_NAME(META_SUFFIX())), f"NO META EXIST {PRODUCE_META_FILE_NAME(META_SUFFIX())}"

        return

    if 'trade' in notebook:
        if not IS_EMPTY_NETWORK():
            assert os.path.exists(PRODUCE_MODEL_FILE_NAME(META_SUFFIX(), MODEL_SUFFIX())), f"NO MODEL EXIST {PRODUCE_MODEL_FILE_NAME(META_SUFFIX(), MODEL_SUFFIX())}"

        assert os.path.exists(PRODUCE_META_FILE_NAME(META_SUFFIX())), f"NO META EXIST {PRODUCE_META_FILE_NAME(META_SUFFIX())}"

        if _AUTOMATION_TYPE in os.environ and os.environ[_AUTOMATION_TYPE] == _BACKTESTING:
            assert len(list(filter(lambda dict: dict[SYMBOL_TRADING_KEY] == symbol, PAIRS()))) == 1, f"NO {symbol} in PAIRS()"
            simulation_cached_file_path = PRODUCE_CACHED_FILE_NAME(PROCESS_SYMBOL(symbol))
            assert os.path.exists(simulation_cached_file_path), f"NO CHACHED FILE EXIST for {symbol} > {simulation_cached_file_path}"

        return


def fill_pairs_missed_keys(pairs):
    for pair in pairs:
        if START_CACHE_PROCESS_KEY not in pair and START_TRAIN_KEY in pair:
            pair[START_CACHE_PROCESS_KEY] = pair[START_TRAIN_KEY]

        if START_TRAIN_KEY not in pair and START_CACHE_PROCESS_KEY in pair:
            pair[START_TRAIN_KEY] = pair[START_CACHE_PROCESS_KEY]

        if END_CACHE_PROCESS_KEY not in pair and END_TRAIN_KEY in pair:
            pair[END_CACHE_PROCESS_KEY] = pair[END_TRAIN_KEY]

        if END_TRAIN_KEY not in pair and END_CACHE_PROCESS_KEY in pair:
            pair[END_TRAIN_KEY] = pair[END_CACHE_PROCESS_KEY]

    return pairs


def GET_META_FILE_NAME_BY_ORDER(order):
    from pathlib import Path

    folder_path = META_FOLDER_PATH

    all_files = [file for file in Path(folder_path).iterdir() if file.is_file() and file.suffix == '.json']
    sorted_files = sorted(all_files, key=lambda x: x.stat().st_mtime, reverse=False)

    meta_files = [file.name for file in sorted_files]
    meta_file = meta_files[order]

    return meta_file


def GET_MODEL_FILE_NAME_BY_ORDER(order):
    from pathlib import Path

    folder_path = MODEL_FOLDER_PATH

    all_files = [file for file in Path(folder_path).iterdir() if file.is_file() and file.suffix == '.pt']
    sorted_files = sorted(all_files, key=lambda x: x.stat().st_mtime, reverse=False)

    model_files = [file.name for file in sorted_files]
    model_file = model_files[order]

    return model_file


def SET_PRETRAIN_STATE(offset_constraints):
    train_symbols_str = "|".join(TRAIN_SYMBOLS())
    offset_constraints_str = "|".join(map(lambda offset: str(offset), list(offset_constraints.values())))
    meta_suffix = f"{offset_constraints_str}||{train_symbols_str}"
    os.environ[META_SUFFIX_KEY] = meta_suffix


def SET_TRAIN_STATE(meta_num, target_feature_window, model_suffix=None):
    from SRC.CORE._CONSTANTS import DISCRETIZATION_KEY, POWER_DEGREE_KEY, NON_LINEARITY_TOP_KEY

    meta_file_name = GET_META_FILE_NAME_BY_ORDER(meta_num)

    chunks = meta_file_name.replace('-meta.json', '').split("__")
    discretization = chunks[0]
    classes = chunks[1].split("_")[1]
    power_degree = chunks[2].split("_")[1]
    non_linearity_top = chunks[3].split("_")[1]
    meta_suffix = chunks[4].split("_")[1]

    os.environ[TARGET_FEATURE_WINDOW_KEY] = str(target_feature_window)
    os.environ[DISCRETIZATION_KEY] = discretization
    os.environ[CLASS_KEY] = classes
    os.environ[POWER_DEGREE_KEY] = power_degree
    os.environ[NON_LINEARITY_TOP_KEY] = non_linearity_top
    os.environ[META_SUFFIX_KEY] = meta_suffix
    os.environ[MODEL_SUFFIX_KEY] = datetime_h_m__d_m(datetime.now()).replace(' ', '||').replace('-', '|').replace(':', '|')
    os.environ[IS_CONVOLUTIONAL_KEY] = 'True' if model_suffix is not None and CONV_KEY in model_suffix else 'False'
    if model_suffix is not None:
        os.environ[MODEL_SUFFIX_KEY] = f"{os.environ[MODEL_SUFFIX_KEY]}||{model_suffix}"

    model_file_path = PRODUCE_MODEL_FILE_NAME(os.environ[META_SUFFIX_KEY], os.environ[MODEL_SUFFIX_KEY])
    model_file_name = model_file_path.split('/')[-1]
    printmd(f"Meta: **{meta_file_name}**")
    printmd(f"Model: **{model_file_name}**")
    printmd(f"Network: **EMPTY**")

    os.environ[NETWORK_KEY] = EMPTY_NETWORK_KEY

    return meta_file_name, model_file_name


def SET_NETWORK_STATE(order, force_discretization=None):
    from SRC.CORE._CONSTANTS import DISCRETIZATION_KEY, POWER_DEGREE_KEY, NON_LINEARITY_TOP_KEY, EPOCHS_KEY, WEIGHTS_BOOSTER_COEF_KEY, LEARNING_RATE_KEY

    if order == 0:
        raise Exception('Wrong order number')
    else:
        order = order if order < 0 else order - 1
        file_name = GET_MODEL_FILE_NAME_BY_ORDER(order)

        chunks = file_name.replace('-model.pt', '').split("__")
        discretization = chunks[0]
        feature_window = '1' if chunks[1].lower() == FEATURE_DIFF.lower() else chunks[1].lower().split("_")[2]
        classes = chunks[2].split("_")[1]
        power_degree = chunks[3].split("_")[1]
        non_linearity_top = chunks[4].split("_")[1]
        epochs = chunks[5].split("_")[1]
        batch_size = chunks[6].split("_")[1]
        learning_rate = chunks[7].split("_")[1]
        weights_booster_coef = chunks[8].split("_")[1]
        suffix = chunks[9].split("_")[1] if len(chunks) > 9 else 'None'
        meta_suffix = suffix.split("|||")[0]
        model_suffix = suffix.split("|||")[1]

        os.environ[DISCRETIZATION_KEY] = discretization
        if force_discretization is not None:
            os.environ[FORCE_DISCRETIZATION_KEY] = force_discretization
        os.environ[TARGET_FEATURE_WINDOW_KEY] = feature_window
        os.environ[CLASS_KEY] = classes
        os.environ[POWER_DEGREE_KEY] = power_degree
        os.environ[NON_LINEARITY_TOP_KEY] = non_linearity_top
        os.environ[EPOCHS_KEY] = epochs
        os.environ[BATCH_SIZE_KEY] = batch_size
        os.environ[LEARNING_RATE_KEY] = learning_rate
        os.environ[WEIGHTS_BOOSTER_COEF_KEY] = weights_booster_coef
        os.environ[META_SUFFIX_KEY] = meta_suffix
        os.environ[MODEL_SUFFIX_KEY] = model_suffix
        os.environ[IS_CONVOLUTIONAL_KEY] = 'True' if 'CONV' in model_suffix else 'False'

        os.environ[NETWORK_KEY] = file_name

    meta_file_path = PRODUCE_META_FILE_NAME(os.environ[META_SUFFIX_KEY])
    model_file_path = PRODUCE_MODEL_FILE_NAME(os.environ[META_SUFFIX_KEY], os.environ[MODEL_SUFFIX_KEY])
    meta_file_name = meta_file_path.split('/')[-1]
    model_file_name = model_file_path.split('/')[-1]

    printmd(f"Meta: **{meta_file_name}**")
    printmd(f"Model: **{model_file_name}**")
    printmd(f"Network: **PRETRAINED**")

    return meta_file_name, model_file_name


def SET_SIMULATION_STATE(network_num, fee_enabled, force_discretization=None):
    import os
    from SRC.CORE._CONSTANTS import _DASHBOARD_SEGMENT_BACKTESTING, _BINANCE_FEE_ENABLED

    SET_NETWORK_STATE(network_num, force_discretization=force_discretization)

    os.environ[_BINANCE_FEE_ENABLED] = str(fee_enabled)
    os.environ[_DASHBOARD_SEGMENT] = _DASHBOARD_SEGMENT_BACKTESTING


def IS_STAGE_3():
    return (_CONFIGS_SUFFIX in os.environ and 'stage3' in os.environ[_CONFIGS_SUFFIX]) or (_MODEL_SUFFIX in os.environ and 'stage3' in os.environ[_MODEL_SUFFIX])
