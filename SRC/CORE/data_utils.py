import inspect
import os
import random
import sys
import time
import traceback
from SRC.CORE.debug_utils import printmd
from SRC.CORE._CONSTANTS import UTC_TZ, KIEV_TZ, _UTC_TIMESTAMP
import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from SRC.LIBRARIES.new_data_utils import retrieve_all

from SRC.CORE._CONSTANTS import PARTITIONING, PROCESSED_FOLDER_PATH, BATCH_MEASURE_SIZE, DISCRETIZATION, FORCE_PROCESS, CACHED_FOLDER_PATH, \
    ROLLING_DIFF_WINDOW_FREQ, FEATURE_DIFF_NORM, MAX_FLUCTUATION_RANGE, FEATURE_KEY, CUT_OFFSET_KEY, SPACE_PRODUCER_KEY, FEATURE_NORM_KEY, SEGMENT_OFFSET_KEY, UNBALANCED_CENTER_RATIO, CLASSES, \
    POWER_DEGREE, FEATURES, START_TRAIN_KEY, SPACE_PRODUCER_NORM_KEY, COUNTS_KEY, BINS_KEY, WEIGHTS_KEY, SPACE_KEY, WEIGHTS_COUNT_PRODUCT_NORM_KEY, SPACE_PRODUCER_SERIALIZED_KEY, BINS_PAIRWISE_KEY, WEIGHTS_BOOSTER_COEF, \
    PROCESSED_DATE_COLS, TARGET_FEATURE, TRAIN_SYMBOLS, SYMBOL_TRADING_KEY, PROCESS_SYMBOL, FEATURE_S_KEY, _SYMBOL, _DISCRETIZATION, FEATURE_DIFF, TZ, _UTC_TIMESTAMP
from SRC.CORE._CONSTANTS import ROLLING_GRAD_WINDOW_FREQs
from SRC.CORE._CONSTANTS import SEGMENT_LENGTH, SEGMENT_OVERLAP
from SRC.CORE._FUNCTIONS import PRODUCE_CACHED_FILE_NAME
from SRC.CORE._FUNCTIONS import PRODUCE_PROCESSED_FILE_NAME
from SRC.CORE._FUNCTIONS import PRODUCE_TRAIN_FILE_NAME, PAIRS, get_exclude_extremums_configs_present, get_filtered_pairs, get_date_train_constraints, \
    FEATURES_CONFIG, get_normalize_configs_present, get_exclude_dates_train_configs_present, get_feature_abs_max, PROCUCE_PAIRS_BY_TRADE_SYMBOLS, get_exclude_dates_cache_process_configs_present, \
    TRAIN_PAIRS
from SRC.CORE.debug_utils import printmd, print_action_title_description, run_medium, print_memory, printmd_low, produce_measure_low, produce_measure_md_low, is_cloud, printmd_high, printmd_medium, \
    display_high, produce_measure_high, print_action_title_description__low, display_medium, print_high, run_high, log_module, display_low, measure_print_action_title_description__low, \
    produce_measure_medium, print_action_title_description__medium
from SRC.CORE.plot_utils import __plot_feature_distribution, plot_series_dependency, get_range_presentation, plot_series_correlation, produce_candles_mins_fig, display_plot
from SRC.CORE.utils import ___calc_space_bins, fetch_featurize_cache_binance, featurize, get_csv_files_from_dir, produce_empty_cached_df, filter_segments_containing_datetime_gaps, \
    process_format_precision_order_6_df, get_sorted_features_list_from_df, get_one_hot_cols, pairwise, get_one_hot_from_class_producer, get_class_from_bin, \
    calc_symmetric_lin_space, wrire_train_meta, case_insensitive_path, run_multi_process, read_train_meta, get_item_from_list_dict, remove_train_meta, \
    process_format_precision_order, _float_6, calc_symmetric_pow_space, build_gradient_presentation_coordinates, datetime_Y_m_d__h_m_s, datetime_h_m__d_m_y
from SRC.CORE.utils import featurize_candles_mins_dashboard
from SRC.CORE.utils import normalize_minus_1__plus_1
from SRC.CORE.utils import split_df_with_overlap


def fetch_featurize_cache_wrapper(process_symbol, trade_symbol, CACHE_FILE_NAME):
    title = f"**FETCHING, FEATURIZING, CACHING..**"
    description = f"**`{process_symbol}`** `>  interval: {DISCRETIZATION()}, features: {FEATURES}, cached file = {CACHE_FILE_NAME}`"
    printmd_low(f"{title}\r\n\r\n{description}")
    run_medium(print_memory)

    measure = produce_measure_md_low(f'**{process_symbol}** FETCHED, FEATURIZED, CACHED')

    _featurize = lambda df, cached_df: featurize(df, cached_df, {
        'PAIR': process_symbol,
        'TRADE_SYMBOL': trade_symbol,
        'PARTITIONING': PARTITIONING(),
        'DISCRETIZATION': DISCRETIZATION(),
        'ROLLING_DIFF_WINDOW_FREQ': ROLLING_DIFF_WINDOW_FREQ,
        'ROLLING_GRAD_WINDOW_FREQs': ROLLING_GRAD_WINDOW_FREQs,
    })
    df = fetch_featurize_cache_binance(process_symbol, trade_symbol, DISCRETIZATION(), CACHE_FILE_NAME, _featurize, save=True)

    printmd_high("**Dataframe head:**")
    display_high(df.head())

    printmd_high("**Dataframe tail:**")
    display_high(df.tail())

    del df

    run_medium(print_memory)
    measure()


def handle__original__normalized_df(grouped__df_dict, feature):
    features_configs = FEATURES_CONFIG()

    original_title = f"####### BOOSTING ORIGINAL DATAFRAME #######"
    print_action_title_description(original_title)

    offset_constraints, original_extremums, original__normalized__pair_grouped__df_dict, original__normalized__extremums_dates_excluded__concatenated_df = exclude__extremums__dates__cached_data(grouped__df_dict, features_configs, exclude_data=False)
    original_feature_space_producer = lambda feature: calc_symmetric_lin_space(CLASSES(), get_feature_abs_max(feature, original_extremums))
    original_cached__bin_weighted_df = produce_optimized_bin_weighted_df(original__normalized__extremums_dates_excluded__concatenated_df, feature, original_feature_space_producer)

    def original_getter():
        return original__normalized__extremums_dates_excluded__concatenated_df, offset_constraints, original_extremums

    def original_plotter():
        original_title = f"####### BOOSTED ORIGINAL DATAFRAME #######"
        excluded_description = f"original size = {len(original__normalized__extremums_dates_excluded__concatenated_df)}"
        print_action_title_description(original_title, excluded_description)

        plot__feature_diff__vs__feature__distribution(original__normalized__extremums_dates_excluded__concatenated_df, FEATURE_DIFF, original_feature_space_producer)
        plot__feature_diff__vs__feature__distribution(original__normalized__extremums_dates_excluded__concatenated_df, feature, original_feature_space_producer)
        display__feature_diff__vs__feature_grad__dependency(original_cached__bin_weighted_df, feature, FEATURE_DIFF, original_feature_space_producer, original_feature_space_producer)

        plot__max_fluctuation_distribution__series_correlation(original__normalized__pair_grouped__df_dict, feature)

    return original_getter, original_plotter


def handle__excluded__normalized_df(grouped__df_dict, feature, feature_norm):
    features_configs = FEATURES_CONFIG()

    original_title = f"BOOSTING EXCLUDING DATAFRAME"
    sub_description = f"{get_exclude_extremums_configs_present()} | {get_exclude_dates_train_configs_present(TRAIN_PAIRS())}"
    original_description = f"exclude configs = {sub_description}"
    print_action_title_description(original_title, original_description)

    offset_constraints, excluded_extremums, excluded__normalized__pair_grouped__df_dict, excluded__normalized__extremums_dates_excluded__concatenated_df = exclude__extremums__dates__cached_data(grouped__df_dict, features_configs, exclude_data=True)

    optimizer_excluded_feature_space_producer = lambda feature: calc_symmetric_pow_space(CLASSES(), POWER_DEGREE(), get_feature_abs_max(feature, excluded_extremums), space_top=get_feature_abs_max(feature, excluded_extremums), unbalanced_center_ratio=UNBALANCED_CENTER_RATIO)
    excluded_cached__bin_weighted_df = produce_optimized_bin_weighted_df(excluded__normalized__extremums_dates_excluded__concatenated_df, feature, optimizer_excluded_feature_space_producer, _multiplier=lambda x: np.power(x, 1/1.5))

    excluded_feature_diff_space_producer = lambda feature_: features_configs[0][SPACE_PRODUCER_KEY](excluded_extremums)
    excluded_feature_diff_norm_space_producer = lambda feature_: features_configs[0][SPACE_PRODUCER_NORM_KEY](excluded_extremums)
    excluded_feature_space_producer = lambda feature_: list(filter(lambda config: config[FEATURE_KEY] == feature, features_configs))[0][SPACE_PRODUCER_KEY](excluded_extremums)
    excluded_feature_norm_space_producer = lambda feature_: list(filter(lambda config: config[FEATURE_NORM_KEY] == feature_norm, features_configs))[0][SPACE_PRODUCER_NORM_KEY](excluded_extremums)

    def excluded_getter():
        return excluded__normalized__extremums_dates_excluded__concatenated_df, offset_constraints, excluded_extremums

    def excluded_plotter():
        original_size = sum([len(df) for key, df in grouped__df_dict.items()])
        excluded_size = sum([len(df) for key, df in excluded__normalized__pair_grouped__df_dict.items()])
        removed = original_size - excluded_size
        removed_percent = removed / original_size * 100

        excluded_title = f"BOOSTED EXCLUDED DATAFRAME"
        excluded_description = f"original size = {original_size} | excluded size = {excluded_size} | removed (count) = {removed} | removed (%) = {removed_percent} | exclude configs: {sub_description}"
        print_action_title_description(excluded_title, excluded_description)

        plot__feature_diff__vs__feature__distribution(excluded__normalized__extremums_dates_excluded__concatenated_df, FEATURE_DIFF, excluded_feature_diff_space_producer)

        plot__feature_diff__vs__feature__distribution(excluded__normalized__extremums_dates_excluded__concatenated_df, feature, excluded_feature_space_producer)
        display__feature_diff__vs__feature_grad__dependency(excluded_cached__bin_weighted_df, feature, FEATURE_DIFF, excluded_feature_space_producer, excluded_feature_diff_space_producer)

        plot__feature_diff__vs__feature__distribution(excluded__normalized__extremums_dates_excluded__concatenated_df, feature_norm, excluded_feature_norm_space_producer)
        display__feature_diff__vs__feature_grad__dependency(excluded_cached__bin_weighted_df, feature_norm, FEATURE_DIFF_NORM, excluded_feature_norm_space_producer, excluded_feature_diff_norm_space_producer)

        plot__max_fluctuation_distribution__series_correlation(excluded__normalized__pair_grouped__df_dict, feature)

    return excluded_getter, excluded_plotter


def plot__feature_distribution__corellation(symbol, pair_df, target_feature, dependent_feature, _condition, title):
    from SRC.CORE.utils import get_loc_by_condition_in_range

    range_selector = _condition(pair_df)
    loc, pair_df_range = get_loc_by_condition_in_range(pair_df, range_selector, in_range=MAX_FLUCTUATION_RANGE())
    range_presentation = get_range_presentation(pair_df_range)

    description = f"symbol = {symbol} | size = {len(pair_df_range)} | loc index = {loc.index.values[0]} | range = {range_presentation}"
    print_action_title_description__low(title, description)
    plot_series_correlation(pair_df_range, target_feature, dependent_feature)

    pair_df_sorted_range = pair_df_range.sort_values(by='timestamp', ascending=True)
    pair_df_sorted_range = build_gradient_presentation_coordinates(symbol, pair_df_sorted_range, ROLLING_GRAD_WINDOW_FREQs)
    display_plot(produce_candles_mins_fig(pair_df_sorted_range, ROLLING_GRAD_WINDOW_FREQs, title=f"{title}: {symbol}"))


def plot__max_fluctuation_distribution__series_correlation(df, feature=TARGET_FEATURE()):
    pairs = get_filtered_pairs(df)
    print_action_title_description__low(f"MAX FLUCTUATIONS DISTRIBUTION", f"pairs={pairs}")

    max_val = 0
    min_val = 0
    df_max = None
    df_min = None
    symbol_max = None
    symbol_min = None
    for pair_key, pair_df in df.items():
        symbol = pair_key.split('-')[0]
        current_max_val = pair_df[feature].max()
        current_min_val = pair_df[feature].min()

        if current_max_val >= max_val:
            max_val = current_max_val
            df_max = pair_df
            symbol_max = symbol

        if current_min_val <= min_val:
            min_val = current_min_val
            df_min = pair_df
            symbol_min = symbol

    plot__feature_distribution__corellation(symbol_max, df_max, feature, FEATURE_DIFF, lambda df: df[feature] == df[feature].max(), "HIGHEST JUMP")
    plot__feature_distribution__corellation(symbol_min, df_min, feature, FEATURE_DIFF, lambda df: df[feature] == df[feature].min(), "DEEPEST DROP")


def retrieve_grouped_cached_data(file_name_pattern, only_pairs=None):
    title = f'**RETRIEVING PAIR GROUPED CACHED DATA..**'
    description = f"`file name pattern={file_name_pattern}`"
    printmd_low(f"{title}\r\n\r\n{description}")

    measure = produce_measure_low('RETRIEVED PAIR GROUPED CACHED DATA')
    run_medium(print_memory)

    cached_data_file_path = lambda file_name: f'{CACHED_FOLDER_PATH}/{file_name}'
    result_df_list = dict()
    cached_files = get_csv_files_from_dir(CACHED_FOLDER_PATH, lambda fn: all(str(ext) in fn for ext in file_name_pattern))
    filtered_pairs = PROCUCE_PAIRS_BY_TRADE_SYMBOLS(only_pairs) if only_pairs is not None else PAIRS()
    pair_filtered_cached_files = [cached_file for cached_file in cached_files if any(PROCESS_SYMBOL(pair[SYMBOL_TRADING_KEY]) in cached_file for pair in filtered_pairs)]
    for cached_file in pair_filtered_cached_files:
        file_path = cached_data_file_path(cached_file)
        file_path_real = case_insensitive_path(file_path)

        df = pd.read_csv(file_path_real, nrows=10)
        float_cols = [c for c in df if df[c].dtype == "float64"]
        float_cols = {c: np.float32 for c in float_cols}
        parse_dates = ['timestamp', 'utc_timestamp', 'kiev_timestamp']
        cached_df = pd.read_csv(
            file_path_real,
            parse_dates=parse_dates,
            dtype=float_cols,
            infer_datetime_format=True)

        result_df_list[cached_file] = cached_df

        printmd_low(f"`cached file: {cached_file} | dataframe size = {len(cached_df)} | first record = {datetime_Y_m_d__h_m_s(cached_df.iloc[0]['utc_timestamp'])} | last record = {datetime_Y_m_d__h_m_s(cached_df.iloc[-1]['utc_timestamp'])}`")
        run_high(lambda: print_memory(df=cached_df))

    measure(f'Total size: {sum([len(df) for df in list(result_df_list.values())])}')

    return result_df_list


def retrieve_cached_data(symbol, CACHE_FILE_NAME):
    cache_file_name_real = case_insensitive_path(CACHE_FILE_NAME)

    title = f'**RETRIEVING CACHED DATA..**'
    description = f"**`{symbol}`** `>  cached file name={cache_file_name_real}`"
    printmd_low(f"{title}\r\n\r\n{description}")

    measure = produce_measure_low('RETRIEVED CACHED DATA')

    fields = [*['pair', 'discretization', 'timestamp', 'utc_timestamp', 'kiev_timestamp',
                'open', 'high', 'low', 'close', 'volume',
                'std', 'mean', 'std_mean_ratio', 'mean_abs_diff'],
                *FEATURES]

    df = pd.read_csv(cache_file_name_real, nrows=10)

    float_cols = [c for c in df if df[c].dtype == "float64"]
    float_cols = {c: np.float32 for c in float_cols}

    df = pd.read_csv(
        cache_file_name_real,
        dtype=float_cols,
        parse_dates=['timestamp', 'utc_timestamp', 'kiev_timestamp'],
        infer_datetime_format=True)[fields]

    run_high(lambda: print_memory(df=df), lambda: print_memory())

    measure(f'Size: {len(df)}')

    return df


def retrieve_processed_data(processed_file_name):
    run_medium(print_memory)

    processed_file_name_real = case_insensitive_path(processed_file_name)
    df = pd.read_csv(processed_file_name_real, nrows=10)
    float_cols = [c for c in df if df[c].dtype == "float64"]
    float_cols = {c: np.float64 for c in float_cols}

    processed_df = pd.read_csv(processed_file_name_real, parse_dates=['start_ts', 'end_ts'], infer_datetime_format=True)

    run_medium(print_memory)

    return processed_df


def run_pair_cache(pair):
    trade_symbol = pair[SYMBOL_TRADING_KEY]
    process_symbol = PROCESS_SYMBOL(trade_symbol)
    fetch_featurize_cache_wrapper(process_symbol, trade_symbol, PRODUCE_CACHED_FILE_NAME(process_symbol))


def runn_all_cache(only_pairs=None):
    from SRC.CORE._FUNCTIONS import PAIRS

    pairs = PROCUCE_PAIRS_BY_TRADE_SYMBOLS(only_pairs) if only_pairs is not None else PAIRS()
    pairs_present = get_exclude_dates_cache_process_configs_present(pairs)
    execution_type = "PARRALEL" if is_cloud() else "SEQUENTIAL"
    measure = produce_measure_md_low(f"**FINISHED CACHE {execution_type}: {pairs_present}..**")

    printmd_low("***`=============================================================================================`***")
    printmd_low(f"**STARTED CACHE {execution_type}: {pairs_present}..**")
    printmd_low("***`=============================================================================================`***")

    run_multi_process(run_pair_cache, pairs)

    printmd_low("***`=============================================================================================`***")
    measure()
    printmd_low("***`=============================================================================================`***")


def runn_all_process(only_pairs=None):
    from SRC.CORE._FUNCTIONS import PAIRS

    pairs = PROCUCE_PAIRS_BY_TRADE_SYMBOLS(only_pairs) if only_pairs is not None else PAIRS()
    pairs_present = get_exclude_dates_cache_process_configs_present(pairs)
    execution_type = "PARRALEL" if is_cloud() else "SEQUENTIAL"
    measure = produce_measure_md_low(f"**FINISHED PROCESS {execution_type}: {pairs_present}..**")

    printmd_low(f"***`=============================================================================================`***")
    printmd_low(f"**STARTED PROCESS {execution_type}: {pairs_present}..**")
    printmd_low(f"***`=============================================================================================`***")

    run_multi_process(run_pair_process, pairs)

    printmd_low("***`=============================================================================================`***")
    measure()
    printmd_low("***`=============================================================================================`***")


def run_pair_process(pair):
    symbol = PROCESS_SYMBOL(pair[SYMBOL_TRADING_KEY])
    
    title_start_pair = f"STARTED PROCESS : {symbol}.."
    printmd_low(f"***`================================\r\n{title_start_pair}\r\n================================`***")

    measure_pair = produce_measure_md_low(f"**`FINISHED PROCESS: {symbol}..`**")

    df = retrieve_cached_data(symbol, PRODUCE_CACHED_FILE_NAME(symbol))
    if len(df) < SEGMENT_LENGTH():
        printmd_low(f"**!!!!!!!!! Symbol {symbol} doesn't have enough records: {len(df)} !!!!!!!!!**")

        return

    filtered_df = filter_cached_data(symbol, PRODUCE_PROCESSED_FILE_NAME(symbol), df)
    segments = produce_segments(symbol, filtered_df)
    process_segments_save_converted(symbol, segments, PRODUCE_PROCESSED_FILE_NAME(symbol))

    measure_pair()
    printmd_low("***`=============================================================================================`***")


def filter_cached_data(pair, produce_processed_file_name, cached_df):
    title = f'**FILTERING CACHED DATA**'
    description = f"**`{pair}`** `>  produce processed file={[produce_processed_file_name(feature) for feature in FEATURES]}`"
    printmd_low(f"{title}\r\n\r\n{description}")

    measure = produce_measure_low('FILTERED CACHED DATA')
    run_medium(print_memory)

    if FORCE_PROCESS:
        measure(f'FROM BEGINNING')

        return cached_df

    last_existing_processed_row_s = []
    for feature in FEATURES:
        try:
            processed_file_name = produce_processed_file_name(feature)
            processed_df = retrieve_processed_data(processed_file_name)
            last_existing_processed_row = processed_df.iloc[-1]
            last_existing_processed_row_s.append(last_existing_processed_row)
        except FileNotFoundError as ex:
            msg = f"***`NO PROCESSED FILE: {ex.filename}`***"
            printmd(msg)
            log_module(msg)

    if len(last_existing_processed_row_s) < len(ROLLING_GRAD_WINDOW_FREQs):
        measure(f'FROM BEGINNING')

        return cached_df

    #TODO:!!!!DATETIME PLAY
    min_end_ts = min(list(map(lambda ser: ser['end_ts'], last_existing_processed_row_s)))
    if min_end_ts >= cached_df.iloc[-1]['utc_timestamp']:
        measure(f"!!NO NEW DATA!!")

        return produce_empty_cached_df()

    min_start_ts = min(list(map(lambda ser: ser['start_ts'], last_existing_processed_row_s)))
    filtered_cached_df = cached_df[cached_df['utc_timestamp'] > min_start_ts]

    run_medium(print_memory)
    measure(f'FROM: {min_start_ts}')

    return filtered_cached_df


# def retrieve_processed_sequential_pair_df(pair, feature):
#     symbol = PROCESS_SYMBOL(pair[SYMBOL_TRADING_KEY])
#     start_from = pair[START_TRAIN_KEY]
#     end_to = pair[END_TRAIN_KEY]
#     file_producer = PRODUCE_PROCESSED_FILE_NAME(symbol)
#     file_path = file_producer(feature)
#     file_path_real = case_insensitive_path(file_path)
#
#     df = pd.read_csv(file_path_real, nrows=10)
#     float_cols = [c for c in df if df[c].dtype == "float64"]
#     float_cols = {c: np.float32 for c in float_cols}
#     processed_sequential_pair_df = pd.read_csv(
#         file_path_real,
#         parse_dates=['start_ts', 'end_ts'],
#         dtype=float_cols,
#         infer_datetime_format=True)
#
#     filtered_processed_sequential_pair_df = processed_sequential_pair_df[processed_sequential_pair_df['start_ts'] >= start_from]
#     filtered_processed_sequential_pair_df = filtered_processed_sequential_pair_df[filtered_processed_sequential_pair_df['end_ts'] <= end_to]
#
#     title = f'RETRIEVED PROCESSED SEQUENTIAL PAIR DF'
#     description = f"symbol = {symbol} | file_path = {file_path} | start_from = {start_from} | end_to = {end_to} | size = {len(filtered_processed_sequential_pair_df)}"
#     print_action_title_description(title, description)
#
#     return filtered_processed_sequential_pair_df


def retrieve_concat_permute_processed_data(file_name_pattern, permute=False, filter_predicate=None, only_pairs=None):
    title = f'RETRIEVING CONCATENATED PROCESSED DATA'
    description = f"file name pattern={file_name_pattern}"
    measure = measure_print_action_title_description__low(title, description)
    run_medium(print_memory)

    processed_data_file_path = lambda file_name: f'{PROCESSED_FOLDER_PATH}/{file_name}'
    result_df_list = dict()
    processed_files = get_csv_files_from_dir(PROCESSED_FOLDER_PATH, lambda fn: all(str(f_n_p) in fn for f_n_p in file_name_pattern))
    filtered_pairs = PROCUCE_PAIRS_BY_TRADE_SYMBOLS(only_pairs) if only_pairs is not None else PAIRS()
    for processed_file in [processed_file for processed_file in processed_files if processed_file.split('-')[0] in [PROCESS_SYMBOL(pair[SYMBOL_TRADING_KEY]) for pair in filtered_pairs]]:
        file_path = processed_data_file_path(processed_file)
        file_path_real = case_insensitive_path(file_path)

        printmd_high(file_path_real)
        df = pd.read_csv(file_path_real, nrows=10)
        float_cols = [c for c in df if df[c].dtype == "float64"]
        float_cols = {c: np.float32 for c in float_cols}
        processed_df = pd.read_csv(
            file_path_real,
            parse_dates=['start_ts', 'end_ts'],
            dtype=float_cols,
            infer_datetime_format=True)

        if filter_predicate is not None:
            symbol = processed_file.split('-')[0]
            processed_df = filter_predicate(processed_df, symbol)

        if permute:
            permuted_processed_df = processed_df.sample(frac=1).reset_index(drop=True)
            result_df_list[processed_file] = permuted_processed_df
            df_to_display = permuted_processed_df
            log_msg = f"`processed file: {processed_file}, permuted dataframe size: {len(permuted_processed_df)}`"
        else:
            result_df_list[processed_file] = processed_df
            df_to_display = processed_df
            log_msg = f"`processed file: {processed_file}, dataframe size: {len(processed_df)}`"

        printmd_low(log_msg)
        run_high(lambda : print_memory(df=df_to_display), lambda: run_medium(print_memory))

    for key, val in result_df_list.items():
        run_high(lambda: printmd_low(f"{key}:"))
        run_high(lambda: display_low(val))

    concat_permuted_processed_df_s = pd.concat(list(result_df_list.values())).reset_index(drop=True)

    run_high(lambda: print_memory(df=concat_permuted_processed_df_s), lambda: print_memory())

    display_medium(concat_permuted_processed_df_s)

    measure(f'Size: {len(concat_permuted_processed_df_s)}')

    return concat_permuted_processed_df_s


def produce_segments(PAIR, df):
    title = f"**PRODUCING SEGMENTS..**"
    description = f"**`{PAIR}`** `>  dataframe size: {len(df)}, segment length: {SEGMENT_LENGTH()}, features: {FEATURES}`"
    printmd_low(f'{title}\r\n\r\n{description}')
    run_medium(print_memory)

    measure = produce_measure_md_low('PRODUCED SEGMENTS')

    df = df[[*['timestamp', 'utc_timestamp'], *FEATURES]]

    raw_segments = split_df_with_overlap(df, SEGMENT_LENGTH(), SEGMENT_OVERLAP())  # Split dataframe to segments
    filtered_segments = filter_segments_containing_datetime_gaps(raw_segments)

    if len(filtered_segments) > 0:
        begin_segment = filtered_segments[0]
        printmd_high("**First segment:**")
        display_high(begin_segment)

        end_segment = filtered_segments[-1:][0]
        printmd_high("**Last segment:**")
        display_high(end_segment)

    run_medium(print_memory)
    measure(f'`{PAIR}` | RAW SIZE = {len(raw_segments)} | FILTERED SIZE: {len(filtered_segments)}, EXCLUDED SIZE: {len(raw_segments)-len(filtered_segments)}')

    return filtered_segments


def process_segments_save_converted(PAIR, segments, produce_processed_file_name):
    title = f'**PROCESSING SEGMENTS..**'
    description = f"**`{PAIR}`** `>  segments count: {len(segments)}, features: {FEATURES}`"
    printmd_low(f'{title}\r\n\r\n{description}')
    run_medium(print_memory)

    measure = produce_measure_low('SEGMENTS PROCESSED & SAVED')

    if len(segments) == 0:
        measure(f'!! NO PROCESSING {PAIR} SEGMENTS !!')

        return

    for feature in FEATURES:
        processed_file_name = produce_processed_file_name(feature)

        rows = []
        force_process = FORCE_PROCESS
        if force_process or not os.path.isfile(processed_file_name): #Filter already existing data if not force rewrite
            filtered_segments = segments
            force_process = True
        else:
            existing_processed_df = retrieve_processed_data(processed_file_name)
            last_existing_row = existing_processed_df.iloc[-1]
            del existing_processed_df
            filtered_segments = list(filter(lambda _segment: _segment.iloc[0]['utc_timestamp'] > last_existing_row['start_ts'], segments))

        if len(filtered_segments) == 0:
            printmd_medium(f"`**!! NO NEW SEGMENTS {PAIR} | {feature} !!**`")

            continue

        i = 0
        batch_measure = produce_measure_medium(f'BATCH MEASURE:{BATCH_MEASURE_SIZE}, 0')
        for segment in filtered_segments:
            field = f'{feature}'
            row = segment[field].tolist()
            start = segment.iloc[0]
            end = segment.iloc[-1]
            row = [*[start['utc_timestamp'], end['utc_timestamp']], *row]
            rows.append(row)
            if i > 0 and i % 100_000 == 0:
                batch_measure()
                batch_measure = produce_measure_medium(f'BATCH MEASURE:{BATCH_MEASURE_SIZE}, {i}')
            i += 1

        feature_cols = list(map(lambda c: f'{feature}_{c}', segment.reset_index().index.to_list()))
        columns = [*['start_ts', 'end_ts'], *feature_cols]
        processed_df = pd.DataFrame(rows, columns=columns)

        del rows
        if force_process:
            write_csv_lambda = lambda: processed_df.to_csv(processed_file_name, index=False)
        else:
            processed_df = processed_df[processed_df['start_ts'] > last_existing_row['start_ts']]
            write_csv_lambda = lambda: processed_df.to_csv(processed_file_name, mode='a', header=False, index=False)

        format_precision = produce_measure_high('FORMAT PRECISION')
        printmd_high(f'`FORMAT PRECISION....`')
        processed_df = process_format_precision_order_6_df(processed_df)
        format_precision()

        processed_df['start_ts'] = processed_df['start_ts'].astype(str)
        processed_df['end_ts'] = processed_df['end_ts'].astype(str)

        write_measure = produce_measure_high('WRITE CSV')
        printmd_high(f'`WRITE CSV....`')
        write_csv_lambda()
        write_measure()

        has_not_cache = len(filtered_segments) == len(segments)
        force_rewrite_suffix = f"WRITE DATA >" if has_not_cache else "APPEND DATA >"
        data_frame_size = len(processed_df)
        printmd_medium(f"**`{PAIR}`** `>  {force_rewrite_suffix} dataframe size: {data_frame_size}, segment length: {len(segment)}, gradient window: {feature}`")
        run_medium(print_memory)
        display_high(processed_df)

        del processed_df

    run_medium(print_memory)
    measure(description)


def run_prepare_save_train_feature(params):
    try:
        features_meta = {'STATE': 'EMPTY'}

        feature_config = get_item_from_list_dict(FEATURES_CONFIG(), FEATURE_KEY, params['feature_config_key'])
        meta_suffix = params['meta_suffix']
        excluded_cached_data = params['excluded_cached_data']
        excluded_extremums = params['excluded_extremums']
        offset_constraints = params['offset_constraints']

        feature = feature_config[FEATURE_KEY]
        feature_norm = feature_config[FEATURE_NORM_KEY]
        feature_segment_offset = feature_config[SEGMENT_OFFSET_KEY]
        feature_space_producer = feature_config[SPACE_PRODUCER_KEY]
        feature_space_producer_norm = feature_config[SPACE_PRODUCER_NORM_KEY]

        file_path = PRODUCE_TRAIN_FILE_NAME(feature_norm, meta_suffix)

        title = f'PREPARE & SAVE TRAIN DATA ({feature})'
        symbols = TRAIN_SYMBOLS()
        description = f"symbols = {symbols} | feature = {feature} | feature norm = {feature_norm} | segment offset = {feature_segment_offset} | discretization = {DISCRETIZATION()} | classes = {CLASSES()} | file path = {file_path}"
        print_action_title_description__low(title, description)

        pretrain_save_measure = produce_measure_low(title)
        run_medium(print_memory)

        excluded_pairs = sorted(excluded_cached_data['pair'].unique().tolist())
        processed_pairs = sorted([PROCESS_SYMBOL(pair[SYMBOL_TRADING_KEY]) for pair in TRAIN_PAIRS()])
        print(f"TRAIN: {excluded_pairs}")
        print(f"CACHED PROCESSED: {processed_pairs}")

        assert excluded_pairs == processed_pairs, 'WRONG PAIRS MATCHING'

        excluded_feature_cached_start = datetime_Y_m_d__h_m_s(excluded_cached_data['utc_timestamp'].min())
        excluded_feature_cached_end = datetime_Y_m_d__h_m_s(excluded_cached_data['utc_timestamp'].max())
        excluded_feature_cached_max = excluded_cached_data[feature].max()
        excluded_feature_cached_min = excluded_cached_data[feature].min()

        date_constraints = get_exclude_dates_train_configs_present(TRAIN_PAIRS())

        processed_date_filter_predicate = lambda df, pair: df[df['start_ts'] >= get_date_train_constraints(pair)[0]][df['end_ts'] <= get_date_train_constraints(pair)[1]]
        excluded_dates__concat_permuted_processed_df = retrieve_concat_permute_processed_data([DISCRETIZATION(), feature], permute=False, filter_predicate=processed_date_filter_predicate, only_pairs=TRAIN_SYMBOLS())

        exclude_dates_processed_measure = measure_print_action_title_description__low(f"EXCLUDING DATES PROCESSED DATAFRAME", f"feature = {feature} | size = {len(excluded_cached_data)} | date_constraints = {date_constraints}")

        feature_cols = get_sorted_features_list_from_df(excluded_dates__concat_permuted_processed_df)
        feature_norm_cols = [col.replace(feature, feature_norm) for col in feature_cols]
        excluded_dates__features_cols__concat_permuted_processed_df = excluded_dates__concat_permuted_processed_df[[*PROCESSED_DATE_COLS, *feature_cols]]

        exclude_dates_processed_measure()

        run_medium(print_memory)

        exclude_extremums_processed_measure = measure_print_action_title_description__low(f"EXCLUDING EXTREMUMS PROCESSED DATAFRAME", f"feature = {feature}, feature MAX = {excluded_feature_cached_max}, feature MIN = {excluded_feature_cached_min}")

        feature_offset_constraint = offset_constraints[feature]
        excluded_dates_extremums__features_cols__concat_permuted_processed_df = excluded_dates__features_cols__concat_permuted_processed_df[(excluded_dates__features_cols__concat_permuted_processed_df[feature_cols] <= feature_offset_constraint).all(axis=1) & (excluded_dates__features_cols__concat_permuted_processed_df[feature_cols] >= -feature_offset_constraint).all(axis=1)]

        excluded_features_processed_start = datetime_Y_m_d__h_m_s(excluded_dates_extremums__features_cols__concat_permuted_processed_df['start_ts'].min())
        excluded_features_processed_end = datetime_Y_m_d__h_m_s(excluded_dates_extremums__features_cols__concat_permuted_processed_df['end_ts'].max())
        excluded_features_processed_max = excluded_dates_extremums__features_cols__concat_permuted_processed_df[feature_cols].max().max()
        excluded_features_processed_min = excluded_dates_extremums__features_cols__concat_permuted_processed_df[feature_cols].min().min()

        exclude_extremums_processed_measure()

        print_action_title_description__low(f"EXCLUDED CACHED DATES & EXTREMUMS ({feature})", f"feature = {feature} | START = {excluded_feature_cached_start} | END = {excluded_feature_cached_end} ||  MAX = {excluded_feature_cached_max} | MIN = {excluded_feature_cached_min}")
        print_action_title_description__low(f"EXCLUDED PROCESSED DATES & EXTREMUMS ({feature})", f"feature = {feature} | START = {excluded_features_processed_start} | END = {excluded_features_processed_end} ||  MAX = {excluded_features_processed_max} | MIN = {excluded_features_processed_min}")

        # assert are_close(excluded_feature_cached_max, excluded_features_processed_max, FEATURE_RATIO_PERCENT_TOLERANCE), f"{excluded_feature_cached_max} != {excluded_features_processed_max}"
        # assert are_close(excluded_feature_cached_min, excluded_features_processed_min, FEATURE_RATIO_PERCENT_TOLERANCE), f"{excluded_feature_cached_min} != {excluded_features_processed_min}"

        run_medium(print_memory)

        normalize_title = f"NORMALIZE EXCLUDED PROCESSED DATAFRAME"
        print_action_title_description__low(normalize_title, f"feature={feature}, size={len(excluded_dates_extremums__features_cols__concat_permuted_processed_df)}")
        normalize_measure = produce_measure_low(normalize_title)

        _normalizer = lambda val: normalize_minus_1__plus_1(val, outer_max=excluded_features_processed_max, outer_min=excluded_features_processed_min)
        concat_permuted_processed_normalized_df = excluded_dates_extremums__features_cols__concat_permuted_processed_df
        del excluded_dates_extremums__features_cols__concat_permuted_processed_df
        concat_permuted_processed_normalized_df[feature_norm_cols] = concat_permuted_processed_normalized_df[feature_cols].apply(_normalizer)#.rename(columns=normalized_cols_rename_mapping, errors="raise")
        concat_permuted_processed_normalized_df = concat_permuted_processed_normalized_df.drop(feature_cols, axis=1)

        normalize_measure()

        # Get extremum values of features for NORMALIZED PROCESSED data
        feature_processed_norm_max = concat_permuted_processed_normalized_df[feature_norm_cols].max().max()
        feature_processed_norm_min = concat_permuted_processed_normalized_df[feature_norm_cols].min().min()

        printmd_medium(f"`excluded_feature_norm_max: {feature_processed_norm_max}`")
        printmd_medium(f"`excluded_feature_norm_min: {feature_processed_norm_min}`")

        # Ensure normalized feature values between -1 and 1
        assert feature_processed_norm_max == 1 or feature_processed_norm_min == -1, f"!!WRONG NORMALIZATION!! > feature_processed_norm_max: {feature_processed_norm_max}, feature_processed_norm_min: {feature_processed_norm_min}"

        run_medium(print_memory)

        # Get one hot from class producer
        one_hot_from_class_factory = get_one_hot_from_class_producer(CLASSES())

        last_item = f'{feature_norm}_{SEGMENT_LENGTH() - feature_segment_offset}'
        last_item_class = f"{last_item}_class"

        space = feature_space_producer(excluded_extremums)
        bins_norm, counts_norm, weights_norm, weights_count_product_norm, space_norm = ___calc_space_bins(excluded_cached_data[feature_norm].dropna().to_list(), feature_space_producer_norm(excluded_extremums))
        bins_pairwise = list(pairwise(list(bins_norm)))
        one_hot_cols = get_one_hot_cols(CLASSES())

        classes_one_hots_title = f"CALCULATE CLASSES AND ONE-HOTS"
        print_action_title_description__low(classes_one_hots_title, f"feature={feature}")
        classes_one_hots_title_measure = produce_measure_low(classes_one_hots_title)

        concat_permuted_processed_normalized_df[last_item_class] = concat_permuted_processed_normalized_df[last_item].apply(lambda bin: int(get_class_from_bin(bin, bins_pairwise)))
        concat_permuted_processed_normalized_df[one_hot_cols] = concat_permuted_processed_normalized_df[last_item_class].apply(lambda clazz: pd.Series(one_hot_from_class_factory(clazz)))

        classes_one_hots_title_measure()

        display_medium(concat_permuted_processed_normalized_df.head())

        run_medium(print_memory)

        input_cols = feature_norm_cols[:-feature_segment_offset]
        output_cols = one_hot_cols

        display_medium(input_cols)
        display_medium(output_cols)

        input_df = concat_permuted_processed_normalized_df[input_cols]
        output_cl_df = concat_permuted_processed_normalized_df[last_item_class]
        output_oh_df = concat_permuted_processed_normalized_df[output_cols]

        display_high(input_df)
        display_high(output_cl_df)
        display_high(output_oh_df)

        saving_cols = [*PROCESSED_DATE_COLS, *input_cols, *output_cols]
        saving_df = concat_permuted_processed_normalized_df[saving_cols]

        display_low(saving_df.head())
        saving_df.to_csv(file_path)

        run_high(lambda: print_memory(df=saving_df), lambda: run_medium(print_memory))
        pretrain_save_measure()

        time.sleep(random.randint(1, 9))

        features_meta = read_train_meta(meta_suffix)[FEATURE_S_KEY]
        features_meta[feature_norm] = {
            SPACE_KEY: space,
            BINS_KEY: bins_norm,
            BINS_PAIRWISE_KEY: bins_pairwise,
            COUNTS_KEY: counts_norm,
            WEIGHTS_KEY: weights_norm,
            WEIGHTS_COUNT_PRODUCT_NORM_KEY: weights_count_product_norm,
            SPACE_PRODUCER_SERIALIZED_KEY: inspect.getsource(feature_space_producer_norm)
        }

        wrire_train_meta(dict(OFFSETS=offset_constraints, EXTREMUMS=excluded_extremums, FEATURES=features_meta), meta_suffix)
    except:
        traceback.print_exc()
        display(features_meta)


def prepare_save_train_data(excluded_getter, meta_suffix=None):
    try:
        features_config = FEATURES_CONFIG()

        title = f'PREPARE & SAVE TRAIN DATA (ALL FEATURES)'
        symbols = TRAIN_SYMBOLS()
        description = f"symbols = {symbols} | features = {' | '.join([feature_config[FEATURE_KEY] for feature_config in features_config])} || discretization = {DISCRETIZATION()} | classes = {CLASSES()}"
        prepare_save_measure = measure_print_action_title_description__low(title, description)

        excluded_cached_data, offset_constraints, excluded_extremums = excluded_getter()
        wrire_train_meta(dict(OFFSETS=offset_constraints, EXTREMUMS=excluded_extremums, FEATURES={}), meta_suffix)
        params = [{'meta_suffix': meta_suffix, 'feature_config_key': config[FEATURE_KEY], 'excluded_cached_data': excluded_cached_data, 'offset_constraints': offset_constraints, 'excluded_extremums': excluded_extremums} for config in features_config]
        run_multi_process(run_prepare_save_train_feature, params)

        prepare_save_measure()
    except:
        train_meta = read_train_meta()
        print(train_meta)
        if len(train_meta[FEATURE_S_KEY]) == 0:
            remove_train_meta()
            prepare_save_measure('ERROR >> REMOVE TRAIN META')
        else:
            prepare_save_measure('ERROR >> PARTIALLY SAVED TRAIN META')


def retrieve_train_data(target_feature_norm, meta_suffix, take_ratio=1):
    file_path = PRODUCE_TRAIN_FILE_NAME(target_feature_norm, meta_suffix)
    file_path_real = case_insensitive_path(file_path)

    title = f'RETRIEVE TRAIN DATA'
    description = f"interval = {DISCRETIZATION()} | classes = {CLASSES()} | file path = {file_path_real} | take ratio = {take_ratio}"
    print_action_title_description(title, description)

    measure = produce_measure_low('RETRIEVED TRAIN DATA')
    run_medium(print_memory)

    concat_permuted_processed_norm_df = pd.read_csv(file_path_real)
    original_df_size = len(concat_permuted_processed_norm_df)
    take = int(len(concat_permuted_processed_norm_df) * take_ratio)
    concat_permuted_processed_norm_df = concat_permuted_processed_norm_df.sample(frac=1).head(take)

    input_cols = list(filter(lambda col: f'{target_feature_norm}' in col, concat_permuted_processed_norm_df.columns.to_list()))
    one_hot_cols = list(filter(lambda col: f'oh_' in col, concat_permuted_processed_norm_df.columns.to_list()))

    input_s = concat_permuted_processed_norm_df[input_cols].values
    output_oh_s = concat_permuted_processed_norm_df[one_hot_cols].values

    run_medium(print_memory)
    measure(f"taken df size = {len(concat_permuted_processed_norm_df)} | original df size = {original_df_size}")

    return input_s, output_oh_s


def display__feature_diff__vs__feature_grad__dependency(concat_cached_data_df, feature_x, feature_y, feature_x_space_producer=None, feature_y_space_producer=None):
    plot_series_dependency(
        concat_cached_data_df,
        feature_x,
        feature_y,
        xaxis_ticks=feature_x_space_producer(feature_x) if feature_x_space_producer is not None else None,
        yaxis_ticks=feature_y_space_producer(feature_y) if feature_y_space_producer is not None else None,
        is_permuted=True,
    )


def normalize_df(cached_df, features_config):
    symbols = cached_df['pair'].unique()
    normalize_configs_present = get_normalize_configs_present()
    title = f'NORMALIZE CACHE DATA'
    description = f"pairs = {symbols} | data frame size = {len(cached_df)} | {normalize_configs_present}"
    print_action_title_description__low(title, description)

    measure = produce_measure_low('NORMALIZED CACHED DATA')

    normalized_cached_df = cached_df
    for feature_norm_map in features_config:
        feature = feature_norm_map[FEATURE_KEY]
        feature_norm = feature_norm_map[FEATURE_NORM_KEY]
        normalized_cached_df = normalize_cached_df(normalized_cached_df, feature, feature_norm)

    measure()

    return normalized_cached_df


def normalize_cached_df(concat_cached_data_df, feature, feature_norm):
    symbols = concat_cached_data_df['pair'].unique()
    title = f'NORMALIZE CACHED DATA'
    description = f"pairs = {symbols} | data frame size = {len(concat_cached_data_df)} | {feature} > {feature_norm}"
    print_action_title_description__medium(title, description)

    feature_max = concat_cached_data_df[feature].max()
    feature_min = concat_cached_data_df[feature].min()

    printmd_medium(f"`{feature}_max: {feature_max}, {feature}_min: {feature_min}`")

    concat_cached_data_df[feature_norm] = concat_cached_data_df[feature].apply(lambda f: normalize_minus_1__plus_1(f, outer_max=feature_max, outer_min=feature_min))

    feature_norm_max = concat_cached_data_df[feature_norm].max()
    feature_norm_min = concat_cached_data_df[feature_norm].min()

    printmd_medium(f"`{feature_norm}_max: {feature_norm_max}, {feature_norm}_min: {feature_norm_min}`")

    return concat_cached_data_df


def plot__feature_diff__vs__feature__distribution(df, feature, feature_space_producer):
    space = feature_space_producer(feature)
    space_present = ['{:.5f}'.format(x) for x in space]
    data = df[feature].to_list()

    feature_max = df[feature].max()
    feature_min = df[feature].min()

    print_action_title_description(f"Distribution {feature}", f"size={len(data)} | {feature}_max={feature_max}, {feature}_min={feature_min} | space={space_present}")

    __plot_feature_distribution(data, space, feature, f"data frame size={len(data)}")


def calculate_features_extremums(df, feature_list, offset_constraints):
    print_action_title_description(f"EXTREMUMS >>>>>>>", f"features={feature_list}")

    extremums = dict()
    for feature in feature_list:
        try:
            if not feature.lower().endswith('norm'):
                print(f"{feature}_offset = +-{offset_constraints[feature]}")

            feature_max = df[feature].max()
            feature_min = df[feature].min()

            extremums[f"{feature}_max"] = feature_max
            extremums[f"{feature}_min"] = feature_min

            print(f"{feature}_max = {process_format_precision_order(feature_max)} | {feature}_min = {process_format_precision_order(feature_min)}")

            if feature.lower().endswith('norm'):
                print('------------------------------------------------------------------')
        except KeyError:
            print(f'!!!NO {feature} yet in data frame!!!')

    print_action_title_description(f"<<<<<<< EXTREMUMS")

    return extremums


#!!PLOT PURPOSES ONLY!! Produces optimized for dependecy plotting dataframe
def produce_optimized_bin_weighted_df(df, feature, space_producer, _multiplier=np.sqrt):
    df = df.dropna()
    space = space_producer(feature)
    bins, counts, weights, weights_count_product_rom, symmetric_space = ___calc_space_bins(df[feature].to_list(), space)
    bin_count_multiplier = _multiplier(weights / np.nanmax([weight for weight in weights if weight != np.inf]))
    print_high(f'bins={bins}')
    print_high(f'counts={counts}')
    print_high(f'weights={weights}')
    print_high(f'weights_count_product_rom={weights_count_product_rom}')
    print_high(f'symmetric_space={symmetric_space}')
    print_high(f'bin_count_multiplier={bin_count_multiplier}')

    i = 0
    df_s = []
    for start, end in pairwise(space):
        df_subset = df
        df_subset = df_subset[df_subset[feature] >= start]
        df_subset = df_subset[df_subset[feature] <= end]
        if len(df_subset) == 0:
            i += 1
            continue

        # df_subset = df_subset.head(int(len(df_subset) * bin_count_multiplier[i]))
        df_subset = df_subset.head(100)
        df_s.append(df_subset)
        i += 1

    optimized_bin_weighted_df = pd.concat(df_s, ignore_index=True)

    return optimized_bin_weighted_df


def exclude__extremums__dates__cached_data(initial__cached__pair_grouped__df_dict, features_configs, exclude_data=False):
    from SRC.CORE.data_utils import calculate_features_extremums
    from SRC.CORE._CONSTANTS import FEATURE_DIFF, FEATURE_DIFF_NORM, FEATURE_2, FEATURE_2_NORM, FEATURE_3, FEATURE_3_NORM

    initial_count = sum([len(df) for df in list(initial__cached__pair_grouped__df_dict.values())])

    features = [FEATURE_DIFF, FEATURE_DIFF_NORM, FEATURE_2, FEATURE_2_NORM, FEATURE_3, FEATURE_3_NORM]

    original_nan_count_s = []
    extremums_excluded_count_s = []
    dates_excluded_count_s = []

    normalized__extremums_dates_excluded__cached__pair_grouped__df_dict = dict()
    for file_name, pair_grouped_df in initial__cached__pair_grouped__df_dict.items():
        extremums_excluded_pair_grouped_df = pair_grouped_df
        original_nan_count = pair_grouped_df.isnull().any(axis=1).sum()
        original_nan_count_s.append(original_nan_count)
        if exclude_data:
            symbol = file_name.split('-')[0]
            start_date_constraint = get_date_train_constraints(symbol)

            print_action_title_description(f"EXCLUDE EXTREMUMS", f"pair = {symbol} | {get_exclude_extremums_configs_present()}")

            for feature_config in [config for config in features_configs if CUT_OFFSET_KEY in config]:
                feature = feature_config[FEATURE_KEY]
                extremum = feature_config[CUT_OFFSET_KEY]

                def conditions(x):
                    if x < -extremum:
                        return np.NaN
                    elif x > extremum:
                        return np.NaN
                    else:
                        return x

                func = np.vectorize(conditions)
                extremums_excluded_pair_grouped_df[feature] = func(extremums_excluded_pair_grouped_df[feature])

            extremums_excluded_count_s.append(len(extremums_excluded_pair_grouped_df) - len(extremums_excluded_pair_grouped_df.dropna()) - original_nan_count)

            print_action_title_description(f"EXCLUDE DATES", f"pair = {symbol} | start date = {start_date_constraint[0]} | end date = {start_date_constraint[1]}")

            cached_date_filter_predicate = lambda df, pair: df[df['utc_timestamp'] >= start_date_constraint[0]][df['utc_timestamp'] <= start_date_constraint[1]]
            dates_extremums_excluded_pair_grouped_df = cached_date_filter_predicate(extremums_excluded_pair_grouped_df, symbol)

            dates_excluded_count_s.append(len(extremums_excluded_pair_grouped_df.dropna()) - len(dates_extremums_excluded_pair_grouped_df.dropna()))

            normalized_extremums_excluded_pair_grouped_df = normalize_df(dates_extremums_excluded_pair_grouped_df, features_configs)
            normalized__extremums_dates_excluded__cached__pair_grouped__df_dict[file_name] = normalized_extremums_excluded_pair_grouped_df

            offset_constraints = {feature_config[FEATURE_KEY]: feature_config[CUT_OFFSET_KEY] for feature_config in FEATURES_CONFIG()}
        else:
            normalized_extremums_excluded_pair_grouped_df = normalize_df(extremums_excluded_pair_grouped_df, features_configs)
            normalized__extremums_dates_excluded__cached__pair_grouped__df_dict[file_name] = normalized_extremums_excluded_pair_grouped_df
            offset_constraints = {feature_config[FEATURE_KEY]: 1 for feature_config in FEATURES_CONFIG()}

    normalized__extremums_dates_excluded__concatenated_df = pd.concat(list(normalized__extremums_dates_excluded__cached__pair_grouped__df_dict.values())).reset_index(drop=True)
    extremums = calculate_features_extremums(normalized__extremums_dates_excluded__concatenated_df, features, offset_constraints)

    remaining_count = len(normalized__extremums_dates_excluded__concatenated_df.dropna()) + sum(original_nan_count_s)
    removed = initial_count - remaining_count
    print(f"REMOVED (COUNT) = {removed}, REMOVED (%) = {removed / initial_count * 100} || EXTREMUMS EXCLUDED COUNT = {sum(extremums_excluded_count_s)} | DATES EXCLUDED COUNT = {sum(dates_excluded_count_s)} ")

    return offset_constraints, extremums, normalized__extremums_dates_excluded__cached__pair_grouped__df_dict, normalized__extremums_dates_excluded__concatenated_df


def validate_cached_df():
    for pair in PAIRS():
        symbol = PROCESS_SYMBOL(pair[SYMBOL_TRADING_KEY])
        start_from = pair[START_TRAIN_KEY]
        file_path = PRODUCE_CACHED_FILE_NAME(symbol)
        file_path_real = case_insensitive_path(file_path)
        df = pd.read_csv(file_path_real)

        validate_timeseries_df(df, 'utc_timestamp')


def validate_processed_df():
    for pair in PAIRS():
        symbol = PROCESS_SYMBOL(pair[SYMBOL_TRADING_KEY])
        file_producer = PRODUCE_PROCESSED_FILE_NAME(symbol)
        for feature_conf in FEATURES_CONFIG():
            feature = feature_conf[FEATURE_KEY]
            file_path = file_producer(feature)
            file_path_real = case_insensitive_path(file_path)
            df = pd.read_csv(file_path_real)

            validate_timeseries_df(df, 'start_ts')


def boost_weights(weights, _lambda, custom_weights=None):
    from matplotlib import pyplot as plt
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(x):
        return _lambda(sigmoid, x)

    def shifted_sigmoid_derivate(x):
        return sigmoid_derivative(x - int((len(x)-1) / 2))

    original_weights_title = 'Original weights'
    boosted_weights_title = 'Boosted weights'
    custom_weights_title = 'Custom weights'
    multipliers_title = 'Multipliers'
    multipliers_normalized_title = 'Multipliers normalized'

    x_s = np.linspace(0, len(weights)-1, len(weights))
    weights_multipliers = shifted_sigmoid_derivate(x_s)
    boosted_weights = [a * b * (1 / max(weights_multipliers)) for a, b in zip(weights, weights_multipliers)]
    plt.plot(range(len(weights)), weights, label=original_weights_title)
    plt.plot(range(len(boosted_weights)), boosted_weights, label=boosted_weights_title)
    plt.plot(x_s, weights_multipliers, label=multipliers_title)
    if custom_weights is not None:
        plt.plot(range(len(custom_weights)), custom_weights, label=custom_weights_title)
    plt.yscale('log')
    plt.legend()
    plt.title(f"{original_weights_title} vs. {boosted_weights_title} + {multipliers_title}")

    weights_multipliers_normalized = ((1 / max(weights_multipliers)) * np.array(weights_multipliers)).tolist()
    weights_multipliers = list(weights_multipliers)

    run_medium(lambda: printmd(f'{original_weights_title}: **{[_float_6(weight) for weight in weights]}**'))
    run_medium(lambda: printmd(f'{boosted_weights_title}: **{[_float_6(boosted_weight) for boosted_weight in boosted_weights]}**'))
    run_medium(lambda: printmd(f'{multipliers_title}: **{[_float_6(weight_multiplier) for weight_multiplier in weights_multipliers]}**'))
    run_medium(lambda: printmd(f'{multipliers_normalized_title}: **{[_float_6(weight_multiplier_normalized) for weight_multiplier_normalized in weights_multipliers_normalized]}**'))
    
    return boosted_weights, weights_multipliers, weights_multipliers_normalized


def retrieve_simulate_data(symbol, start_simulation_from, end_simulation_to):
    try:
        cached_df = retrieve_cached_data(symbol, PRODUCE_CACHED_FILE_NAME(symbol))

        date_filtered_cached_df = cached_df[cached_df['kiev_timestamp'] >= start_simulation_from][cached_df['kiev_timestamp'] <= end_simulation_to]

        featurized_date_filtered_cached_df = date_filtered_cached_df
        featurized_date_filtered_cached_df['close_time'] = featurized_date_filtered_cached_df['utc_timestamp'].apply(lambda utc_ts: utc_ts.astimezone(TZ()))
        featurized_date_filtered_cached_df[['close_time', 'open', 'high', 'low', 'close']] = featurized_date_filtered_cached_df[['close_time', 'open', 'high', 'low', 'close']].astype(
            {'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float'})

        buffer = featurized_date_filtered_cached_df.to_dict(orient='records')
        featurized_date_filtered_cached_df = featurize_candles_mins_dashboard(list(buffer), build_presentation_coordinates=False)

        # display(featurized_date_filtered_cached_df.head())
        # display(featurized_date_filtered_cached_df.tail())

        nonan_featurized_date_filtered_cached_df = featurized_date_filtered_cached_df[ROLLING_GRAD_WINDOW_FREQs[-1]:-ROLLING_GRAD_WINDOW_FREQs[-1]]
        # assert not has_nan_df(nonan_featurized_date_filtered_cached_df)
        # assert not has_gap_df(nonan_featurized_date_filtered_cached_df)
        # validate_timeseries_df(nonan_featurized_date_filtered_cached_df, 'utc_timestamp')

        display(nonan_featurized_date_filtered_cached_df.head())
        display(nonan_featurized_date_filtered_cached_df.tail())
    except Exception as ex:
        # run_on_ui_loop(lambda: print(f'retrieve_simulate_data\r\nException: {ex}'))
        print(f'retrieve_simulate_data\r\nException: {ex}')
        raise ex

    return nonan_featurized_date_filtered_cached_df


def retrieve_simulate_data_closure(symbol):
    df = None

    def retrieve_data():
        nonlocal df

        if df is not None:
            return df

        from SRC.CORE._CONSTANTS import SIMULATION__START_DATE, SIMULATION__TAKE_DELTA, SYMBOL_TRADING_KEY
        from SRC.CORE._FUNCTIONS import PAIRS
        from SRC.CORE.utils import get_item_from_list_dict

        start_simulation_from = SIMULATION__START_DATE()
        try:
            from SRC.CORE._CONSTANTS import SIMULATION__END_DATE
            end_simulation_to = SIMULATION__END_DATE()
        except:
            end_simulation_to = start_simulation_from + SIMULATION__TAKE_DELTA()

        pair = get_item_from_list_dict(PAIRS(), SYMBOL_TRADING_KEY, symbol)
        seg_df = retrieve_simulate_data(PROCESS_SYMBOL(pair[SYMBOL_TRADING_KEY]), start_simulation_from, end_simulation_to)

        df = seg_df

        return df

    return retrieve_data


WEIGHTS_MULTIPLIER = lambda weights, custom_weights=None: boost_weights(weights, lambda _s, x: _s(x * WEIGHTS_BOOSTER_COEF()) * (1 - _s(x * WEIGHTS_BOOSTER_COEF())), custom_weights)
