from dateutil import parser

from SRC.CORE.utils import hashablelist
from SRC.LIBRARIES.new_data_utils import fetch_featurize_all, read_group_df, get_group_constraints_df, produce_differential, fecth_klines_retry
from SRC.NN.IModelBase import produce_model


def fetch_data(market_type, symbol, start_dt_str):
    start_dt = parser.parse(start_dt_str)

    discretization_s = ["15M", "30M", "1H", "2H", "4H", "8H"]

    realtime_klines = fecth_klines_retry(market_type, symbol, discretization_s[0])

    # group = fetch_featurize_realtime_group_all(market_type, symbol, discretization_s, segments, print_out=True)
    # realtime_df = group[0]

    df_origin_s = fetch_featurize_all(market_type, symbol, discretization_s, start_dt=start_dt, end_dt=None, print_out=True)
    historical_df = df_origin_s[0]

    group_df = read_group_df(discretization_s=hashablelist(discretization_s))

    return {
        'group_df': group_df,
        'df_origin_s': df_origin_s,
        'historical_df': historical_df
    }


def prepare_data(model_name, df_origin_s, group_df, start_dt_str):
    start_dt = parser.parse(start_dt_str)

    model = produce_model(model_name)
    segments = model.segments_count()
    threshold = model.threshold()

    df_s = [df_origin.copy() for df_origin in df_origin_s]
    df_s[0] = produce_differential(df_s[0], threshold, include_existing_features=True, print_out=True)
    group_constraints_df = get_group_constraints_df(df_s=df_s, group_df=group_df, start_dt=start_dt, end_dt=None)

    return {
        'df_s': df_s,
        'group_constraints_df': group_constraints_df,
        'model': model,
        'segments': segments,
        'threshold': threshold,
    }