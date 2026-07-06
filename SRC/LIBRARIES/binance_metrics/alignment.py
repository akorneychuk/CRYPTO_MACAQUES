import pandas as pd
from .constants import BINANCE_ZIP_TF_MINUTES, CREATE_TIME, SYMBOL
import SRC.LIBRARIES.new_utils as nu


def attach_binance_metrics(tf: str, df_counter: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
    tf_number = nu.get_tf_number(tf)
    tf_symbol = nu.get_tf_symbol(tf)
    minutes_per_candle = tf_number

    if tf_symbol != "M" or minutes_per_candle < BINANCE_ZIP_TF_MINUTES or minutes_per_candle % BINANCE_ZIP_TF_MINUTES != 0:
        raise RuntimeError(f"Can't use Binance metrics with {tf} TF")

    num_intervals = minutes_per_candle // BINANCE_ZIP_TF_MINUTES

    df_counter = df_counter.copy()
    metrics_df = metrics_df.copy()

    if metrics_df[CREATE_TIME].duplicated().any():
        raise RuntimeError("Duplicate Binance metrics timestamps")

    metrics_df = metrics_df.set_index(CREATE_TIME)

    metric_columns = [
        column
        for column in metrics_df.columns
        if column != SYMBOL
    ]

    for column in metric_columns:
        series = metrics_df[column]

        for i in range(num_intervals):
            offset_minutes = (i + 1) * BINANCE_ZIP_TF_MINUTES
            df_counter[f"{column}_m{offset_minutes}"] = series.reindex(df_counter.index + pd.Timedelta(minutes=offset_minutes)).to_numpy()

    return df_counter
