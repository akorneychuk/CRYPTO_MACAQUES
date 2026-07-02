import re
import pandas as pd
from .constants import CREATE_TIME, BINANCE_ZIP_TF_MINUTES


def attach_binance_metrics(tf: str, df_counter: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
    tf_number = int(re.search(r"\d+", tf).group())
    tf_symbol = re.sub(r"\d+", "", tf)

    if tf_symbol == 'M' and tf_number >= BINANCE_ZIP_TF_MINUTES and tf_number % BINANCE_ZIP_TF_MINUTES == 0:
        num_intervals = tf_number // BINANCE_ZIP_TF_MINUTES
    else:
        raise RuntimeError(f'Can`t use Binance metrics with {tf} TF')

    df_counter = df_counter.copy()
    metrics_df = metrics_df.copy()

    if metrics_df[CREATE_TIME].duplicated().any():
        raise RuntimeError(f"Duplicate Binance metrics timestamps")

    metrics_df = metrics_df.set_index(CREATE_TIME, drop=False)

    metric_columns = [
        column
        for column in metrics_df.columns
        if column not in (CREATE_TIME, "symbol")
    ]

    for candle_time in df_counter.index:
        for column in metric_columns:
            for i in range(num_intervals):
                column_number = (i + 1) * BINANCE_ZIP_TF_MINUTES
                column_name = f"{column}_m{column_number}"
                target_time = candle_time + pd.Timedelta(minutes=column_number)
                df_counter.at[candle_time, column_name] = metrics_df[column].get(target_time, pd.NA)

    return df_counter
