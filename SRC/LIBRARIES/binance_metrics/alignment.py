import re
import pandas as pd
from .constants import BINANCE_ZIP_TF_MINUTES, CREATE_TIME, SYMBOL


def attach_binance_metrics(tf: str, df_counter: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
    tf_number = int(re.search(r"\d+", tf).group())
    tf_symbol = re.sub(r"\d+", "", tf)

    if tf_symbol != "M" or tf_number < BINANCE_ZIP_TF_MINUTES or tf_number % BINANCE_ZIP_TF_MINUTES != 0:
        raise RuntimeError(f"Can't use Binance metrics with {tf} TF")

    num_intervals = tf_number // BINANCE_ZIP_TF_MINUTES

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
            minutes = (i + 1) * BINANCE_ZIP_TF_MINUTES
            df_counter[f"{column}_m{minutes}"] = series.reindex(df_counter.index + pd.Timedelta(minutes=minutes)).to_numpy()

    return df_counter
