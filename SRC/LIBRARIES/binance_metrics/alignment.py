import pandas as pd
from .constants import CREATE_TIME


def attach_binance_metrics_for_15m(df_counter: pd.DataFrame,  metrics_df: pd.DataFrame) -> pd.DataFrame:
    df_counter = df_counter.copy()
    metrics_df = metrics_df.copy()
    metrics_df = metrics_df.set_index(CREATE_TIME, drop=False)

    metric_columns = [
        column
        for column in metrics_df.columns
        if column not in (CREATE_TIME, "symbol")
    ]

    for column in metric_columns:
        df_counter[f"{column}_m5"] = pd.NA
        df_counter[f"{column}_m10"] = pd.NA
        df_counter[f"{column}_m15"] = pd.NA

    for candle_time in df_counter.index:
        metrics = metrics_df.loc[
            (metrics_df.index > candle_time)
            &
            (metrics_df.index <= candle_time + pd.Timedelta(minutes=15))
        ]

        if len(metrics) != 3:
            raise RuntimeError(f"{candle_time}: expected 3 metrics rows, found {len(metrics)}")

        for column in metric_columns:
            df_counter.at[candle_time, f"{column}_m5"] = metrics.iloc[0][column]
            df_counter.at[candle_time, f"{column}_m10"] = metrics.iloc[1][column]
            df_counter.at[candle_time, f"{column}_m15"] = metrics.iloc[2][column]

    return df_counter
