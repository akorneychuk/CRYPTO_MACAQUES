from datetime import datetime, timedelta
from zipfile import ZipFile
import pandas as pd
from SRC.LIBRARIES.binance_downloader import get_binance_zip_files
from .constants import *


def load_binance_klines_spot_for_trade(
    symbol: str,
    trade_entry_time: datetime,
    history_minutes: int
) -> pd.DataFrame:
    window_start_time = trade_entry_time - timedelta(minutes=history_minutes)
    window_end_time = trade_entry_time - timedelta(minutes=1)
    zip_files = get_binance_zip_files(
        symbol=symbol,
        start_date=window_start_time,
        end_date=window_end_time,
        data_dir=DATA_DIR,
        base_url=BASE_URL,
        url_suffix=_get_url_suffix(symbol)
    )

    if not zip_files:
        raise RuntimeError("No zip files found")

    frames = []

    for zip_path in zip_files:
        frames.append(_load_binance_klines_zip_dataframe(zip_path))

    df = pd.concat(frames)

    if not df.index.is_monotonic_increasing:
        raise RuntimeError("Combined dataframe is not sorted")

    if not df.index.is_unique:
        raise RuntimeError("Combined dataframe contains duplicate timestamps")

    df_window = df.loc[window_start_time:window_end_time]
    expected_index = pd.date_range(start=window_start_time, end=window_end_time, freq="1min")

    if len(df_window) != len(expected_index):
        raise RuntimeError(f"Window length mismatch: got {len(df_window)}, expected {len(expected_index)}")

    if not df_window.index.equals(expected_index):
        raise RuntimeError("Window index does not match expected index")

    return df_window


def _get_url_suffix(symbol: str) -> str:
    return f'1m/{symbol.upper()}-1m'


def _load_binance_klines_zip_dataframe(zip_path: Path) -> pd.DataFrame:
    with ZipFile(zip_path) as archive:
        with archive.open(archive.namelist()[0]) as file:
            df = pd.read_csv(
                file,
                header=None,
                names=KLINES_COLUMNS,
                dtype=KLINES_DTYPES,
            )

    # ---------- Определяем единицу времени ----------

    first_timestamp = df["open_time"].iat[0]

    if first_timestamp >= 10 ** 15:
        time_unit = "us"
    elif 10 ** 12 <= first_timestamp < 10 ** 15:
        time_unit = "ms"
    else:
        raise RuntimeError(f"Unknown timestamp format in '{zip_path.name}': {first_timestamp}")

    # ---------- DatetimeIndex ----------

    df.index = pd.to_datetime(df["open_time"], unit=time_unit, utc=True)
    df.index = df.index.tz_convert("Etc/GMT")
    df.index.name = "datetime"

    return df
