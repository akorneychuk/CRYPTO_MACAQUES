from datetime import datetime
from pathlib import Path
from zipfile import ZipFile
import pandas as pd
from .constants import DATA_DIR, BASE_URL, CREATE_TIME, ZIP_COLUMNS
from SRC.LIBRARIES.binance_downloader import get_binance_zip_files


def load_binance_metrics_futures(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    columns: list[str] | None = None
) -> pd.DataFrame:
    if start_date > end_date:
        raise ValueError("start_date must be <= end_date")

    zip_files = get_binance_zip_files(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        data_dir=DATA_DIR,
        base_url=BASE_URL,
        url_suffix=_get_url_suffix(symbol)
    )

    if not zip_files:
        return pd.DataFrame(columns=ZIP_COLUMNS)

    frames = []

    for zip_path in zip_files:
        frames.append(_load_binance_metrics_zip_dataframe(zip_path))

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(CREATE_TIME, ignore_index=True)

    if columns is not None:
        required_columns = [CREATE_TIME]

        for column in columns:
            if column not in required_columns:
                required_columns.append(column)

        unknown = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if unknown:
            raise ValueError(f"Unknown columns: {unknown}")

        df = df[required_columns].copy()

    return df


def _get_url_suffix(symbol: str) -> str:
    return f'{symbol.upper()}-metrics'


def _load_binance_metrics_zip_dataframe(zip_path: Path) -> pd.DataFrame:
    """
    Normalizes historical Binance metrics archives to a single timestamp format.

    Old archives:
        00:00 ... 23:55

    New archives:
        00:05 ... 00:00(next day)

    After normalization every archive follows the new format.
    """
    with ZipFile(zip_path) as archive:
        with archive.open(archive.namelist()[0]) as file:
            df = pd.read_csv(file)

    df[CREATE_TIME] = pd.to_datetime(df[CREATE_TIME], utc=True).dt.floor("min")
    df = _normalize_binance_metrics_archive_format(df)

    first_time = df.iloc[0][CREATE_TIME].strftime("%H:%M:%S")

    if first_time != "00:05:00":
        raise RuntimeError(f"Unexpected first timestamp after normalization: {first_time}: {zip_path}")

    return df


def _normalize_binance_metrics_archive_format(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    first_time = df.iloc[0][CREATE_TIME].strftime("%H:%M:%S")

    if first_time == "00:00:00":
        df[CREATE_TIME] += pd.Timedelta(minutes=5)
    elif first_time == "00:05:00":
        pass
    else:
        raise RuntimeError(
            f"Unknown Binance metrics archive format. "
            f"First timestamp: {df.iloc[0][CREATE_TIME]}"
        )

    return df
