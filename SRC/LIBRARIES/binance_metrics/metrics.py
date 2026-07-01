from datetime import datetime
from pathlib import Path
from zipfile import ZipFile
import pandas as pd
from .constants import (CREATE_TIME, ZIP_COLUMNS)
from .downloader import download_metrics
from .storage import (find_missing_dates, list_zip_files_between)


def load_metrics(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    columns: list[str] | None = None,
    auto_download: bool = True,
) -> pd.DataFrame:
    if start_date > end_date:
        raise ValueError("start_date must be <= end_date")

    if auto_download:
        missing_dates = find_missing_dates(symbol=symbol, start_date=start_date, end_date=end_date)

        if missing_dates:
            failed = download_metrics(symbol=symbol, dates=missing_dates)

            if failed:
                raise RuntimeError(
                    f"Failed to download {len(failed)} archive(s): "
                    f"{', '.join(d.strftime('%Y-%m-%d') for d in failed)}"
                )

    zip_files = list_zip_files_between(symbol=symbol, start_date=start_date, end_date=end_date)

    if not zip_files:
        return pd.DataFrame(columns=ZIP_COLUMNS)

    frames = []

    for zip_path in zip_files:
        frames.append(_load_zip_dataframe(zip_path))

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


def _load_zip_dataframe(zip_path: Path) -> pd.DataFrame:
    with ZipFile(zip_path) as archive:
        csv_name = archive.namelist()[0]

        with archive.open(csv_name) as file:
            df = pd.read_csv(file)

    df[CREATE_TIME] = pd.to_datetime(df[CREATE_TIME], utc=True).dt.floor("min")

    return df
