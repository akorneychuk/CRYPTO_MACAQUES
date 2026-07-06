from datetime import datetime, timedelta
from pathlib import Path
from .constants import DATA_DIR


def get_binance_metrics_zip_path(symbol: str, date: datetime) -> Path:
    directory = DATA_DIR / "archives" / symbol.upper()
    directory.mkdir(parents=True, exist_ok=True)

    return directory / f"{date:%Y-%m-%d}.zip"


def find_binance_metrics_missing_dates(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
) -> list[datetime]:
    missing = []
    current = start_date

    while current.date() <= end_date.date():
        if not get_binance_metrics_zip_path(symbol, current).exists():
            missing.append(current)

        current += timedelta(days=1)

    return missing


def list_binance_metrics_zip_files_between(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
) -> list[Path]:
    files = []
    current = start_date

    while current.date() <= end_date.date():
        path = get_binance_metrics_zip_path(symbol, current)

        if path.exists():
            files.append(path)

        current += timedelta(days=1)

    return files
