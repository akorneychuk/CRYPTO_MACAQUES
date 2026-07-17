from datetime import datetime, timedelta
from pathlib import Path


def get_binance_zip_path(symbol: str, date: datetime, data_dir: Path) -> Path:
    directory = data_dir / "archives" / symbol.upper()
    directory.mkdir(parents=True, exist_ok=True)

    return directory / f"{date:%Y-%m-%d}.zip"


def find_binance_missing_dates(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    data_dir: Path
) -> list[datetime]:
    missing = []
    current = start_date

    while current.date() <= end_date.date():
        if not get_binance_zip_path(symbol, current, data_dir).exists():
            missing.append(current)

        current += timedelta(days=1)

    return missing


def get_list_binance_zip_files_between(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    data_dir: Path
) -> list[Path]:
    files = []
    current = start_date

    while current.date() <= end_date.date():
        path = get_binance_zip_path(symbol, current, data_dir)

        if path.exists():
            files.append(path)

        current += timedelta(days=1)

    return files
