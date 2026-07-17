import requests
from SRC.LIBRARIES.binance_storage import *


def get_binance_zip_files(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    data_dir: Path,
    base_url: str,
    url_suffix: str
) -> list[Path]:
    missing_dates = find_binance_missing_dates(symbol=symbol, start_date=start_date, end_date=end_date, data_dir=data_dir)

    if missing_dates:
        failed = _download_binance_zip_files(symbol=symbol, dates=missing_dates, data_dir=data_dir, base_url=base_url, url_suffix=url_suffix)

        if failed:
            raise RuntimeError(
                f"Failed to download {len(failed)} archive(s): "
                f"{', '.join(d.strftime('%Y-%m-%d') for d in failed)}"
            )

    return get_list_binance_zip_files_between(symbol=symbol, start_date=start_date, end_date=end_date, data_dir=data_dir)


def _download_binance_zip_files(symbol: str, dates: list[datetime], data_dir: Path, base_url: str, url_suffix: str) -> list[datetime]:
    failed_dates = []

    for date in sorted(dates):
        try:
            _download_binance_zip(symbol=symbol, date=date, data_dir=data_dir, base_url=base_url, url_suffix=url_suffix)
        except Exception:
            failed_dates.append(date)

    return failed_dates


def _download_binance_zip(symbol: str, date: datetime, data_dir: Path, base_url: str, url_suffix: str, timeout: int = 30) -> None:
    response = requests.get(f"{base_url}/{symbol.upper()}/{url_suffix}-{date:%Y-%m-%d}.zip", timeout=timeout)
    response.raise_for_status()
    get_binance_zip_path(symbol, date, data_dir).write_bytes(response.content)
