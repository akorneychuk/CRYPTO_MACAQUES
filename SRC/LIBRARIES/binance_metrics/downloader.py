import requests
from datetime import datetime
from .constants import BASE_URL
from .storage import get_zip_path


def download_metrics(symbol: str, dates: list[datetime]) -> list[datetime]:
    failed_dates = []

    for date in sorted(dates):
        try:
            _download_metrics_zip(symbol=symbol, date=date)
        except Exception:
            failed_dates.append(date)

    return failed_dates


def _download_metrics_zip(symbol: str, date: datetime, timeout: int = 30) -> None:
    response = requests.get(f"{BASE_URL}/{symbol.upper()}/{symbol.upper()}-metrics-{date:%Y-%m-%d}.zip", timeout=timeout)
    response.raise_for_status()
    get_zip_path(symbol, date).write_bytes(response.content)
