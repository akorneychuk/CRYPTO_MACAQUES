from pathlib import Path

# =============================================================================
# PROJECT
# =============================================================================

DATA_DIR = Path(__file__).resolve().parents[3] / "DATA" / "binance" / "klines_spot"

# =============================================================================
# BINANCE
# =============================================================================

BASE_URL = "https://data.binance.vision/data/spot/daily/klines"

# =============================================================================
# CSV COLUMNS
# =============================================================================

KLINES_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]

KLINES_DTYPES = {
    "open_time": "int64",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
    "close_time": "int64",
    "quote_volume": "float64",
    "count": "int64",
    "taker_buy_volume": "float64",
    "taker_buy_quote_volume": "float64",
    "ignore": "int8",
}
