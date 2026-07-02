from pathlib import Path

# =============================================================================
# PROJECT
# =============================================================================

DATA_DIR = Path(__file__).resolve().parents[3] / "DATA" / "binance" / "metrics"

# =============================================================================
# BINANCE
# =============================================================================

BASE_URL = "https://data.binance.vision/data/futures/um/daily/metrics"
BINANCE_ZIP_TF_MINUTES = 5

# =============================================================================
# CSV COLUMNS
# =============================================================================

CREATE_TIME = "create_time"
SYMBOL = "symbol"
SUM_OPEN_INTEREST = "sum_open_interest"
SUM_OPEN_INTEREST_VALUE = "sum_open_interest_value"
COUNT_TOPTRADER_LONG_SHORT_RATIO = "count_toptrader_long_short_ratio"
SUM_TOPTRADER_LONG_SHORT_RATIO = "sum_toptrader_long_short_ratio"
COUNT_LONG_SHORT_RATIO = "count_long_short_ratio"
SUM_TAKER_LONG_SHORT_VOL_RATIO = "sum_taker_long_short_vol_ratio"

ZIP_COLUMNS = [
    CREATE_TIME,
    SYMBOL,
    SUM_OPEN_INTEREST,
    SUM_OPEN_INTEREST_VALUE,
    COUNT_TOPTRADER_LONG_SHORT_RATIO,
    SUM_TOPTRADER_LONG_SHORT_RATIO,
    COUNT_LONG_SHORT_RATIO,
    SUM_TAKER_LONG_SHORT_VOL_RATIO,
]
