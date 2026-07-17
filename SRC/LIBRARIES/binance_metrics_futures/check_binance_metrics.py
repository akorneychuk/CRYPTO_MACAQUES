from datetime import datetime
from SRC.LIBRARIES.binance_metrics_futures import load_binance_metrics_futures
from SRC.LIBRARIES.binance_metrics_futures.constants import *


def print_header(title: str):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def main():
    SYMBOL = "ETHUSDT"
    START_DATE = datetime(2026, 6, 26)
    END_DATE = datetime(2026, 6, 29)

    # =============================================================================
    # TEST 1
    # =============================================================================

    print_header("TEST 1 - LOAD ALL COLUMNS")

    df = load_binance_metrics_futures(symbol=SYMBOL, start_date=START_DATE, end_date=END_DATE)

    print(df.head())
    print()
    print(df.tail())
    print()
    print(df.shape)
    print()
    print(df.dtypes)
    print()
    print(df.isna().sum())
    print()
    print(df.columns.tolist())

    # =============================================================================
    # TEST 2
    # =============================================================================

    print_header("TEST 2 - LOAD SELECTED COLUMNS")

    df = load_binance_metrics_futures(
        symbol=SYMBOL,
        start_date=START_DATE,
        end_date=END_DATE,
        columns=[
            SUM_OPEN_INTEREST,
            SUM_TAKER_LONG_SHORT_VOL_RATIO
        ],
    )

    print(df.head())
    print()
    print(df.columns.tolist())

    # =============================================================================
    # TEST 3
    # =============================================================================

    print_header("TEST 3 - CHECK SORTING")

    print(df[CREATE_TIME].head())
    print()
    print(df[CREATE_TIME].tail())
    print()
    print("Sorted:", df[CREATE_TIME].is_monotonic_increasing)

    # =============================================================================
    # TEST 4
    # =============================================================================

    print_header("TEST 4 - DUPLICATES")

    print("Duplicates:", df.duplicated(subset=[CREATE_TIME]).sum())

    # =============================================================================
    # TEST 5
    # =============================================================================

    print_header("TEST 5 - SUMMARY")

    print("Rows:", len(df))
    print("Start:", df[CREATE_TIME].min())
    print("End:", df[CREATE_TIME].max())


if __name__ == "__main__":
    main()
