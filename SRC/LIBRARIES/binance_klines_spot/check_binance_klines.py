import pandas as pd
from SRC.LIBRARIES.binance_klines_spot import load_binance_klines_spot_for_trade
from SRC.LIBRARIES.binance_klines_spot.constants import *


def print_header(title: str):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def main():
    SYMBOL = "ETHUSDT"

    # UTC / Etc-GMT timezone-aware datetime
    TRADE_ENTRY_TIME = pd.Timestamp(
        "2026-07-16 21:45",
        tz="Etc/GMT",
    ).to_pydatetime()

    WINDOW_MINUTES = 3000

    # =============================================================================
    # TEST 1
    # =============================================================================

    print_header("TEST 1 - LOAD WINDOW")

    df = load_binance_klines_spot_for_trade(
        symbol=SYMBOL,
        trade_entry_time=TRADE_ENTRY_TIME,
        window_minutes=WINDOW_MINUTES,
    )

    print(df.head())
    print()
    print(df.tail())
    print()
    print(df.shape)

    # =============================================================================
    # TEST 2
    # =============================================================================

    print_header("TEST 2 - DTYPES")

    print(df.dtypes)
    print()
    print(df.isna().sum())
    print()
    print(df.columns.tolist())

    # =============================================================================
    # TEST 3
    # =============================================================================

    print_header("TEST 3 - INDEX")

    print("Timezone:", df.index.tz)
    print("Start:", df.index.min())
    print("End:", df.index.max())
    print()
    print("Sorted:", df.index.is_monotonic_increasing)
    print("Unique:", df.index.is_unique)

    # =============================================================================
    # TEST 4
    # =============================================================================

    print_header("TEST 4 - CONTINUITY")

    expected_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="1min",
    )

    print("Continuous:", df.index.equals(expected_index))
    print("Expected rows:", len(expected_index))
    print("Actual rows:", len(df))

    # =============================================================================
    # TEST 5
    # =============================================================================

    print_header("TEST 5 - COLUMNS")

    print("Columns match:", set(df.columns) == set(KLINES_COLUMNS))
    print()
    print(df.columns.tolist())

    # =============================================================================
    # TEST 6
    # =============================================================================

    print_header("TEST 6 - TIMESTAMPS")

    print("First open_time:", df["open_time"].iat[0])
    print("First datetime:", df.index[0])
    print()

    print("Last open_time:", df["open_time"].iat[-1])
    print("Last datetime:", df.index[-1])

    # =============================================================================
    # TEST 7
    # =============================================================================

    print_header("TEST 7 - WINDOW SUMMARY")

    print("Window minutes:", WINDOW_MINUTES)
    print("Rows:", len(df))
    print()

    print("Window start:", df.index[0])
    print("Window end:", df.index[-1])
    print("Trade entry:", TRADE_ENTRY_TIME)
    print()

    print("Last index == trade entry:", df.index[-1] == TRADE_ENTRY_TIME)
    print("Timedelta:", df.index[-1] - df.index[0])
    print()

    print("Unique time deltas:")
    print(df.index.to_series().diff().dropna().value_counts())


if __name__ == "__main__":
    main()
