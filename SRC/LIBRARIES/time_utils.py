import hashlib
import random
import re
from contextlib import suppress
from datetime import datetime, timedelta
from datetime import timezone

import pandas as pd
import pytz
from pandas import Timestamp

from SRC.CORE._CONSTANTS import KIEV_TZ, UTC_TZ, PARTITIONING_MAP
from SRC.CORE._CONSTANTS import project_root_dir
from SRC.CORE.utils import read_json, write_json, datetime_h_m_s

with suppress(Exception):
    from SRC.LIBRARIES.new_utils import run_safety_interrupter

INTERVAL = lambda discretization: int(''.join(re.findall(r'\d+', discretization)))
PARTITIONING = lambda discretization: ''.join(re.findall(r'[a-zA-Z]+', discretization))
INTERVAL_PARTITION = lambda discretization: f"{INTERVAL(discretization)}{PARTITIONING_MAP[PARTITIONING(discretization)]}"

TIME_DELTA = lambda discretization: \
    timedelta(seconds=INTERVAL(discretization)) if PARTITIONING(discretization) == 'S' else \
    timedelta(minutes=INTERVAL(discretization)) if PARTITIONING(discretization) == 'M' else \
    timedelta(hours=INTERVAL(discretization)) if PARTITIONING(discretization) == 'H' else \
    timedelta(days=INTERVAL(discretization)) if PARTITIONING(discretization) == 'D' else \
    timedelta(days=INTERVAL(discretization) * 7) if PARTITIONING(discretization) == 'W' else None

TIME_DELTA_S = lambda secs: TIME_DELTA(f"{secs}S")

def get_hours_difference_between_timezones(tz1, tz2):
    hours_difference = int(abs(((datetime.min + timedelta(days=1)).astimezone(tz1).utcoffset() - (datetime.min + timedelta(days=1)).astimezone(tz2).utcoffset()).total_seconds() / 3600))

    return hours_difference


def timezone_to_pytz(dt: datetime) -> datetime:
    """
    Convert a datetime with datetime.timezone tzinfo to pytz tzinfo
    preserving the same offset.
    """
    tz = dt.tzinfo
    if tz is None:
        return dt  # naive, nothing to do

    if isinstance(tz, timezone):
        # Get offset in minutes
        offset_minutes = int(tz.utcoffset(dt).total_seconds() // 60)
        pytz_tz = pytz.FixedOffset(offset_minutes)
        return dt.astimezone(pytz_tz)

    # Already a pytz tzinfo → do nothing
    return dt


def round_up_to_nearest_step(dt, step: timedelta):
    if isinstance(dt, Timestamp):
        dt = dt.to_pydatetime()

    _round_down = lambda dt_: dt_ - (dt_ - datetime.min) % step
    _round_up = lambda dt_: _round_down(dt_) if (dt_ - datetime.min) % step == timedelta(0) else _round_down(dt_) + step

    if dt.tzinfo is not None:
        tz_info = dt.tzinfo
        if isinstance(tz_info, timezone):
            dt = timezone_to_pytz(dt)
            tz_info = dt.tzinfo

        no_tz_dt = dt.replace(tzinfo=None)
        rounded_up = _round_up(no_tz_dt)
        rounded_up_tz = tz_info.localize(rounded_up)

        return rounded_up_tz

    rounded_up = _round_up(dt)

    return rounded_up


def round_up_to_nearest_sec(dt):
    return round_up_to_nearest_step(dt, TIME_DELTA('1S'))


def round_up_to_nearest_min(dt):
    return round_up_to_nearest_step(dt, TIME_DELTA('1M'))


def round_up_to_nearest_hour(dt):
    return round_up_to_nearest_step(dt, TIME_DELTA('1H'))


def round_down_to_nearest_step(dt, step: timedelta):
    if type(dt) == Timestamp:
        dt = dt.to_pydatetime()

    _round_down = lambda dt_: dt_ - (dt_ - datetime.min) % step

    if dt.tzinfo is not None:
        tz_info = dt.tzinfo
        no_tz_dt = dt.replace(tzinfo=None)
        rounded_down = _round_down(no_tz_dt)
        rounded_down_tz = tz_info.localize(rounded_down)

        return rounded_down_tz

    rounded_down = _round_down(dt)

    return rounded_down


def round_down_to_nearest_sec(dt):
    return round_down_to_nearest_step(dt, TIME_DELTA('1S'))


def round_down_to_nearest_min(dt):
    return round_down_to_nearest_step(dt, TIME_DELTA('1M'))


def round_down_to_nearest_hour(dt):
    return round_down_to_nearest_step(dt, TIME_DELTA('1H'))


def utc_now():
    return UTC_TZ.localize(datetime.utcnow()).astimezone(UTC_TZ)


def kiev_now():
    return UTC_TZ.localize(datetime.utcnow()).astimezone(KIEV_TZ)


def as_tz(dt, tz):
    if type(dt) == int or type(dt) == float:
        dt = datetime.fromtimestamp(dt / 1_000)

    if type(dt) == str:
        if dt == '':
            return None

        dt = datetime.fromisoformat(dt)

    if type(dt) == Timestamp:
        dt = dt.to_pydatetime()

    if dt.tzinfo is None:
        dt_tz = UTC_TZ.localize(dt).astimezone(tz)
    else:
        dt_tz = dt.replace(tzinfo=timezone.utc).astimezone(tz)

    return dt_tz


def as_utc_tz(dt):
    return as_tz(dt, UTC_TZ)


def as_kiev_tz(dt):
    return as_tz(dt, KIEV_TZ)


def localize_tz(dt, tz):
    if type(dt) == int or type(dt) == float:
        dt = datetime.fromtimestamp(dt / 1_000, tz=tz)

    if type(dt) == str:
        if dt == '':
            return None

        dt = datetime.fromisoformat(dt)

    if type(dt) == Timestamp:
        dt = dt.to_pydatetime()

    if dt.tzinfo is None:
        dt = UTC_TZ.localize(dt).astimezone(tz)

    return dt


def localize_kiev_tz(dt):
    return localize_tz(dt, KIEV_TZ)


def localize_utc_tz(dt):
    return localize_tz(dt, UTC_TZ)


def get_datetime_splitters(datetime_list, discretization='1D', as_tz=None):
    if not datetime_list:
        return []

    day_splitter_s = []
    for i in range(1, len(datetime_list)):
        if as_tz is not None:
            if as_tz(datetime_list[i]).date() != as_tz(datetime_list[i - 1]).date():
                midnight_dt = round_up_to_nearest_step(datetime_list[i - 1], TIME_DELTA(discretization=discretization))
                day_splitter_s.append(midnight_dt)
        else:
            if datetime_list[i].date() != datetime_list[i - 1].date():
                midnight_dt = round_up_to_nearest_step(datetime_list[i - 1], TIME_DELTA(discretization=discretization))
                day_splitter_s.append(midnight_dt)

    return day_splitter_s


def block_until_next(discretization: str, title:str=None):
    from SRC.LIBRARIES.new_utils import run_safety_interrupter

    now = kiev_now()
    next_time = round_up_to_nearest_step(now, TIME_DELTA(discretization))
    wait_time = (next_time - now).total_seconds()
    if title:
        run_safety_interrupter(f"NEXT [{title}]", wait_time)
    else:
        run_safety_interrupter(f"NEXT", wait_time)


def parse_iso_format_timestamp(iso_timestamp_str):
    if iso_timestamp_str:
        try:
            return datetime.strptime(iso_timestamp_str, "%Y-%m-%dT%H:%M:%S.%f%z").replace(tzinfo=None)
        except ValueError:
            try:
                return datetime.strptime(iso_timestamp_str, "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
            except ValueError:
                return datetime.strptime(iso_timestamp_str, "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=None)

    return None


def in_range(dt, start, end):
    return as_utc_tz(start) <= as_utc_tz(dt) <= as_utc_tz(end)


def utc_now_formatted():
    return utc_now().strftime("%Y-%m-%d %H:%M:%S%Z")


def kiev_now_formatted():
    return kiev_now().strftime("%Y-%m-%d %H:%M:%S%Z")


def deterministic_int(dt):
    s = dt.isoformat()  # важно: стабильное представление
    h = hashlib.sha256(s.encode()).digest()

    return int.from_bytes(h[:8], 'big')  # берём 64 бита


def TEST_round_down_to_nearest_step():
    datetime_obj_kiev_tz = KIEV_TZ.localize(pd.Timestamp.now())
    print(f"{datetime_obj_kiev_tz} >>>> {round_down_to_nearest_step(datetime_obj_kiev_tz, TIME_DELTA('5M'))}")

    datetime_obj_not_tz = datetime_obj_kiev_tz.replace(tzinfo=None)
    print(f"{datetime_obj_not_tz} >>>> {round_down_to_nearest_step(datetime_obj_not_tz, TIME_DELTA('5M'))}")


def TEST__convert__datetime__pandas_timestamp():
    # Convert pandas Timestamp to Python datetime
    timestamp = pd.Timestamp.now()
    datetime_obj = timestamp.to_pydatetime()
    print("Pandas Timestamp:", timestamp)
    print("Python datetime:", datetime_obj)

    # Convert Python datetime to pandas Timestamp
    datetime_obj = pd.Timestamp(datetime_obj)
    timestamp = pd.to_datetime(datetime_obj)
    print("Python datetime:", datetime_obj)
    print("Pandas Timestamp:", timestamp)


def TEST__write__read__candle(dt_now):
    candle_file_path = f'{project_root_dir()}/price.json'

    candle = {'close_time': dt_now, 'open': 1, 'high': 1, 'low': 1, 'close': 1}
    write_json(candle, candle_file_path)

    candle = read_json(candle_file_path)
    dt = candle['close_time']
    dt_rounded = round_down_to_nearest_step(dt, TIME_DELTA('5M'))
    print(f"UTC: {dt_rounded}")
    print(f"KIEV: {dt_rounded.astimezone(KIEV_TZ)}")
    print(dt.tzinfo)


if __name__ == "__main__":
    print(kiev_now())

    block_until_next('10S')
    print(kiev_now())

    block_until_next('30S')
    print(kiev_now())

    block_until_next('1M')
    print(kiev_now())

    block_until_next('5M')
    print(kiev_now())

    block_until_next('30M')
    print(kiev_now())

    block_until_next('1H')
    print(kiev_now())

    now_dt_kiev = UTC_TZ.localize(datetime.utcnow()).astimezone(KIEV_TZ) + timedelta(seconds=random.randint(-5, 5))
    now_dt_kiev_present = datetime_h_m_s(now_dt_kiev)
    remains_dt_kiev_present = f"- {datetime_h_m_s(now_dt_kiev + timedelta(minutes=3) - now_dt_kiev)}"
    print(remains_dt_kiev_present)

    print("-------------------------------------")
    TEST_round_down_to_nearest_step()
    print("-------------------------------------")
    TEST__convert__datetime__pandas_timestamp()
    print("-------------------------------------")
    TEST__write__read__candle(UTC_TZ.localize(datetime.utcnow()).astimezone(KIEV_TZ))
    print("-------------------------------------")
    TEST__write__read__candle(UTC_TZ.localize(datetime.utcnow()))
    print("=====================================")
