import distutils
import os
import random
import re
import sys
import dateutil
import numpy as np
import pytz
from datetime import datetime, timedelta
import warnings
import platform
from functools import lru_cache
from pathlib import Path


warnings.filterwarnings('ignore')

DISCRETIZATION_KEY = 'DISCRETIZATION'
FORCE_DISCRETIZATION_KEY = 'FORCE_DISCRETIZATION'
TARGET_FEATURE_WINDOW_KEY = 'TARGET_FEATURE_WINDOW'
CLASS_KEY = 'CLASS'
POWER_DEGREE_KEY = 'POWER_DEGREE'
NON_LINEARITY_TOP_KEY = 'NON_LINEARITY_TOP'
EPOCHS_KEY = 'EPOCHS'
BATCH_SIZE_KEY = 'BATCH_SIZE'
LEARNING_RATE_KEY = 'LEARNING_RATE'
WEIGHTS_BOOSTER_COEF_KEY = 'WEIGHTS_BOOSTER_COEF'
IS_CONVOLUTIONAL_KEY = 'IS_CONVOLUTIONAL'
CONV_KEY = 'CONV'

STABLE_SYMBOL_DIVIDE_BUY_KEY = 'STABLE_SYMBOL_DIVIDE_BUY'

INITIAL_SIM_TRADE_DATA_WINDOW_KEY = 'INITIAL_SIM_TRADE_DATA_WINDOW'

MODEL_SUFFIX_KEY = 'MODEL_SUFFIX'
META_SUFFIX_KEY = 'META_SUFFIX'

SIMULATION__START_DATE_KEY = 'SIMULATION__START_DATE'
SIMULATION__END_DATE_KEY = 'SIMULATION__END_DATE'
SIMULATION__TAKE_DELTA_KEY = 'SIMULATION__TAKE_DELTA'

# TRAIN__START_DATE_KEY = 'TRAIN__START_DATE'
# TRAIN__END_DATE_KEY = 'TRAIN__END_DATE'

_BINANCE_FEE_ENABLED = 'BINANCE_FEE_ENABLED'
_BINANCE_FEE = 'BINANCE_FEE'
# _TAKER_FEE = 'TAKER_FEE'
# _MAKER_FEE = 'MAKER_FEE'

@lru_cache(maxsize=None)
def default_cpu_work_count(text):
	if _DASHBOARD_SEGMENT in os.environ and os.environ[_DASHBOARD_SEGMENT] == _DASHBOARD_SEGMENT_BACKTESTING:
		raise Exception(text)

	return 1

CPU_COUNT = lambda: default_cpu_work_count("Specify os.environ[_CPU_COUNT]: -> int") if _CPU_COUNT not in os.environ else int(os.environ[_CPU_COUNT])
WORKERS_COUNT = lambda: default_cpu_work_count("Specify os.environ[_WORKERS_COUNT]: -> int") if _WORKERS_COUNT not in os.environ else int(os.environ[_WORKERS_COUNT])

_USE_GPU = 'USE_GPU'

def GPU_COUNT():
	import torch

	return torch.cuda.device_count()

USE_GPU = lambda: True if _USE_GPU not in os.environ else string_bool(os.environ[_USE_GPU])

INTERRUPT_SEGMENTIZE = lambda: True
# INTERRUPT_SEGMENTIZE = lambda: False

INTERRUPT_PRETRAIN = lambda: True
INTERRUPT_PRETRAIN = lambda: False

INTERRUPT_PRETRAIN_SAVE = lambda: True
# INTERRUPT_PRETRAIN_SAVE = lambda: False

INTERRUPT_TRAIN = lambda: True
INTERRUPT_VALIDATE = lambda: False
INTERRUPT_SIMULATE = lambda: True

INTERRUPT_TRADE = lambda: True

CLEAR_OUTPUT_DASH = lambda: any([x in os.environ[NOTEBOOK_NAME_KEY] for x in ["trade", "train"]])
CLEAR_OUTPUT_DASH = lambda: False

RSI_TOP = lambda: 70
RSI_BOTTOM = lambda: 30

FORCE_DISCRETIZATION = lambda: os.environ[FORCE_DISCRETIZATION_KEY] if FORCE_DISCRETIZATION_KEY in os.environ else DISCRETIZATION()
# DISCRETIZATION = lambda: os.environ[DISCRETIZATION_KEY]
# DISCRETIZATION = lambda: '1M'                       if DISCRETIZATION_KEY not in os.environ else os.environ[DISCRETIZATION_KEY]
DISCRETIZATION = lambda: '3M' if DISCRETIZATION_KEY not in os.environ else os.environ[DISCRETIZATION_KEY]
# DISCRETIZATION = lambda: '5M'                       if DISCRETIZATION_KEY not in os.environ else os.environ[DISCRETIZATION_KEY]
# DISCRETIZATION = lambda: '15M'                       if DISCRETIZATION_KEY not in os.environ else os.environ[DISCRETIZATION_KEY]
# DISCRETIZATION = lambda: '1H'                       if DISCRETIZATION_KEY not in os.environ else os.environ[DISCRETIZATION_KEY]

INTERVAL = lambda: int(''.join(re.findall(r'\d+', DISCRETIZATION())))
PARTITIONING = lambda: ''.join(re.findall(r'[a-zA-Z]+', DISCRETIZATION()))

FORCE_INTERVAL = lambda: int(''.join(re.findall(r'\d+', FORCE_DISCRETIZATION())))
FORCE_PARTITIONING = lambda: ''.join(re.findall(r'[a-zA-Z]+', FORCE_DISCRETIZATION()))

SMALL_COINS = ['MULTI', 'PERL', 'VIB', 'AUCTION', 'DOCK', 'HOT', 'OOKI', 'BETA', 'REQ', 'T', 'LOOM', 'BNX', 'DOGE', 'FRONT', 'UNFI', 'CYBER', 'BNT', 'LUNA']
BIG_COINS = ['ETH', 'BNB', 'BTC']
CACHE_PROCESS_SYMBOLS = lambda: SMALL_COINS
CACHE_PROCESS_SYMBOLS = lambda: BIG_COINS
# CACHE_PROCESS_SYMBOLS = lambda: [*BIG_COINS, *SMALL_COINS]
# CACHE_PROCESS_SYMBOLS = lambda: ['TUSD/USDT', 'TUSD/BUSD', 'FDUSD/USDT', 'FDUSD/BUSD']
CACHE_PROCESS_SYMBOLS = lambda: ['BETA/USDT', 'MULTI/USDT', 'VIB/USDT', 'DOCK/USDT', 'HOT/USDT', 'OOKI/USDT', 'FRONT/USDT', 'BNX/USDT', 'UNFI/USDT']
# CACHE_PROCESS_SYMBOLS = lambda: ['BETA/USDT']

# TRAIN_SYMBOLS = lambda: CACHE_PROCESS_SYMBOLS()[:18]
TRAIN_SYMBOLS = lambda: CACHE_PROCESS_SYMBOLS()

CACHED__START_DATE = datetime(2023, 10, 8)  # !!!!!DON'T TOUCH IT!!!!!!

if PARTITIONING() == 'M':

	###########################################################################################
	if INTERVAL() == 1:
		TARGET_FEATURE_WINDOW = lambda: 2 if TARGET_FEATURE_WINDOW_KEY not in os.environ else int(os.environ[TARGET_FEATURE_WINDOW_KEY])

		CLASSES = lambda: 15 if CLASS_KEY not in os.environ else int(os.environ[CLASS_KEY])
		POWER_DEGREE = lambda: 2.75 if POWER_DEGREE_KEY not in os.environ else float(os.environ[POWER_DEGREE_KEY])
		NON_LINEARITY_TOP = lambda: 0.3 if NON_LINEARITY_TOP_KEY not in os.environ else float(os.environ[NON_LINEARITY_TOP_KEY])

		EPOCHS = lambda: 15 if EPOCHS_KEY not in os.environ else int(os.environ[EPOCHS_KEY])
		BATCH_SIZE = lambda: 100 if BATCH_SIZE_KEY not in os.environ else int(os.environ[BATCH_SIZE_KEY])
		LEARNING_RATE = lambda: 0.00001 if LEARNING_RATE_KEY not in os.environ else float(os.environ[LEARNING_RATE_KEY])
		WEIGHTS_BOOSTER_COEF = lambda: 2.5 if WEIGHTS_BOOSTER_COEF_KEY not in os.environ else float(os.environ[WEIGHTS_BOOSTER_COEF_KEY])
		IS_CONVOLUTIONAL = lambda: False if IS_CONVOLUTIONAL_KEY not in os.environ else os.environ[IS_CONVOLUTIONAL_KEY] == 'True'

		TEST_SIZE = lambda: 0.0001

	###########################################################################################
	if INTERVAL() == 3:
		TARGET_FEATURE_WINDOW = lambda: 2 if TARGET_FEATURE_WINDOW_KEY not in os.environ else int(os.environ[TARGET_FEATURE_WINDOW_KEY])

		CLASSES = lambda: 15 if CLASS_KEY not in os.environ else int(os.environ[CLASS_KEY])
		POWER_DEGREE = lambda: 3 if POWER_DEGREE_KEY not in os.environ else float(os.environ[POWER_DEGREE_KEY])
		NON_LINEARITY_TOP = lambda: 0.3 if NON_LINEARITY_TOP_KEY not in os.environ else float(os.environ[NON_LINEARITY_TOP_KEY])

		EPOCHS = lambda: 15 if EPOCHS_KEY not in os.environ else int(os.environ[EPOCHS_KEY])
		BATCH_SIZE = lambda: 1_000 if BATCH_SIZE_KEY not in os.environ else int(os.environ[BATCH_SIZE_KEY])
		LEARNING_RATE = lambda: 0.0001 if LEARNING_RATE_KEY not in os.environ else float(os.environ[LEARNING_RATE_KEY])
		WEIGHTS_BOOSTER_COEF = lambda: 2.1 if WEIGHTS_BOOSTER_COEF_KEY not in os.environ else float(os.environ[WEIGHTS_BOOSTER_COEF_KEY])
		IS_CONVOLUTIONAL = lambda: False if IS_CONVOLUTIONAL_KEY not in os.environ else os.environ[IS_CONVOLUTIONAL_KEY] == 'True'

		TEST_SIZE = lambda: 0.001

	###########################################################################################
	if INTERVAL() == 5:
		TARGET_FEATURE_WINDOW = lambda: 2 if TARGET_FEATURE_WINDOW_KEY not in os.environ else int(os.environ[TARGET_FEATURE_WINDOW_KEY])

		CLASSES = lambda: 15 if CLASS_KEY not in os.environ else int(os.environ[CLASS_KEY])
		POWER_DEGREE = lambda: 3 if POWER_DEGREE_KEY not in os.environ else float(os.environ[POWER_DEGREE_KEY])
		NON_LINEARITY_TOP = lambda: 0.3 if NON_LINEARITY_TOP_KEY not in os.environ else float(os.environ[NON_LINEARITY_TOP_KEY])

		EPOCHS = lambda: 5 if EPOCHS_KEY not in os.environ else int(os.environ[EPOCHS_KEY])
		BATCH_SIZE = lambda: 10_000 if BATCH_SIZE_KEY not in os.environ else int(os.environ[BATCH_SIZE_KEY])
		LEARNING_RATE = lambda: 0.001 if LEARNING_RATE_KEY not in os.environ else float(os.environ[LEARNING_RATE_KEY])
		WEIGHTS_BOOSTER_COEF = lambda: 2.5 if WEIGHTS_BOOSTER_COEF_KEY not in os.environ else float(os.environ[WEIGHTS_BOOSTER_COEF_KEY])
		IS_CONVOLUTIONAL = lambda: False if IS_CONVOLUTIONAL_KEY not in os.environ else os.environ[IS_CONVOLUTIONAL_KEY] == 'True'

		TEST_SIZE = lambda: 0.001

	###########################################################################################
	if INTERVAL() == 15 or INTERVAL() == 30:
		TARGET_FEATURE_WINDOW = lambda: 2 if TARGET_FEATURE_WINDOW_KEY not in os.environ else int(os.environ[TARGET_FEATURE_WINDOW_KEY])

		CLASSES = lambda: 15 if CLASS_KEY not in os.environ else int(os.environ[CLASS_KEY])
		POWER_DEGREE = lambda: 3 if POWER_DEGREE_KEY not in os.environ else float(os.environ[POWER_DEGREE_KEY])
		NON_LINEARITY_TOP = lambda: 0.3 if NON_LINEARITY_TOP_KEY not in os.environ else float(os.environ[NON_LINEARITY_TOP_KEY])

		EPOCHS = lambda: 5 if EPOCHS_KEY not in os.environ else int(os.environ[EPOCHS_KEY])
		BATCH_SIZE = lambda: 10_000 if BATCH_SIZE_KEY not in os.environ else int(os.environ[BATCH_SIZE_KEY])
		LEARNING_RATE = lambda: 0.001 if LEARNING_RATE_KEY not in os.environ else float(os.environ[LEARNING_RATE_KEY])
		WEIGHTS_BOOSTER_COEF = lambda: 2.5 if WEIGHTS_BOOSTER_COEF_KEY not in os.environ else float(os.environ[WEIGHTS_BOOSTER_COEF_KEY])
		IS_CONVOLUTIONAL = lambda: False if IS_CONVOLUTIONAL_KEY not in os.environ else os.environ[IS_CONVOLUTIONAL_KEY] == 'True'

		TEST_SIZE = lambda: 0.001

	MODEL_SUFFIX = lambda: None if MODEL_SUFFIX_KEY not in os.environ else os.environ[MODEL_SUFFIX_KEY] if os.environ[MODEL_SUFFIX_KEY] != 'None' else None
	META_SUFFIX = lambda: None if META_SUFFIX_KEY not in os.environ else os.environ[META_SUFFIX_KEY].replace('/', '|') if os.environ[META_SUFFIX_KEY] != 'None' else None
	MAX_FLUCTUATION_RANGE = lambda: timedelta(minutes=int(60))

	SIMULATION__START_DATE = lambda: dateutil.parser.parse("2023-08-1T00:00:00Z" if SIMULATION__START_DATE_KEY not in os.environ else os.environ[SIMULATION__START_DATE_KEY])
	SIMULATION__END_DATE = lambda: dateutil.parser.parse("2023-09-30T00:00:00Z" if SIMULATION__END_DATE_KEY not in os.environ else os.environ[SIMULATION__END_DATE_KEY])
	SIMULATION__TAKE_DELTA = lambda: timedelta(days=60)

	TRAIN__TAKE_RATIO = lambda: 1
	# TRAIN__START_DATE = lambda: dateutil.parser.parse("2019-01-01T00:00:00Z" if TRAIN__START_DATE_KEY not in os.environ else os.environ[TRAIN__START_DATE_KEY])
	# TRAIN__END_DATE = lambda: dateutil.parser.parse("2024-01-01T00:00:00Z" if TRAIN__END_DATE_KEY not in os.environ else os.environ[TRAIN__END_DATE_KEY])

###########################################################################################
if 'H' in DISCRETIZATION():
	TARGET_FEATURE_WINDOW = lambda: 2 if TARGET_FEATURE_WINDOW_KEY not in os.environ else int(os.environ[TARGET_FEATURE_WINDOW_KEY])

	CLASSES = lambda: 15 if CLASS_KEY not in os.environ else int(os.environ[CLASS_KEY])
	POWER_DEGREE = lambda: 4 if POWER_DEGREE_KEY not in os.environ else float(os.environ[POWER_DEGREE_KEY])
	NON_LINEARITY_TOP = lambda: 0.2 if NON_LINEARITY_TOP_KEY not in os.environ else float(os.environ[NON_LINEARITY_TOP_KEY])

	EPOCHS = lambda: 50 if EPOCHS_KEY not in os.environ else int(os.environ[EPOCHS_KEY])
	BATCH_SIZE = lambda: 10 if BATCH_SIZE_KEY not in os.environ else int(os.environ[BATCH_SIZE_KEY])
	LEARNING_RATE = lambda: 0.001 if LEARNING_RATE_KEY not in os.environ else float(os.environ[LEARNING_RATE_KEY])
	WEIGHTS_BOOSTER_COEF = lambda: 2.1 if WEIGHTS_BOOSTER_COEF_KEY not in os.environ else float(os.environ[WEIGHTS_BOOSTER_COEF_KEY])
	IS_CONVOLUTIONAL = lambda: False if IS_CONVOLUTIONAL_KEY not in os.environ else os.environ[IS_CONVOLUTIONAL_KEY] == 'True'

	TEST_SIZE = lambda: 0.001

	MODEL_SUFFIX = lambda: None if MODEL_SUFFIX_KEY not in os.environ else os.environ[MODEL_SUFFIX_KEY] if os.environ[MODEL_SUFFIX_KEY] != 'None' else None
	META_SUFFIX = lambda: None if META_SUFFIX_KEY not in os.environ else os.environ[META_SUFFIX_KEY].replace('/', '|') if os.environ[META_SUFFIX_KEY] != 'None' else None
	MAX_FLUCTUATION_RANGE = lambda: timedelta(days=int(60))

	SIMULATION__START_DATE = lambda: dateutil.parser.parse("2023-08-1T00:00:00Z" if SIMULATION__START_DATE_KEY not in os.environ else os.environ[SIMULATION__START_DATE_KEY])
	SIMULATION__END_DATE = lambda: dateutil.parser.parse("2023-09-30T00:00:00Z" if SIMULATION__END_DATE_KEY not in os.environ else os.environ[SIMULATION__END_DATE_KEY])
	SIMULATION__TAKE_DELTA = lambda: timedelta(days=60)

	TRAIN__TAKE_RATIO = lambda: 0.2
	# TRAIN__START_DATE = lambda: dateutil.parser.parse("2019-01-01T00:00:00Z" if TRAIN__START_DATE_KEY not in os.environ else os.environ[TRAIN__START_DATE_KEY])
	# TRAIN__END_DATE = lambda: dateutil.parser.parse("2024-01-01T00:00:00Z" if TRAIN__END_DATE_KEY not in os.environ else os.environ[TRAIN__END_DATE_KEY])

MEMORY_MONITOR_UPDATE_INTERVAL_MINS = lambda: 15 if "trade" in os.environ[NOTEBOOK_NAME_KEY] else 3 if '3.8.10' in sys.version else 1

DASH_TRAIN_UPDATE_INTERVAL_SECS = lambda: 30

PARTITIONING_MAP = {'S': 'S', 'M': 'T', 'H': 'H', 'D': 'D'}


IS_MEMORY_MONITOR_ENABLED = lambda: False

BINANCE_FEE = lambda: 0.1 if _BINANCE_FEE not in os.environ else float(os.environ[_BINANCE_FEE]) if '-' not in os.environ[_BINANCE_FEE] else float(os.environ[_BINANCE_FEE].split("-")[0])
BINANCE_FEE_ENABLED = lambda: (True if _BINANCE_FEE_ENABLED not in os.environ else os.environ[_BINANCE_FEE_ENABLED].lower() == str(True).lower())

BINANCE_FEE_PERCENT = lambda: BINANCE_FEE() * 1 if BINANCE_FEE_ENABLED() else 0
BINANCE_COMISSION = lambda: 0.01 * BINANCE_FEE_PERCENT()
BINANCE_MAKER_COMISSION = lru_cache(maxsize=None)(lambda: BINANCE_COMISSION() if '-' not in os.environ[_BINANCE_FEE] else round(0.01 * float(os.environ[_BINANCE_FEE].split("-")[0]), 8))
BINANCE_TAKER_COMISSION = lru_cache(maxsize=None)(lambda: BINANCE_COMISSION() if '-' not in os.environ[_BINANCE_FEE] else round(0.01 * float(os.environ[_BINANCE_FEE].split("-")[1]), 8))

PROCESS_SYMBOL = lambda trade_symbol: f'{trade_symbol.split("/")[0]}{trade_symbol.split("/")[1]}'
PROCESS_SYMBOL_S = lambda trade_symbol_s: list(map(lambda trade_symbol: PROCESS_SYMBOL(trade_symbol), trade_symbol_s))

LOG_BASE = lambda: 4
LOG_START = lambda: 0.000001


@lru_cache(maxsize=None)
def project_root_dir():
	current_directory = Path.cwd()
	counter = 0

	while not (current_directory / 'README.md').is_file():
		if counter >= 10:
			raise Exception("!!No README found!!")
		current_directory = current_directory.parent
		counter += 1

	return current_directory


def string_bool(string_bool):
	return bool(distutils.util.strtobool(string_bool))


SEGMENT_LENGTH = lambda: 30
SEGMENT_OVERLAP = lambda: 29
# SEGMENT_LENGTH = lambda: 90
# SEGMENT_OVERLAP = lambda: 89

FEATURE_DIFF = 'mean_rel_diff'
MEAN_GRAD = 'mean_grad'

FEATURE_2 = f'{MEAN_GRAD}_2'
FEATURE_3 = f'{MEAN_GRAD}_3'

FEATURE_DIFF_NORM = f"{FEATURE_DIFF}_NORM"
FEATURE_2_NORM = f"{FEATURE_2}_NORM"
FEATURE_3_NORM = f"{FEATURE_3}_NORM"

TARGET_FEATURE = lambda: FEATURE_DIFF if TARGET_FEATURE_WINDOW() == 1 else f'{MEAN_GRAD}_{TARGET_FEATURE_WINDOW()}'
TARGET_FEATURE_NORM = lambda: f"{TARGET_FEATURE()}_NORM"

NETWORK_INPUT_SIZE = lambda: SEGMENT_LENGTH() - TARGET_FEATURE_WINDOW()

CENTER_CLASS = lambda: int((CLASSES() - 1) / 2)
SYMMETRIC_CLASSES = lambda: np.array([int(clas) for clas in np.linspace(0, CLASSES() - 1, CLASSES())]) - CENTER_CLASS()

AUC_ROC_DOWN_SAMPLING_LIMIT = lambda: 5_000

SHUFFLE = True

NON_LINEARITY_TOP_DEFAULT = 0.3
N_CLASSES_LINEAR = 101
UNBALANCED_CENTER_RATIO = -1

PYTHON_STARTING_DIR_KEY = 'PYTHON_STARTING_DIR'
PARENT_PROCESS_ID_KEY = 'PARENT_PROCESS_ID'
NOTEBOOK_NAME_KEY = 'NOTEBOOK_NAME'
LOG_FILE_NAME_KEY = 'LOG_FILE_NAME'
EXPORT_FILE_NAME_KEY = 'EXPORT_FILE_NAME'

SYMBOL_PROCESS_KEY = 'SYMBOL'
SYMBOL_TRADING_KEY = 'SYMBOL_TRADING'

START_TRAIN_KEY = 'START_TRAIN'
END_TRAIN_KEY = 'END_TRAIN'

START_CACHE_PROCESS_KEY = 'START_CACHE_PROCESS'
END_CACHE_PROCESS_KEY = 'END_CACHE_PROCESS'

FEATURE_RATIO_PERCENT_TOLERANCE = 10

START_COL = 'start_ts'
END_COL = 'end_ts'
PROCESSED_DATE_COLS = [START_COL, END_COL]

FORCE_FEATURIZE = False
FORCE_PROCESS = False

ROLLING_DIFF_WINDOW_FREQ = 2
ROLLING_GRAD_WINDOW_FREQs = [2, 3]

FEATURES = [*[FEATURE_DIFF], *list(map(lambda grad_freq: f'{MEAN_GRAD}_{grad_freq}', ROLLING_GRAD_WINDOW_FREQs))]

CACHED_FOLDER_PATH = f'../__DATA/CACHED'
PROCESSED_FOLDER_PATH = f'../__DATA/SEGMENTIZED'
TRAIN_FOLDER_PATH = f'../__DATA/PRETRAIN'
META_FOLDER_PATH = f'../__DATA/META'
MODEL_FOLDER_PATH = f'../__DATA/MODEL'
TRADE_FOLDER_PATH = lambda is_simulation: f"../__DATA/TRADING/{'SIMULATION' if is_simulation else 'TRADING'}"

FEATURE_KEY = 'FEATURE'
FEATURE_S_KEY = 'FEATURES'
FEATURE_NORM_KEY = 'FEATURE_NORM'
SEGMENT_OFFSET_KEY = 'SEGMENT_OFFSET'
CUT_OFFSET_KEY = 'CUT_OFFSET'
SPACE_PRODUCER_KEY = 'SPACE_PRODUCER'
SPACE_PRODUCER_NORM_KEY = 'SPACE_PRODUCER_NORM'

EXTREMUMS_KEY = 'EXTREMUMS'
FEATURES_KEY = 'FEATURES'
SPACE_KEY = 'SPACE'
BINS_KEY = 'BINS'
BINS_PAIRWISE_KEY = 'BINS_PAIRWISE'
COUNTS_KEY = 'COUNTS'
WEIGHTS_KEY = 'WEIGHTS'
WEIGHTS_COUNT_PRODUCT_NORM_KEY = 'WEIGHTS_COUNT_PRODUCT_NORM'
SPACE_PRODUCER_SERIALIZED_KEY = 'SPACE_PRODUCER_SERIALIZED'

STABLE_COIN_KEY = 'STABLE_COIN'
ALT_COIN_KEY = 'ALT_COIN'
BALANCE_KEY = 'BALANCE'
DATA_KEY = 'DATA'
DATE_KEY = 'DATE'

BATCH_MEASURE_SIZE = 100_000

BIN_SIZES = {"1m": 1, "5m": 5, "1h": 60, "1d": 1440}
TIME_ZONE_HOURS_OFFSET = 3
CANDLES_SECS_BUFFER_SIZE = 1800
COLORS = {
	'Blue': '#1f77b4',
	'Orange': '#ff7f0e',
	'Green': '#2ca02c',
	'Red': '#d62728',
	'Purple': '#9467bd',
	'Brown': '#8c564b',
	'Pink': '#e377c2',
	'Gray': '#7f7f7f',
	'Cyan': '#17becf',
	'Yellow': '#bcbd22',
	'Lime': '#00ff00',
	'Teal': '#008080',
	'Indigo': '#4b0082',
	'Magenta': '#ff00ff',
	'Olive': '#808000',
	'Navy': '#000080',
	'Maroon': '#800000',
	'Aquamarine': '#7fffd4',
	'Gold': '#ffd700',
	'Violet': '#ee82ee',
	'Turquoise': '#40e0d0',
	'Slate': '#6a5acd',
	'Coral': '#ff7f50',
	'Sky Blue': '#87ceeb',
}

@lru_cache(maxsize=None)
def generate_color_s(count):
	color_s = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(count)]

	return color_s


STATE_IN = "IN"
STATE_OUT = "OUT"

ACTION_BUY = "BUY"
ACTION_SELL = "SELL"
ACTION_NO = "NO"

KIEV_UTC_HOURS_DIFF = int(datetime.now(pytz.timezone("Europe/Kyiv")).utcoffset().total_seconds() / 3600 )

KIEV_TZ = pytz.timezone(f'Etc/GMT-{KIEV_UTC_HOURS_DIFF}')
UTC_TZ = pytz.timezone('Etc/GMT')
TZ = lambda: KIEV_TZ

TZ_MAP = {
	f'{KIEV_UTC_HOURS_DIFF}:00:00': KIEV_TZ,
	'0:00:00': UTC_TZ,
}

EMPTY_NETWORK_KEY = 'EMPTY'

NETWORK_KEY = 'NETWORK'
_REGIME = 'REGIME'
_REGIME_EVALUATE = 'EVAL'

FEE_KEY = 'FEE'

_IGNORE = 'IGNORE'
_LONG = 'LONG'
_SHORT = 'SHORT'

_SIGNAL = 'signal'

SIGNAL_IGNORE = 'ignore'
SIGNAL_LONG_IN = 'long_in'
SIGNAL_SHORT_IN = 'short_in'

SIGNAL_LONG_OUT = 'long_out'
SIGNAL_SHORT_OUT = 'short_out'

_DISCRETIZATION = 'discretization'
_TIMESTAMP = 'timestamp'
_UTC_TIMESTAMP = 'utc_timestamp'
_KIEV_TIMESTAMP = 'kiev_timestamp'
_SYMBOL = 'symbol'
_OPEN = 'open'
_HIGH = 'high'
_LOW = 'low'

_CLOSE = 'close'
_CLOSE_NEXT = 'close_next'
_CLOSE_GRAD = lambda order: f'{_CLOSE}_grad_{order}'
_CLOSE_GRAD_NEXT = lambda order: f'{_CLOSE_NEXT}_grad_{order}'

_symbol = _SYMBOL

_BORROWABLE = 'borrowable'

__INCLUDED = lambda threshold: f'included_{threshold}'
__SIGNAL = lambda threshold: f'signal_{threshold}'

__DIFF = lambda threshold: f'diff_{threshold}'
__DIFF_CL = lambda threshold: f'diff_cl_{threshold}'
__DD_CL = lambda threshold: f'dd_cl_{threshold}'

__DIST = lambda threshold: f'dist_{threshold}'
__DIST_CL = lambda threshold: f'dist_cl_{threshold}'

__PTP = lambda threshold, plr: f'ptp_{threshold}_{plr}'
__TPR = lambda threshold: f'tpr_{threshold}'
__TPR_NORM = lambda threshold: f'tpr_norm_{threshold}'

_PLOT_ENABLED = 'PLOT_ENABLED'

MAX_FEATURE_NAN_START_COUNT = lambda: 50

DISCRET_5M = '5M'
DISCRET_15M = '15M'
DISCRET_30M = '30M'

_RESOURCES_FORMAT_JUPYTER = 'jupyter'
_RESOURCES_FORMAT_PLOTLY = 'plotly'

_CPU_COUNT = 'CPU_COUNT'
_WORKERS_COUNT = 'WORKERS_COUNT'
_TRAIN_REGIME = 'TRAIN_REGIME'
_NORM_CONF_MTX = 'NORM_CONF_MTX'
_CLASS_TENSOR_WEIGHTS = 'CLASS_TENSOR_WEIGHTS'
_POS_TENSOR_WEIGHTS = 'POS_TENSOR_WEIGHTS'

_SYMBOL_JOIN = 'SYMBOL'
_SYMBOL_SLASH = 'SYMBOL_SLASH'
_SYMBOL_DASH = 'SYMBOL_DASH'
_COIN_ASSET = 'COIN_ASSET'
_STABLE_ASSET = 'STABLE_ASSET'

_EVAL_NET = 'EVAL_NET'

_NET_FOLDER = 'net_folder'
_PRED_OH_PROB_DTYPE = np.float32

_USE_FILE_PRINTER = 'USE_FILE_PRINTER'
_TRANSFER_CROSS_ISOLATED = 'TRANSFER_CROSS_ISOLATED'
_SHOULD_VALIDATE_GROUP_CONSTRAINTS = 'SHOULD_VALIDATE_GROUP_CONSTRAINTS'

_SKIP_CL = 'SKIP_CL'
_INF_DISCR = 'INF_DISCR'
_PART = 'PART'
_MAX_ORDERS_COUNT = 'MAX_ORDERS_COUNT'
_PROFIT_LOSS_RATIO = 'PROFIT_LOSS_RATIO'
_TAKE_PROFIT_RATIO = 'TAKE_PROFIT_RATIO'
_STOP_LOSS_STOP_RATIO = 'STOP_LOSS_STOP_RATIO'
_INITIAL_STABLE_NET = 'INITIAL_STABLE_NET'
_NO_TRADES_TIMEOUT = 'NO_TRADES_TIMEOUT'
_OVER_TIMEOUT = 'OVER_TIMEOUT'
_RUN_TO_END = 'RUN_TO_END'
_STOP_ON_DROP_DOWN_RATIO = 'STOP_ON_DROP_DOWN_RATIO'
_STOP_ON_JUMP_UP_RATIO = 'STOP_ON_JUMP_UP_RATIO'

_LIMIT_MAKER = 'LIMIT_MAKER'
_STOP_LOSS_LIMIT = 'STOP_LOSS_LIMIT'

_PART_OFFSET = 'PART_OFFSET'
_MARGIN_RATIO = 'MARGIN_RATIO'
_LEVERAGE = 'LEVERAGE'
_TRANSACTION_RETRY_COUNT = 'TRANSACTION_RETRY_COUNT'

_CLEAR_EXISTING_TRADES = 'CLEAR_EXISTING_TRADES'
_USE_PROXY_CLIENT = 'USE_PROXY_CLIENT'
_RUN_DASH_PRESENTER = 'RUN_DASH_PRESENTER'
_RUN_IMAGE_PRESENTER = 'RUN_IMAGE_PRESENTER'
_PORT_MAPPING_FILE_NAME = 'PORT_MAPPING_FILE_NAME'
_INVERTED_TRADES_INFO = 'INVERTED_TRADES_INFO'

_WEIGHTS = 'WEIGHTS'
_WEIGHTS_SUFFIX = 'WEIGHTS_SUFFIX'
_CONFIGS_SUFFIX = 'CONFIGS_SUFFIX'
_EXTREMUMS_SUFFIX = 'EXTREMUMS_SUFFIX'
_MODEL_SUFFIX = 'MODEL_SUFFIX'
_USE_GPU_DATA_PARALLEL = 'USE_GPU_DATA_PARALLEL'
_FINE_TUNE_NET = 'FINE_TUNE_NET'

IS_INVERTED_TRADES_INFO = lambda: False if _INVERTED_TRADES_INFO not in os.environ else string_bool(os.environ[_INVERTED_TRADES_INFO])
DASH_SIMULATE_TRADE_UPDATE_INTERVAL_SECS = lambda: 30 if _DASHBOARD_SEGMENT_BACKTESTING in os.environ[_DASHBOARD_SEGMENT] else 60
INITIAL_BALANCE = lambda: 10 if _INITIAL_STABLE_NET not in os.environ else float(os.environ[_INITIAL_STABLE_NET])
NO_TRADES_TIMEOUT = lambda: timedelta(minutes=3) if _NO_TRADES_TIMEOUT not in os.environ else timedelta(minutes=int(os.environ[_NO_TRADES_TIMEOUT]))
OVER_TIMEOUT = lambda: timedelta(hours=1) if _OVER_TIMEOUT not in os.environ else timedelta(minutes=int(os.environ[_OVER_TIMEOUT]))
NO_CHANGE_RATIO = lambda: 0.05
RUN_TO_END = lambda: False if _RUN_TO_END not in os.environ else string_bool(os.environ[_RUN_TO_END])

GROUP_SEGMENTS_MAX_LENGTH = lambda: int(150 if 'GROUP_SEGMENTS_MAX_LENGTH' not in os.environ else os.environ['GROUP_SEGMENTS_MAX_LENGTH'])

GROUPS_FOLDER_PATH = lambda : f"{project_root_dir()}/GROUPS"
CACHE_FOLDER_PATH = lambda : f"{project_root_dir()}/CACHE"
DATA_FOLDER_PATH = lambda : f"{project_root_dir()}/DATA"
MODEL_FOLDER_PATH = lambda:  f"{project_root_dir()}/MODELS"
CONFIGS_FOLDER_PATH = lambda suffix="": f"{project_root_dir()}/CONFIGS"

CACHED_FOLDER_PARENT_PATH = lambda symbol: f'{DATA_FOLDER_PATH()}/{symbol}'
CACHED_FILE_PATH = lambda symbol, discretization: f'{CACHED_FOLDER_PARENT_PATH(symbol)}/{symbol}-{discretization}.csv'

DISCRETIZATIONS_GROUP_FILE_PATH = lambda discret_s: f"{GROUPS_FOLDER_PATH()}/{'_'.join(discret_s)}.csv"

CONFIGS_FILE_PATH = lambda suffix="": f'{CONFIGS_FOLDER_PATH()}/configs{"" if suffix == "" else f"-{suffix}"}.json'
MODEL_EVALUATION_IMG_PATH = lambda: f'{MODEL_FOLDER_PATH()}/_MODEL_EVAL_IMG'
MARGIN_ACCOUNT_INFO_FILE_PATH = lambda: f"{CACHE_FOLDER_PATH()}/cache__margin_account_info.json"
SYMBOLS_INFO_FILE_PATH = lambda: f"{CACHE_FOLDER_PATH()}/cache__symbols_info.json"
EXCHANGE_INFO_FILE_PATH = lambda: f"{CACHE_FOLDER_PATH()}/cache__exchange_info.json"
EXCHANGE_MP_INFO_FILE_PATH = lambda: f"{CACHE_FOLDER_PATH()}/cache_mp__exchange_info.json"
EXCHANGE_INFO_FUTURES_FILE_PATH = lambda: f"{CACHE_FOLDER_PATH()}/exchange_info_futures.json"
EXCHANGE_INFO_FUTURES_COIN_FILE_PATH = lambda: f"{CACHE_FOLDER_PATH()}/exchange_info_futures_coin.json"

EXTREMUNS_FILE_PATH = lambda suffix="": f'{DATA_FOLDER_PATH()}/___/extremums{"" if suffix == "" else f"-{suffix}"}.json'
WEIGHTS_FILE_PATH = lambda suffix="": f'{DATA_FOLDER_PATH()}/___/weights{"" if suffix == "" else f"-{suffix}"}.json'
WEIGHTS_DIFF_DIST_DATA_MAP_FILE_PATH = lambda: f'{DATA_FOLDER_PATH()}/___/weights_diff_dist_data_map.json'
STAGE4_DATA_MAP_FILE_PATH = lambda suffix='stage4': f'{CONFIGS_FOLDER_PATH()}/data_map-{suffix}.json'

PART_OFFSET = lambda: 1 if _PART_OFFSET not in os.environ or int(os.environ[_PART_OFFSET]) == 0 else int(os.environ[_PART_OFFSET])
MARGIN_RATIO = lambda: 1 if _MARGIN_RATIO not in os.environ else int(os.environ[_MARGIN_RATIO])
LEVERAGE = lambda: 1 if _LEVERAGE not in os.environ else int(os.environ[_LEVERAGE])
TRANSACTION_RETRY_COUNT = lambda: 3 if _TRANSACTION_RETRY_COUNT not in os.environ else int(os.environ[_TRANSACTION_RETRY_COUNT])

OUT_SEGMENT = lambda: 'OUT'


@lru_cache(maxsize=None)
def _SUB_FOLDER_SEGMENT_CACHED(net_folder):
	from SRC.LIBRARIES.new_utils import parse_net_folder_hashed, produce_net_folder

	sub_folder_segment = produce_net_folder(parse_net_folder_hashed(net_folder.replace(" ", "")))

	return sub_folder_segment


def _SUB_FOLDER_SEGMENT(net_folder=None):
	net_folder = net_folder if net_folder is not None else os.environ[_NET_FOLDER] if _NET_FOLDER in os.environ else os.environ['NET_FOLDER']
	sub_folder_segment = _SUB_FOLDER_SEGMENT_CACHED(net_folder)

	return sub_folder_segment

_DASHBOARD_SEGMENT_FULL_PATH = lambda dashboard_segment=None: f"{project_root_dir()}/{OUT_SEGMENT()}/{dashboard_segment if dashboard_segment is not None else os.environ[_DASHBOARD_SEGMENT]}"
_DASHBOARD_SEGMENT_NET_FULL_PATH = lambda dashboard_segment=None, net_folder=None: f"{_DASHBOARD_SEGMENT_FULL_PATH(dashboard_segment)}/{_SUB_FOLDER_SEGMENT(net_folder)}"
_CANDLE_FILE_PATH = lambda dashboard_segment=None, net_folder=None, rel_path="": f"{_DASHBOARD_SEGMENT_NET_FULL_PATH(dashboard_segment=dashboard_segment, net_folder=net_folder)}{rel_path}/candle.json"
_OUT_DISCRETIZATION_FILE_PATH = lambda dashboard_segment=None, net_folder=None, rel_path="": f"{_DASHBOARD_SEGMENT_NET_FULL_PATH(dashboard_segment=dashboard_segment, net_folder=net_folder)}{rel_path}/out_discretizaton.csv"
_STATS_DF_FILE_PATH = lambda dashboard_segment=None, net_folder=None, rel_path="": f"{_DASHBOARD_SEGMENT_NET_FULL_PATH(dashboard_segment=dashboard_segment, net_folder=net_folder)}{rel_path}/stats.csv"
_DATETIME_PRICE_FILE_PATH = lambda dashboard_segment=None, net_folder=None, rel_path="": f"{_DASHBOARD_SEGMENT_NET_FULL_PATH(dashboard_segment=dashboard_segment, net_folder=net_folder)}{rel_path}/price.json"
_DATETIME_PRICES_FILE_PATH = lambda dashboard_segment=None, net_folder=None, rel_path="": f"{_DASHBOARD_SEGMENT_NET_FULL_PATH(dashboard_segment=dashboard_segment, net_folder=net_folder)}{rel_path}/price_s.json"
_BALANCE_FILE_PATH = lambda dashboard_segment=None, net_folder=None, rel_path="": f"{_DASHBOARD_SEGMENT_NET_FULL_PATH(dashboard_segment=dashboard_segment, net_folder=net_folder)}/balance.json"
_UNCOMPLETED_FUTURES_DASHBOARD_TRADES_FILE_PATH = lambda symbol, autotrading_regime: f"{project_root_dir()}/{OUT_SEGMENT()}/{_DASHBOARD_SEGMENT_AUTOTRADING}/__UNCOMPLETED_FUTURES_TRANSACTIONS/{symbol}-{autotrading_regime}.json"
_TRADES_FILE_PATH = lambda dashboard_segment=None, net_folder=None, rel_path="": f"{_DASHBOARD_SEGMENT_NET_FULL_PATH(dashboard_segment=dashboard_segment, net_folder=net_folder)}/trades.txt"
_TRADES_ERRORS_FILE_PATH = lambda dashboard_segment=None, net_folder=None, rel_path="": f"{_DASHBOARD_SEGMENT_NET_FULL_PATH(dashboard_segment=dashboard_segment, net_folder=net_folder)}/trades_erros.txt"
_TRADES_DF_FIG_IMG_FILE_PATH = lambda dashboard_segment=None, net_folder=None, rel_path="": f"{_DASHBOARD_SEGMENT_NET_FULL_PATH(dashboard_segment=dashboard_segment, net_folder=net_folder)}/trades_df.png"
_TRADES_DF_FIG_HTML_FILE_PATH = lambda dashboard_segment=None, net_folder=None, rel_path="": f"{_DASHBOARD_SEGMENT_NET_FULL_PATH(dashboard_segment=dashboard_segment, net_folder=net_folder)}/trades_df.html"
_TRANSACTIONS_INFO_JSON_FILE_PATH = lambda dashboard_segment=None, net_folder=None, rel_path="": f"{_DASHBOARD_SEGMENT_NET_FULL_PATH(dashboard_segment=dashboard_segment, net_folder=net_folder)}/transactions_info.json"
_OCO_ORDER_CHECKER_LOCK_FILE_PATH = lambda dashboard_segment=None, net_folder=None, rel_path="": f"{_DASHBOARD_SEGMENT_NET_FULL_PATH(dashboard_segment=dashboard_segment, net_folder=net_folder)}/oco_checker.lock"
_TOP_UP_NET_FOLDER_PATH = lambda dashboard_segment=None, net_folder=None: f"{_DASHBOARD_SEGMENT_NET_FULL_PATH(dashboard_segment=dashboard_segment, net_folder=net_folder)}/killme"
_TRADES_RUNNER_FILE_PATH = lambda dashboard_segment=None, net_folder=None: f"{_DASHBOARD_SEGMENT_FULL_PATH(dashboard_segment=dashboard_segment)}/runner.txt"


# PORT_MAPPING_FILE_NAME = lambda: 'port_mapping' if _PORT_MAPPING_FILE_NAME not in os.environ else os.environ[_PORT_MAPPING_FILE_NAME]
PORT_MAPPING_FILE_NAME = lambda: 'port_mapping'
USE_PROXY_CLIENT = lambda: True if _USE_PROXY_CLIENT in os.environ and string_bool(os.environ[_USE_PROXY_CLIENT]) else False

TOTAL_ACCOUNT_USDT_EQUITY_FIGURE_IMG_PATH = lambda : f'{project_root_dir()}/TOTAL_ACCOUNT_USDT_EQUITY.png'

_AUTOMATION_FOLDER = lambda dashboard_segment=None: f"{project_root_dir()}/{OUT_SEGMENT()}/{dashboard_segment or os.environ[_DASHBOARD_SEGMENT]}"
_AUTOMATION_STATE_FILE_PATH = lambda db_name, dashboard_segment=None: f"{_AUTOMATION_FOLDER(dashboard_segment)}/state/{db_name}_lock"
_AUTOMATION_STATE_LOCK_FILE_PATH = lambda db_name, dashboard_segment=None: f"{_AUTOMATION_FOLDER(dashboard_segment)}/state/{db_name}"

_RECOMMENDATION_FOLDER_PATH = lambda: f"{project_root_dir()}/{OUT_SEGMENT()}/__TRADINGVIEW_ANALYTICS"
_SYMBOL_TRADINGVIEW_RECOMMENDATION_DF_FILE_PATH = lambda symbol: f"{_RECOMMENDATION_FOLDER_PATH()}/{symbol}.csv"
_MARGIN_LOAN_WATCHER_FOLDER_PATH = lambda: f"{project_root_dir()}/{OUT_SEGMENT()}/__BINANCE_ANALYTICS"
_SYMBOL_MARGIN_LOAN_WATCHER_FILE_PATH = lambda symbol: f"{_MARGIN_LOAN_WATCHER_FOLDER_PATH()}/{symbol}.csv"


_WEBAPP_FOLDER_PATH = lambda: f"{project_root_dir()}/SRC/WEBAPP"
_WEBAPP_TEMPLATE_FOLDER_PATH = lambda: f"{_WEBAPP_FOLDER_PATH()}/templates"
_WEBAPP_STATIC_FOLDER_PATH = lambda: f"{_WEBAPP_FOLDER_PATH()}/static"
_WEBAPP_DYNAMIC_FOLDER_PATH = lambda: f"{project_root_dir()}/OUT/WEBAPP"

_ANALYTICS_SYMBOL_IMAGE_REL_FILE_PATH = lambda symbol: f"analytics/img/{symbol}"
_WEBAPP_ANALYTICS_SYMBOL_IMAGE_FILE_PATH = lambda symbol: f"{_WEBAPP_DYNAMIC_FOLDER_PATH()}/{_ANALYTICS_SYMBOL_IMAGE_REL_FILE_PATH(symbol)}.png"


_MONITORING_SYMBOL_IMAGE_REL_FILE_PATH = lambda symbol, transaction_id: f"monitoring/img/{symbol}/{transaction_id}"
_WEBAPP_MONITORING_SYMBOL_IMAGE_FILE_PATH = lambda symbol, transaction_id: f"{_WEBAPP_DYNAMIC_FOLDER_PATH()}/{_MONITORING_SYMBOL_IMAGE_REL_FILE_PATH(symbol, transaction_id)}.png"
_WEBAPP_MONITORING_SYMBOL_LOCK_FILE_PATH = lambda symbol, transaction_id: _WEBAPP_MONITORING_SYMBOL_IMAGE_FILE_PATH(symbol, transaction_id).replace(".png", ".lock")


_MONITORING_SYMBOL_HTML_REL_FILE_PATH = lambda symbol, transaction_id: f"monitoring/html/{symbol}/{transaction_id}"
_WEBAPP_MONITORING_SYMBOL_HTML_FILE_PATH = lambda symbol, transaction_id: f"{_WEBAPP_STATIC_FOLDER_PATH()}/{_MONITORING_SYMBOL_HTML_REL_FILE_PATH(symbol, transaction_id)}.html"


_DASHBOARD_NETFOLDER_IMAGE_REL_FILE_PATH = lambda net_folder: f"dashboard/img/{net_folder}.png"
_WEBAPP_DASHBOARD_NETFOLDER_IMAGE_FILE_PATH = lambda net_folder: f"{_WEBAPP_DYNAMIC_FOLDER_PATH()}/{_DASHBOARD_NETFOLDER_IMAGE_REL_FILE_PATH(net_folder)}"

_DASHBOARD_NETFOLDER_TRADES_HTML_REL_FILE_PATH = lambda net_folder: f"dashboard/html/{net_folder}.html"
_WEBAPP_DASHBOARD_NETFOLDER_TRADES_HTML_FILE_PATH = lambda net_folder: f"{_WEBAPP_DYNAMIC_FOLDER_PATH()}/{_DASHBOARD_NETFOLDER_TRADES_HTML_REL_FILE_PATH(net_folder)}"


_TRADES_OUT_FOLDER_PATH = f"{project_root_dir()}/OUT/___TRADES"
_PARQUET_PRICE_CHANGES_DATA_PATH = f"{_TRADES_OUT_FOLDER_PATH}/parquet_price_changes"
_PARQUET_PRICE_CHANGES_CHECKPOINT_PATH = f"{_TRADES_OUT_FOLDER_PATH}/checkpoint_price_changes_sink"


_PARQUET_TRADE_TRANSACTIONS_DATA_PATH = f"{_TRADES_OUT_FOLDER_PATH}/parquet_trade_transactions"
_PARQUET_TRADE_TRANSACTIONS_CHECKPOINT_PATH = f"{_TRADES_OUT_FOLDER_PATH}/checkpoint_trades_transactions_sink"


_PARQUET_TRADE_TRANSACTIONS_FAILED_DATA_PATH = f"{_TRADES_OUT_FOLDER_PATH}/parquet_trade_transactions_failed"
_PARQUET_TRADE_TRANSACTIONS_FAILED_CHECKPOINT_PATH = f"{_TRADES_OUT_FOLDER_PATH}/checkpoint_trades_transactions_sink_failed"


_PRICE_CHANGES_REDIS_STREAM_CACHE_FILE_PATH = f"{_TRADES_OUT_FOLDER_PATH}/redis_price_changes_cache.json"
_PRICE_CHANGES_REDIS_STREAM_CACHE_LOCK_FILE_PATH = f"{_TRADES_OUT_FOLDER_PATH}/redis_price_changes_cache.lock"


_PARQUET_PRICE_CHANGES_SYMBOL_TRANSACTION_PARTITION_PATH = lambda symbol, transaction_id: f"{_PARQUET_PRICE_CHANGES_DATA_PATH}/symbol={symbol}/transaction_id={transaction_id}"
_PARQUET_TRADE_TRANSACTIONS_PARTITION_PATH = lambda symbol, transaction_id: f"{_PARQUET_TRADE_TRANSACTIONS_DATA_PATH}/symbol={symbol}/transaction_id={transaction_id}"

_FETCH_CLOSED_TRANSACTION_SYMBOL_TIMESTAMP_MAPPING_LOCK_FILE_PATH = f'{project_root_dir()}/locks/fetch_closed_transaction_symbol_timestamp_mapping.lock'
_PARQUET_CLOSE_TRANSACTION_LOCK_FILE_PATH = lambda : f"{_TRADES_OUT_FOLDER_PATH}/parquet_close_transaction.lock"
_PARQUET_PRICE_CHANGES_SYMBOL_TRANSACTION_PARTITION_LOCK_FILE_PATH = lambda symbol, transaction_id: f"{_TRADES_OUT_FOLDER_PATH}/parquet_price_changes_locks/{transaction_id}.lock"

_SPARK_FETCH_CLOSED_TRANSACTION_SYMBOL_TIMESTAMP_MAPPING__INVALIDATE_CACHE__CHANNEL = 'SPARK_FETCH_CLOSED_TRANSACTION_SYMBOL_TIMESTAMP_MAPPING__INVALIDATE_CACHE__CHANNEL'

_REDIS__SPARK_PRICE_CHANGES_REPARTITION_LOCK_KEY = 'spark_price_changes_repartition_lock'
_REDIS__SPARK_CLOSE_TRANSACTION_LOCK_KEY = 'spark_close_transaction_lock'

_PREDICTION_BROKEN_OUT_FOLDER_PATH = f"{project_root_dir()}/OUT/BROKEN_PREDICTIONs"
_PREDICTION_BROKEN_GROUP_FOLDER_PATH = lambda transaction_id: f"{_PREDICTION_BROKEN_OUT_FOLDER_PATH}/{transaction_id}"
_PREDICTION_BROKEN_DF_FILE_PATH = lambda transaction_id, discretization: f"{_PREDICTION_BROKEN_GROUP_FOLDER_PATH(transaction_id)}/{discretization}.csv"

_TRAINING_PAIRS_MODEL_META_OUT_FOLDER_PATH = lambda net_name: f"{project_root_dir()}/OUT/TRAINING/{net_name}.json"

_SYMBOL_TRANSACTION__MARKET_TYPE__MAPPING_KEY = "SYMBOL_TRANSACTION__MARKET_TYPE__MAPPING_KEY"
_TRANSACTION_SYMBOL_MAPPING_KEY = "TRANSACTION_SYMBOL_MAPPING_KEY"

_PENDING_TRADE_TRANSACTIONS_KEY = "PENDING_TRADE_TRANSACTIONS_KEY"
_TRADING_TRADE_TRANSACTIONS_KEY = "OPENED_TRADE_TRANSACTIONS_KEY"
_CLOSED_TRADE_TRANSACTIONS_KEY = "CLOSED_TRADE_TRANSACTIONS_KEY"

_COMPLETED_TRADE_TRANSACTION_SIGNAL = "COMPLETED_TRADE_TRANSACTION_SIGNAL"
_PROGRESS_PRICE_CHANGE_TRADE_TRANSACTION_SIGNAL = "PROGRESS_PRICE_CHANGE_TRADE_TRANSACTION_SIGNAL"

_OPEN_DASHBOARD_POSITION_REQUEST_CHANNEL = lambda symbol, regime, market_type: f'OPEN-DASHBOARD-POSITION-REQUEST--{symbol}-{regime}-{market_type}'
_CLOSE_DASHBOARD_POSITION_REQUEST_CHANNEL = lambda symbol, regime, market_type: f'CLOSE-DASHBOARD-POSITION-REQUEST--{symbol}-{regime}-{market_type}'

_OPEN_MONITORING_POSITION_REQUEST_CHANNEL = lambda: f'OPEN-MONITORING-POSITION-REQUEST'
_CLOSE_CANCEL_MONITORING_POSITION_REQUEST_CHANNEL = lambda: f'CLOSE_CANCEL-MONITORING-POSITION-REQUEST'


_START_DASHBOARD_CONSUMER_REQUEST_CHANNEL = lambda: f'START-DASHBOARD-CONSUMER-REQUEST-CHANNEL'
_STOP_DASHBOARD_CONSUMER_REQUEST_CHANNEL = lambda: f'STOP-DASHBOARD-CONSUMER-REQUEST-CHANNEL'

_START_SIGNAL_PRODUCER_REQUEST_CHANNEL = lambda: f'START-SIGNAL-PRODUCER-REQUEST-CHANNEL'
_STOP_SIGNAL_PRODUCER_REQUEST_CHANNEL = lambda: f'STOP-SIGNAL-PRODUCER-REQUEST-CHANNEL'

_CORRELATION_SYMBOL_MAPPING_KEY = "CORRELATION_SYMBOL_MAPPING_KEY"
_AUTOTRADE_CORRELATIONS_KEY = "AUTOTRADE_CORRELATIONS_KEY"
_CURRENT_AUTOTRADE_CORRELATIONS_KEY = "CURRENT_AUTOTRADE_CORRELATIONS_KEY"

_SIGNAL_PRODUCER_SESSION_MAP_KEY = 'SIGNAL_PRODUCER_SESSION_MAP_KEY'
_SIGNAL_CONSUMER_SESSION_MAP_KEY = 'SIGNAL_CONSUMER_SESSION_MAP_KEY'

_SIGNAL_PRODUCER_CONFIG_MAP_KEY = lambda unique_session_identifier: f'SIGNAL_PRODUCER_CONFIG_MAP_KEY:{unique_session_identifier}'

_CALLERS_APP_HOST = 'CALLERS_APP_HOST'
_PREDICTION_SERVICE_APP_HOST = 'PREDICTION_SERVICE_APP_HOST'

_WRITE_IGNORE_BALANCES = 'WRITE_IGNORE_BALANCES'

_IS_CLOUD = 'IS_CLOUD'
_IS_REDIS_CLOUD = 'IS_REDIS_CLOUD'
_IS_BINANCE_PROD = 'IS_BINANCE_PROD'

_PRESENTATION_TYPE = 'PRESENTATION_TYPE'
_MONITORING = 'MONITORING'
_DASHBOARD = 'DASHBOARD'

_AUTOMATION_TYPE = 'AUTOMATION_TYPE'
_AUTOTRADING = 'AUTOTRADING'
_BACKTESTING = 'BACKTESTING'

_DASHBOARD_SEGMENT = 'DASHBOARD_SEGMENT'
_DASHBOARD_SEGMENT_AUTOTRADING = _AUTOTRADING
_DASHBOARD_SEGMENT_BACKTESTING = _BACKTESTING

_MARKET_TYPE = 'MARKET_TYPE'
_BACKTEST = 'BACKTEST'
_MARGIN = 'MARGIN'
_FUTURES = 'FUTURES'
_FUTURES_COIN = 'FUTURES_COIN'

_AUTOTRADING_REGIME = 'AUTOTRADING_REGIME'
_MOCK = 'MOCK'
_PROD = 'PROD'

_CALLER = 'CALLER'
_MANUAL = 'MANUAL'
_TRADINGVIEW = 'TRADINGVIEW'
_POSTMAN = 'POSTMAN'

_EXCHANGE = 'EXCHANGE'
_BINANCE = 'BINANCE'
_TESTS = 'TESTS'

_RANDOM = 'RANDOM'
_WORKERS = 'WORKERS'
_FORCE = 'FORCE'

_DEBUG = 'DEBUG'
_NOTICE = 'NOTICE'
_CONSOLE = 'CONSOLE'

_presentation_type = _PRESENTATION_TYPE.lower()
_automation_type = _AUTOMATION_TYPE.lower()
_autotrading_regime = _AUTOTRADING_REGIME.lower()
_market_type = _MARKET_TYPE.lower()
_caller = _CALLER.lower()

_START_AUTOTRADING_REQUEST_CHANNEL = lambda: f'START-AUTOTRADING-REQUEST'
_STOP_AUTOTRADING_REQUEST_CHANNEL = lambda: f'STOP-AUTOTRADING-REQUEST'

_START_BACKTESTING_REQUEST_CHANNEL = lambda: f'START-BACKTESTING-REQUEST'
_STOP_BACKTESTING_REQUEST_CHANNEL = lambda: f'STOP-BACKTESTING-REQUEST'

_BACKTESTING_PROGRESS_SYMBOLS_KEY = 'BACKTESTING_PROGRESS_SYMBOLS_KEY'

_TRADE_STATUS_PENDING = 'PENDING'
_TRADE_STATUS_TRADING = 'TRADING'
_TRADE_STATUS_PROFIT = 'PROFIT'
_TRADE_STATUS_LOSS = 'LOSS'
_TRADE_STATUS_INTERRUPTED = 'INTERRUPTED'
_TRADE_STATUS_BROKEN = 'BROKEN'

_DASHBOARD_CORRELATION_STATUS__ALIVE = 'ALIVE'
_DASHBOARD_CORRELATION_STATUS__DIED = 'DIED'
_DASHBOARD_CORRELATION_STATUS__ERROR = 'ERROR'

_MARGIN_DELIST_SCHEDULE_JSON_FILE_PATH = f"{project_root_dir()}/OUT/delisting_schedule.json"
_MARGIN_DELIST_SCHEDULE_LOCK_FILE_PATH = f"{project_root_dir()}/OUT/delisting_schedule.lock"

_LOGGING_PROCESS_LOCK_FILE_PATH = lambda : f"{project_root_dir()}/locks/logging_{os.getpid()}.lock"

_SIGNAL_ACTIVITY_STATE_KEY = "SIGNAL_ACTIVITY_STATE_KEY"


ALLOWED_IPS_CONFIG_KEY = 'allowed_ips'

def _BROKEN_DF_IDX_FILE_PATH(symbol, discretization, market_type, idx):
	from SRC.LIBRARIES.new_data_utils import _market_type_cryptobot

	return f"{project_root_dir()}/{OUT_SEGMENT()}/BROKEN_IDXs/{symbol}/{discretization}-{_market_type_cryptobot(market_type)}/{str(idx)}.png"

IGNORE_SEPARATOR = '--------'
POSITION_SEPARATOR = '========'
ALERT_SEPARATOR = '########'
EXCEPTION_SEPARATOR = '!!!!!!!!!!!'


__all__ = [
	"_TRADES_OUT_FOLDER_PATH",
	"_PARQUET_PRICE_CHANGES_DATA_PATH",
	"_PARQUET_TRADE_TRANSACTIONS_DATA_PATH",
	"_PARQUET_PRICE_CHANGES_SYMBOL_TRANSACTION_PARTITION_PATH",
	"_PARQUET_TRADE_TRANSACTIONS_PARTITION_PATH",

	"_TRANSACTION_SYMBOL_MAPPING_KEY",
	"_PENDING_TRADE_TRANSACTIONS_KEY",
	"_TRADING_TRADE_TRANSACTIONS_KEY",
	"_CLOSED_TRADE_TRANSACTIONS_KEY",

	"_COMPLETED_TRADE_TRANSACTION_SIGNAL",
	"_PROGRESS_PRICE_CHANGE_TRADE_TRANSACTION_SIGNAL",

	"_OPEN_MONITORING_POSITION_REQUEST_CHANNEL",
	"_CLOSE_CANCEL_MONITORING_POSITION_REQUEST_CHANNEL",

	"_TRADE_STATUS_PENDING",
	"_TRADE_STATUS_TRADING",
	"_TRADE_STATUS_PROFIT",
	"_TRADE_STATUS_LOSS",
	"_TRADE_STATUS_INTERRUPTED",

	"_SIGNAL_ACTIVITY_STATE_KEY",
]
