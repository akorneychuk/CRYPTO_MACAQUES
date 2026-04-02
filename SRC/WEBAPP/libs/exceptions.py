import traceback
from werkzeug.exceptions import HTTPException, Forbidden


class RedisMessageNotSentError(HTTPException):
    def __init__(self, channel):
        self.code = 400
        self.description = f"REDIS MESSAGE NOT SENT: {channel} [Check channel listeners or redis server state]"


class BrokenDataException(Exception):
    def __init__(self, symbol, transaction_id, details=None):
        self.symbol = symbol
        self.transaction_id = transaction_id
        self.details = details

    def __str__(self):
        msg = f"---------------------------------------------"
        msg = f"{msg}\r\n{self.symbol} | {self.transaction_id}"
        if self.details is not None and self.__cause__ is None:
            msg = f"{msg}\r\nCAUSE: {str(self.details)}"
        else:
            msg = f"{msg}\r\nCAUSE: {str(self.__cause__)}"

        tb = traceback.extract_tb(self.__traceback__)
        for frame in tb:
            msg = f"{msg}\r\n{frame.filename}: {frame.lineno}"

        msg = f"{msg}\r\n---------------------------------------------"

        return msg

    def short_presentation(self):
        msg = f"{self.symbol} | {self.transaction_id}"
        if self.details is not None and self.__cause__ is None:
            msg = f"{msg} | CAUSE: {str(self.details)}"
        else:
            msg = f"{msg} | CAUSE: {str(self.__cause__)}"

        return msg


class TradeError(Exception):
    def __init__(self, code, trade_data, binanceEx):
        self.code = code
        self.trade_data = trade_data
        self.binanceEx = binanceEx

    def __str__(self):
        if self.code == 0:
            suffix = "NO BORROWABLE ASSET"
        if self.code == 1:
            suffix = "INSUFFICIENT BALANCE"

        return f"!!! {suffix}\r\ntrade_data: {self.trade_data}\r\nBINANCE CODE {self.binanceEx.code} [{str(self.binanceEx)}] !!!"


class PredictionError(Exception):
    def __init__(self, model_name, symbol, transaction_id, correlation_id, error_msg):
        self.model_name = model_name
        self.symbol = symbol
        self.transaction_id = transaction_id
        self.correlation_id = correlation_id
        self.error_msg = error_msg

    def __str__(self):
        return f"!!! {self.model_name} | {self.symbol} | {self.transaction_id} | {self.correlation_id} !!!\r\n{self.error_msg}"


class PiceNotReachedError(Exception):
    def __init__(self, symbol, side, current_price, target_price):
        self.symbol = symbol
        self.side = side
        self.current_price = current_price
        self.target_price = target_price

    def __str__(self):
        return f"PRICE NOT REACHED: {self.symbol} | {self.side} | current_price: {self.current_price} | target_price: {self.target_price}"


class DuplicatePositionSideError(Exception):
    def __init__(self, caller, signal, symbol, autotrading_regime, market_type, presentation_type):
        self.caller = caller
        self.signal = signal
        self.symbol = symbol
        self.autotrading_regime = autotrading_regime
        self.market_type = market_type
        self.presentation_type = presentation_type

    def __str__(self):
        return f"DUPLICATE POSITION SIDE FORBIDDEN: {self.caller} | {self.signal} | {self.symbol} | {self.autotrading_regime} | {self.market_type} | {self.presentation_type}"


class ChangePositionSideError(Exception):
    def __init__(self, caller, signal, symbol, autotrading_regime, market_type, presentation_type):
        self.caller = caller
        self.signal = signal
        self.symbol = symbol
        self.autotrading_regime = autotrading_regime
        self.market_type = market_type
        self.presentation_type = presentation_type

    def __str__(self):
        return f"CHANGE POSITION SIDE FORBIDDEN: {self.caller} | {self.signal} | {self.symbol} | {self.autotrading_regime} | {self.market_type} | {self.presentation_type}"


class NoTradesInfoError(Exception):
    def __init__(self, net_folder, details):
        self.net_folder = net_folder
        self.details = details

    def __str__(self):
        return f"NO TRADES INFO YET [{self.net_folder}]: >> {self.details}"


class NoActivityError(Exception):
    def __init__(self, net_folder, check_interval):
        self.net_folder = net_folder
        self.check_interval = check_interval

    def __str__(self):
        return f"NO ACTIVITY [{self.net_folder}] >> {self.check_interval}"


class AutotradingSessionNotStartedError(Exception):
    def __init__(self, session_type, start_data, error_message):
        self.session_type = session_type
        self.start_data = start_data
        self.error_message = error_message

    def __str__(self):
        return f"!! AUTOTRADING [{self.session_type}] NOT STARTED !!:\r\n{self.start_data}\r\n{self.error_message}"


class EndDataStreamError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


class ExitAutomationError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg