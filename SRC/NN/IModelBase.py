import os
from abc import ABC
from abc import abstractmethod
from datetime import datetime

from SRC.CORE._CONSTANTS import UTC_TZ, _MARGIN, _BINANCE_FEE, _LONG, _SHORT
from SRC.LIBRARIES.new_data_utils import fetch_featurize_realtime_group_all
from SRC.LIBRARIES.time_utils import TIME_DELTA, round_down_to_nearest_step


class IModelBase(ABC):
    @abstractmethod
    def predict_(self, group):
        pass

    @abstractmethod
    def segments_count(self) -> int:
        pass

    @abstractmethod
    def discretization_s(self) -> []:
        pass

    @abstractmethod
    def inference_discretization_s(self) -> []:
        pass

    @abstractmethod
    def discretization_feature_s(self) -> []:
        pass

    @abstractmethod
    def discretization_meta_feature_s(self) -> []:
        pass

    def produce_rped_tpr(self, prediction):
        signal, take_profit_ratio = prediction['signal'], prediction['take_profit_ratio']
        rel_tpr = take_profit_ratio - 1
        if signal == _LONG:
            pred_tpr = rel_tpr

            return pred_tpr
        if signal == _SHORT:
            pred_tpr = -rel_tpr

            return pred_tpr

        return 0

    def predict_signal(self, group):
        prediction = self.predict_(group)
        prediction['pred_tpr'] = self.produce_rped_tpr(prediction)

        return prediction

    @staticmethod
    def test_single_inference(net):
        # symbols = "DOGEUSDT|NKNUSDT|XECUSDT|THETAUSDT|DEGOUSDT|GNSUSDT|KMNOUSDT|SLPUSDT|TSTUSDT|PEOPLEUSDT|PHBUSDT"
        symbols = "DOGEUSDT"
        symbol_s = symbols.split("|")
        os.environ[_BINANCE_FEE] = '0.018 - 0.036'
        os.environ['STRATEGY'] = os.environ['STRATEGY'] if 'STRATEGY' in os.environ else '1'

        inference_discretization_s = net.inference_discretization_s()
        segments = net.segments_count()

        for symbol in symbol_s:
            input_group = fetch_featurize_realtime_group_all(_MARGIN, symbol, inference_discretization_s, segments)

            pred_label = net.predict_signal(input_group)
            print(f"{symbol} >> {pred_label}")


def produce_model(model_name) -> IModelBase:
    return __import__(f"SRC.NN.{model_name}", fromlist=['NN']).NN()
