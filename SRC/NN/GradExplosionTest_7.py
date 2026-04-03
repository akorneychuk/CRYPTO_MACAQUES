from SRC.CORE._CONSTANTS import _LONG, _SHORT, _IGNORE
from SRC.NN.S_Base import S_Base


class NN(S_Base):
    def predict_(self, group):
        volume_grad_diff_9_0 = group[0].iloc[-1]['volume_grad_diff_9']

        if volume_grad_diff_9_0 > 0.5:
            return {
                'signal': _LONG,
                'take_profit_ratio': 1.005,
            }

        return {
            'signal': _IGNORE,
            'take_profit_ratio': 0
        }

    def segments_count(self) -> int:
        return 20

    def discretization_feature_s(self):
        return {
            '15M': {
                'volume_grad_diff_9'
            }
        }