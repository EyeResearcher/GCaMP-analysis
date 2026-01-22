from .corr_strategy import CorrelationStrategy
from .dtw_strategy import DTWStrategy
from .sttc_strategy import STTCStrategy

STRATEGY_REGISTRY = {
    "sttc": STTCStrategy,
    "dtw": DTWStrategy,
    "corr": CorrelationStrategy,
}
