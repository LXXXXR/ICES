from .rnn_agent import RNNAgent
from .ices_agent import ICESAgent

REGISTRY = {}


REGISTRY["rnn"] = RNNAgent
REGISTRY["ices"] = ICESAgent
