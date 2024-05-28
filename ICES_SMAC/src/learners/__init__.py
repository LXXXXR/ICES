from .q_learner import QLearner
from .nq_learner import NQLearner
from .ices_nq_learner import ICESNQLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["nq_learner"] = NQLearner
REGISTRY["ices_nq_learner"] = ICESNQLearner
