REGISTRY = {}

from .basic_controller import BasicMAC
from .n_controller import NMAC
from .basic_central_controller import CentralBasicMAC
from .ices_n_controller import ICESNMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["n_mac"] = NMAC
REGISTRY["basic_central_mac"] = CentralBasicMAC
REGISTRY["ices_n_mac"] = ICESNMAC


# TODO: delete others except for n_mac and exp_n_mac due to compatibility
