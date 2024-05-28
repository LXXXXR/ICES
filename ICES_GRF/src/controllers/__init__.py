from .basic_controller import BasicMAC
from .ices_controller import ICESMAC

REGISTRY = {}


REGISTRY["basic_mac"] = BasicMAC
REGISTRY["ices_mac"] = ICESMAC
