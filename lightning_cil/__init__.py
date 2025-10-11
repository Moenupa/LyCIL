from .methods.icarl import ICaRL
from .methods.lucir import LUCIR
from .methods.lwf import LWF
from .methods.finetune import FineTune
from .methods.replay import Replay
from .methods.ewc import EWC
from .methods.gem import GEM
from .methods.bic import BiC
from .methods.base import BaseIncremental
__all__ = ["ICaRL", "LUCIR", "LWF", "FineTune", "Replay", "EWC", "GEM", "BiC", "WA", "PODNet", "DER", "PASSV1", "RMM", "BaseIncremental"]

from .methods.wa import WA
from .methods.podnet import PODNet
from .methods.der import DER
from .methods.pass_v1 import PASSV1
from .methods.rmm import RMM
