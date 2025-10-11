from .methods.icarl import ICaRL
from .methods.lucir import LUCIR
from .methods.lwf import LWF
from .methods.finetune import FineTune
from .methods.replay import Replay
from .methods.ewc import EWC
from .methods.gem import GEM
from .methods.bic import BiC
from .methods.base import BaseIncremental
__all__ = ["ICaRL", "LUCIR", "LWF", "BaseIncremental"]
