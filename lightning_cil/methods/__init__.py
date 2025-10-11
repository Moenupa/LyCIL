from .base import BaseIncremental
from .icarl import ICaRL
from .lucir import LUCIR
from .lwf import LWF
from .finetune import FineTune
from .replay import Replay
from .ewc import EWC
from .gem import GEM
from .bic import BiC
__all__ = ["BaseIncremental", "ICaRL", "LUCIR", "LWF"]
