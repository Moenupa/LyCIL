from .base import BaseIncremental
from .icarl import ICaRL
from .lucir import LUCIR
from .lwf import LWF
from .finetune import FineTune
from .replay import Replay
from .ewc import EWC
from .gem import GEM
from .bic import BiC
__all__ = ["ICaRL", "LUCIR", "LWF", "FineTune", "Replay", "EWC", "GEM", "BiC", "WA", "PODNet", "DER", "PASSV1", "RMM", "BaseIncremental"]

from .wa import WA
from .podnet import PODNet
from .der import DER
from .pass_v1 import PASSV1
from .rmm import RMM
