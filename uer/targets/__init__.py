from uer.targets.mlm_target import MlmTarget
from uer.targets.target import Target


str2target = {"mlm": MlmTarget}

__all__ = ["Target", "MlmTarget", "str2target"]
