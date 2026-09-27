"""SwitchLLM: role-based routing of every task to the right model, effort and mode."""

from .catalog import Effort, ModelSpec, Mode
from .classifier import TaskProfile, classify
from .engine import RunResult, SwitchLLM
from .policy import Policy, PolicyError, Role
from .router import RouteDecision, route

__all__ = [
    "Effort", "ModelSpec", "Mode", "Policy", "PolicyError", "Role", "RouteDecision", "RunResult",
    "SwitchLLM", "TaskProfile", "classify", "route",
]
