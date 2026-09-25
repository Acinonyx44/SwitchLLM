"""The company policy: models, roles, task-class quality floors and baseline.

A policy is written once (TOML today; plain-English authoring compiles to the
same structure) and governs every request from every employee.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from .catalog import Effort, ModelSpec, Mode


class PolicyError(ValueError):
    pass


@dataclass(frozen=True)
class Role:
    name: str
    description: str = ""
    max_tier: int = 3
    max_effort: Effort = Effort.HIGH
    allowed_modes: frozenset[Mode] = frozenset(Mode)
    monthly_budget_usd: float | None = None  # per user, per calendar month
    providers: frozenset[str] | None = None  # None = any provider


@dataclass(frozen=True)
class TaskClassRule:
    name: str
    min_tier: int = 1  # quality floor: never route this class below this tier


@dataclass
class ProviderConfig:
    name: str
    type: str
    base_url: str | None = None
    api_key_env: str | None = None
    max_tokens_param: str = "max_completion_tokens"


@dataclass
class Policy:
    models: list[ModelSpec]
    roles: dict[str, Role]
    task_classes: dict[str, TaskClassRule] = field(default_factory=dict)
    default_role: str = "general"
    baseline_model: str = ""
    baseline_effort: Effort = Effort.HIGH
    baseline_mode: Mode = Mode.SINGLE
    confidence_threshold: float = 0.7
    shadow: bool = False
    shadow_execute_routed: bool = False
    group_roles: dict[str, str] = field(default_factory=dict)
    users: dict[str, list[str]] = field(default_factory=dict)
    providers: dict[str, ProviderConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate()

    # -- loading -----------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "Policy":
        with open(path, "rb") as f:
            return cls.from_dict(tomllib.load(f))

    @classmethod
    def example(cls) -> "Policy":
        text = resources.files("switchllm").joinpath("example_policy.toml").read_text()
        return cls.from_dict(tomllib.loads(text))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Policy":
        d = data.get("defaults", {})
        models = [ModelSpec(**m) for m in data.get("models", [])]
        roles = {name: _role(name, r) for name, r in data.get("roles", {}).items()}
        task_classes = {
            name: TaskClassRule(name, int(r.get("min_tier", 1)))
            for name, r in data.get("task_classes", {}).items()
        }
        directory = data.get("directory", {})
        providers = {
            name: ProviderConfig(name=name, **cfg) for name, cfg in data.get("providers", {}).items()
        }
        return cls(
            models=models,
            roles=roles,
            task_classes=task_classes,
            default_role=d.get("default_role", "general"),
            baseline_model=d.get("baseline_model", max(models, key=lambda m: m.tier).id if models else ""),
            baseline_effort=Effort(d.get("baseline_effort", "high")),
            baseline_mode=Mode(d.get("baseline_mode", "single")),
            confidence_threshold=float(d.get("confidence_threshold", 0.7)),
            shadow=d.get("mode", "live") == "shadow",
            shadow_execute_routed=bool(d.get("shadow_execute_routed", False)),
            group_roles=dict(directory.get("group_roles", {})),
            users={u: list(g) for u, g in directory.get("users", {}).items()},
            providers=providers,
        )

    # -- lookups -----------------------------------------------------------

    def model(self, model_id: str) -> ModelSpec:
        for m in self.models:
            if m.id == model_id:
                return m
        raise PolicyError(f"unknown model {model_id!r}")

    @property
    def baseline(self) -> ModelSpec:
        return self.model(self.baseline_model)

    def rule_for(self, task_class: str) -> TaskClassRule:
        return self.task_classes.get(task_class, TaskClassRule(task_class))

    def cheapest_at_tier(self, tier: int, providers: frozenset[str] | None = None) -> ModelSpec | None:
        candidates = [
            m for m in self.models if m.tier == tier and (providers is None or m.provider in providers)
        ]
        return min(candidates, key=lambda m: (m.output_per_mtok, m.input_per_mtok), default=None)

    # -- validation --------------------------------------------------------

    def _validate(self) -> None:
        if not self.models:
            raise PolicyError("policy defines no models")
        ids = [m.id for m in self.models]
        if len(ids) != len(set(ids)):
            raise PolicyError("duplicate model ids")
        for m in self.models:
            if m.tier not in (1, 2, 3):
                raise PolicyError(f"model {m.id!r}: tier must be 1, 2 or 3")
        self.model(self.baseline_model)
        if self.default_role not in self.roles:
            raise PolicyError(f"default_role {self.default_role!r} is not a defined role")
        for group, role in self.group_roles.items():
            if role not in self.roles:
                raise PolicyError(f"group {group!r} maps to undefined role {role!r}")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise PolicyError("confidence_threshold must be between 0 and 1")


def _role(name: str, r: dict[str, Any]) -> Role:
    modes = frozenset(Mode(m) for m in r.get("allowed_modes", [m.value for m in Mode]))
    providers = r.get("providers")
    return Role(
        name=name,
        description=r.get("description", ""),
        max_tier=int(r.get("max_tier", 3)),
        max_effort=Effort(r.get("max_effort", "high")),
        allowed_modes=modes | {Mode.SINGLE},  # a plain call is always allowed
        monthly_budget_usd=r.get("monthly_budget_usd"),
        providers=frozenset(providers) if providers is not None else None,
    )
