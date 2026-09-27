"""Identity: who is asking, and which role does that make them?

Production deployments sync groups from Okta, Entra ID, Slack or Glean; the
static directory reads the same user -> groups mapping from the policy file.
"""

from __future__ import annotations

from typing import Protocol

from .policy import Policy, Role


class RoleDirectory(Protocol):
    def groups_for(self, user: str) -> list[str]: ...


class StaticDirectory:
    def __init__(self, users: dict[str, list[str]]):
        self._users = {u.lower(): groups for u, groups in users.items()}

    def groups_for(self, user: str) -> list[str]:
        return list(self._users.get(user.lower(), []))


def resolve_role(policy: Policy, directory: RoleDirectory, user: str) -> Role:
    """Map a user to a role. The first matching entry in the policy's
    group_roles table wins, so admins control precedence by ordering it."""
    groups = set(directory.groups_for(user))
    for group, role in policy.group_roles.items():
        if group in groups:
            return policy.roles[role]
    return policy.roles[policy.default_role]
