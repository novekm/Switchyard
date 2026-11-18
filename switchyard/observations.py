# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded host-reported routing observations, separate from algorithm decisions.

The host supplies its authenticated execution records. This collector does not
verify execution, infer task success, price missing usage, or update an algorithm.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any


@dataclass(frozen=True)
class RoutingObservation:
    scope_id: str
    route_id: str
    target_id: str
    model: str
    session_id: str
    run_id: str
    selection_id: str
    outcome_id: str | None
    algorithm: str | None
    state: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    elapsed_seconds: float | None = None
    cost_usd: float | None = None
    solved: bool | None = None
    task_class: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "scope_id",
            "route_id",
            "target_id",
            "model",
            "session_id",
            "run_id",
            "selection_id",
        ):
            value = getattr(self, name)
            if type(value) is not str or not 0 < len(value) <= 1000:
                raise ValueError(f"{name} must be a bounded nonempty string")
        for name in ("outcome_id", "algorithm", "task_class"):
            value = getattr(self, name)
            if value is not None and (type(value) is not str or not 0 < len(value) <= 1000):
                raise ValueError(f"{name} must be absent or a bounded nonempty string")
        if self.state not in {"succeeded", "failed", "cancelled", "interrupted"}:
            raise ValueError("routing feedback requires a terminal execution state")
        for name in ("input_tokens", "output_tokens"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or not 0 <= value <= (1 << 63) - 1):
                raise ValueError(f"{name} must be absent or a nonnegative integer")
        for name in ("elapsed_seconds", "cost_usd"):
            value = getattr(self, name)
            if value is not None and (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= 1e18
            ):
                raise ValueError(f"{name} must be absent or finite and nonnegative")
        if self.solved is not None and type(self.solved) is not bool:
            raise ValueError("solved must be an explicit boolean or unknown")


class RoutingObservations:
    """Process-local observations, idempotent by source scope and actual run.

    Full capacity refuses new observations; old identities are never evicted and
    recounted. The host's execution database remains the durable source of truth.
    """

    def __init__(self, *, capacity: int = 4096) -> None:
        if type(capacity) is not int or not 1 <= capacity <= 100_000:
            raise ValueError("observation capacity must be between 1 and 100000")
        self._capacity = capacity
        self._rows: dict[tuple[str, str], RoutingObservation] = {}
        self._lock = Lock()

    def record(self, observation: RoutingObservation) -> bool:
        if type(observation) is not RoutingObservation:
            raise TypeError("use a typed routing observation")
        observation.__post_init__()
        key = (observation.scope_id, observation.run_id)
        with self._lock:
            prior = self._rows.get(key)
            if prior is not None:
                if prior != observation:
                    raise ValueError("the retained execution observation changed")
                return False
            if len(self._rows) >= self._capacity:
                raise ValueError("routing observation capacity reached")
            self._rows[key] = observation
            return True

    def snapshot(self, scope_id: str) -> dict[str, Any]:
        with self._lock:
            rows = tuple(row for row in self._rows.values() if row.scope_id == scope_id)
        targets: dict[str, dict[str, Any]] = {}
        for row in rows:
            stats = targets.setdefault(
                row.target_id,
                {
                    "runs": 0,
                    "states": {},
                    "observed_cost_usd": 0.0,
                    "cost_samples": 0,
                    "elapsed_seconds": 0.0,
                    "elapsed_samples": 0,
                    "solved": 0,
                    "solved_samples": 0,
                    "input_tokens": 0,
                    "input_token_samples": 0,
                    "output_tokens": 0,
                    "output_token_samples": 0,
                },
            )
            stats["runs"] += 1
            stats["states"][row.state] = stats["states"].get(row.state, 0) + 1
            if row.cost_usd is not None:
                stats["observed_cost_usd"] += row.cost_usd
                stats["cost_samples"] += 1
            if row.elapsed_seconds is not None:
                stats["elapsed_seconds"] += row.elapsed_seconds
                stats["elapsed_samples"] += 1
            if row.solved is not None:
                stats["solved_samples"] += 1
                stats["solved"] += row.solved
            for name, samples in (
                ("input_tokens", "input_token_samples"),
                ("output_tokens", "output_token_samples"),
            ):
                if getattr(row, name) is not None:
                    stats[name] += getattr(row, name)
                    stats[samples] += 1
        for stats in targets.values():
            # Unknown cost/task success cannot become a cost-per-solved claim.
            stats["cost_per_solved_task_usd"] = (
                stats["observed_cost_usd"] / stats["solved"]
                if stats["cost_samples"] == stats["solved_samples"] == stats["runs"]
                and stats["solved"] > 0
                else None
            )
            if not stats["cost_samples"]:
                stats["observed_cost_usd"] = None
            if not stats["elapsed_samples"]:
                stats["elapsed_seconds"] = None
            if not stats["solved_samples"]:
                stats["solved"] = None
            if not stats["input_token_samples"]:
                stats["input_tokens"] = None
            if not stats["output_token_samples"]:
                stats["output_tokens"] = None
        return {
            "profile": "switchyard.routing-observations/1",
            "scope_id": scope_id,
            "persistence": "process_local",
            "algorithm_updated": False,
            "targets": targets,
            "observations": [asdict(row) for row in rows],
        }

    def enrich(self, observation: RoutingObservation) -> bool:
        """Attach explicit host-supplied cost/outcome labels to the SAME run.

        The host must independently verify those measurements. Existing source
        identity/metrics and known labels cannot change; missing labels may be
        filled once. This operation does not attest their origin or train a router.
        """
        from dataclasses import replace

        if type(observation) is not RoutingObservation:
            raise TypeError("use a typed routing observation")
        observation.__post_init__()
        key = observation.scope_id, observation.run_id
        with self._lock:
            prior = self._rows.get(key)
            if prior is None:
                raise ValueError("the original execution observation is required")
            labels = {
                name: getattr(observation, name) for name in ("cost_usd", "solved", "task_class")
            }
            if replace(prior, **labels) != observation:
                raise ValueError("execution identity or source metrics changed")
            if any(
                getattr(prior, name) is not None and getattr(prior, name) != value
                for name, value in labels.items()
            ):
                raise ValueError("a known execution measurement cannot be replaced")
            if prior == observation:
                return False
            self._rows[key] = observation
            return True
