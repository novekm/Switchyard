# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observed cost-per-solved-task inputs for the existing weighted-random router.

Eligibility and measurement provenance belong to the host. This is a bounded
empirical estimate for a supplied task class, not a reward learner or a forecast
of savings. Unknown costs, solved labels or latency never become zero values.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

from switchyard.observations import RoutingObservation, RoutingObservations


@dataclass(frozen=True)
class ObservedCostInputs:
    targets: tuple[str, ...]
    weights: tuple[float, ...]
    evidence_json: bytes

    def run_stream(
        self,
        request: Any,
        *,
        headers: Any = None,
        seed: int | None = None,
    ) -> Any:
        """Drive the actual libsy algorithm with the exact eligible target order."""
        from switchyard.libsy import algorithms

        algorithm = algorithms.random(weights=list(self.weights), seed=seed)
        return algorithm.run_stream(request, {"any": list(self.targets)}, headers=headers)


def observed_cost_inputs(
    observations: RoutingObservations,
    *,
    scope_id: str,
    route_id: str,
    eligible_targets: tuple[str, ...],
    task_class: str,
    latency_budget_seconds: float,
    minimum_samples: int,
) -> ObservedCostInputs:
    """Prefer minimum measured total-cost / explicitly solved-count, within latency.

    Every row in a target's selected task-class cohort needs all measurements.
    The host rechecks actual release/policy eligibility after selection. A new
    session consumes one snapshot; this does not migrate an existing session.
    """
    if type(observations) is not RoutingObservations:
        raise TypeError("use the actual routing observation collector")
    if (
        type(eligible_targets) is not tuple
        or not eligible_targets
        or len(set(eligible_targets)) != len(eligible_targets)
        or any(type(name) is not str or not name for name in eligible_targets)
    ):
        raise ValueError("current eligible targets must be a nonempty ordered unique tuple")
    if any(
        type(value) is not str or not 0 < len(value) <= 1000
        for value in (scope_id, route_id, task_class)
    ):
        raise ValueError("scope, route and task class must be explicit bounded identities")
    if (
        type(latency_budget_seconds) not in (int, float)
        or not math.isfinite(latency_budget_seconds)
        or latency_budget_seconds <= 0
        or type(minimum_samples) is not int
        or not 1 <= minimum_samples <= 100_000
    ):
        raise ValueError("latency budget and minimum sample count must be explicit positive bounds")
    rows = tuple(
        RoutingObservation(**row)
        for row in observations.snapshot(scope_id)["observations"]
        if row["route_id"] == route_id and row["task_class"] == task_class
    )
    summaries: dict[str, dict[str, Any]] = {}
    costs: dict[str, float] = {}
    for target in eligible_targets:
        selected = [row for row in rows if row.target_id == target]
        summary: dict[str, Any] = {
            "samples": len(selected),
            "cost_per_solved_task_usd": None,
            "observed_max_elapsed_seconds": None,
            "unavailable": None,
        }
        summaries[target] = summary
        if len(selected) < minimum_samples:
            summary["unavailable"] = "insufficient_samples"
        elif len({row.model for row in selected}) != 1:
            summary["unavailable"] = "model_identity_changed"
        elif any(
            row.cost_usd is None or row.solved is None or row.elapsed_seconds is None
            for row in selected
        ):
            summary["unavailable"] = "incomplete_measurements"
        else:
            maximum = max(row.elapsed_seconds or 0.0 for row in selected)
            summary["observed_max_elapsed_seconds"] = maximum
            solved = sum(bool(row.solved) for row in selected)
            if maximum > latency_budget_seconds:
                summary["unavailable"] = "observed_latency_exceeds_budget"
            elif not solved:
                summary["unavailable"] = "no_explicitly_solved_task"
            else:
                costs[target] = math.fsum(row.cost_usd or 0.0 for row in selected) / solved
                summary["cost_per_solved_task_usd"] = costs[target]
    if not costs:
        raise ValueError("no eligible target has complete cost/solved/latency observations")
    best = min(costs.values())
    # Positive weights only on the empirical minima. libsy owns random tie
    # breaking and restricts fallbacks to positive-weight eligible targets.
    weights = tuple(1.0 if costs.get(target) == best else 0.0 for target in eligible_targets)
    measured = [asdict(row) for row in rows if row.target_id in eligible_targets]

    def canonical(value: Any) -> bytes:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

    evidence = canonical(
        {
            "profile": "switchyard.observed-cost-inputs/1",
            "scope_id": scope_id,
            "route_id": route_id,
            "task_class": task_class,
            "eligible_targets": eligible_targets,
            "latency_budget_seconds": latency_budget_seconds,
            "minimum_samples": minimum_samples,
            "estimator": "observed_total_cost_divided_by_explicitly_solved_count",
            "latency_basis": "observed_max_elapsed_seconds",
            "targets": summaries,
            "source_sha256": hashlib.sha256(canonical(measured)).hexdigest(),
            "algorithm": "weighted_random",
            "weights": weights,
            "algorithm_trained": False,
            "future_performance_guaranteed": False,
        }
    )
    return ObservedCostInputs(eligible_targets, weights, evidence)
