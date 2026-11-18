# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decision-only per-call use of the existing Rust-owned algorithms.

The host owns binding identities, current eligibility and provider forwarding.
This module does not receive credentials or turn a decision into authorization.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from switchyard.libsy import Step, algorithms


@dataclass(frozen=True)
class CallDecision:
    target: str
    outcome_id: str | None
    algorithm: str | None
    request_sha256: str


@dataclass(frozen=True)
class PerCallRoute:
    route_id: str
    targets: tuple[str, ...]
    strategy: str = "random"
    weights: tuple[float, ...] = ()
    efficient: str | None = None
    capable: str | None = None
    picker: str = "efficient_first"
    confidence_threshold: float = 0.7

    def __post_init__(self) -> None:
        if (
            type(self.route_id) is not str
            or not self.route_id
            or type(self.targets) is not tuple
            or not self.targets
            or any(type(t) is not str or not t for t in self.targets)
            or len(set(self.targets)) != len(self.targets)
        ):
            raise ValueError("route and unique declared targets are required")
        if self.strategy not in {"random", "stage"}:
            raise ValueError("per-call profile supports native random or stage")
        if self.strategy == "stage" and (
            self.efficient not in self.targets
            or self.capable not in self.targets
            or self.efficient == self.capable
        ):
            raise ValueError("stage requires distinct declared efficient and capable targets")
        if type(self.weights) is not tuple or (
            self.weights
            and (
                len(self.weights) != len(self.targets)
                or not any(self.weights)
                or any(
                    type(w) not in (int, float) or not math.isfinite(w) or w < 0
                    for w in self.weights
                )
            )
        ):
            raise ValueError("weights must match the declared targets and be finite/nonnegative")
        if (
            type(self.confidence_threshold) not in (int, float)
            or not math.isfinite(self.confidence_threshold)
            or not 0 <= self.confidence_threshold <= 1
        ):
            raise ValueError("stage confidence must be finite and between zero and one")

    async def decide(
        self, normalized_request: dict[str, Any], eligible: tuple[str, ...]
    ) -> CallDecision:
        """Choose for this call, without reusing a previous session decision.

        A fresh native algorithm instance prevents algorithm session affinity
        from becoming an implicit physical-model lock for subsequent calls.
        """
        if (
            not eligible
            or len(set(eligible)) != len(eligible)
            or not set(eligible) <= set(self.targets)
        ):
            raise ValueError("current eligible targets must be a nonempty declared subset")
        request = json.loads(json.dumps(normalized_request, allow_nan=False))
        request["model"] = self.route_id
        request.pop("metadata", None)
        models = {"any": list(eligible)}
        selectable = set(eligible)
        if self.strategy == "stage":
            efficient, capable = self.efficient, self.capable
            if efficient is None or capable is None or not {efficient, capable} <= set(eligible):
                raise ValueError("current policy/capabilities do not admit both stage roles")
            models.update(efficient=[efficient], capable=[capable])
            algorithm = algorithms.stage_router(
                picker=self.picker,
                confidence_threshold=self.confidence_threshold,
            )
        else:
            weights = self.weights or tuple(1.0 for _ in self.targets)
            selected_weights = [weights[self.targets.index(name)] for name in eligible]
            if not any(selected_weights):
                raise ValueError("no currently eligible target has positive routing weight")
            selectable = {
                name for name, weight in zip(eligible, selected_weights, strict=True) if weight > 0
            }
            algorithm = algorithms.random(weights=selected_weights)
        stream = algorithm.run_stream(request, models)
        try:
            async for step in stream:
                match step:
                    case Step.CallModel(_):
                        raise RuntimeError(
                            "decision-only per-call profile requested an unapproved judge"
                        )
                    case Step.Done(outcome):
                        chosen = outcome.selected_model_ids
                        if not chosen or any(name not in selectable for name in chosen):
                            raise RuntimeError("native decision escaped the current eligible set")
                        metadata = outcome.metadata
                        return CallDecision(
                            chosen[0],
                            metadata.outcome_id if metadata else None,
                            metadata.algorithm if metadata else None,
                            hashlib.sha256(
                                json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
                            ).hexdigest(),
                        )
            raise RuntimeError("native algorithm ended without a routing decision")
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()
