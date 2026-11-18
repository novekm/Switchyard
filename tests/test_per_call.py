# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-call decisions use real native libsy, without a provider transport."""

import pytest

from switchyard.per_call import PerCallRoute


async def test_each_call_consumes_only_its_current_eligible_set():
    route = PerCallRoute("declared-pool", ("nim", "bedrock"))
    request = {"messages": [{"role": "user", "content": [{"type": "text", "text": "task"}]}]}
    first = await route.decide(request, ("nim",))
    second = await route.decide(request, ("bedrock",))
    assert (first.target, second.target) == ("nim", "bedrock")
    assert first.outcome_id and second.outcome_id and first.outcome_id != second.outcome_id
    assert first.algorithm and first.request_sha256 == second.request_sha256


async def test_native_stage_preserves_request_and_requires_both_approved_roles():
    route = PerCallRoute(
        "declared-pool", ("nim", "bedrock"), strategy="stage", efficient="nim", capable="bedrock"
    )
    request = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]}
    first = await route.decide(request, ("nim", "bedrock"))
    assert first.target == "nim" and "model" not in request
    with pytest.raises(ValueError, match="both stage roles"):
        await route.decide(request, ("nim",))


async def test_no_empty_unapproved_or_zero_weight_fallback():
    route = PerCallRoute("declared-pool", ("nim", "bedrock"), weights=(1, 0))
    request = {"messages": []}
    for eligible in ((), ("unapproved",), ("nim", "nim")):
        with pytest.raises(ValueError, match="eligible"):
            await route.decide(request, eligible)
    with pytest.raises(ValueError, match="positive"):
        await route.decide(request, ("bedrock",))
