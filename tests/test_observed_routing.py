"""Synthetic measurements test the consumer, not measured routing gains."""

import json
from dataclasses import replace

import pytest

from switchyard.observations import RoutingObservation, RoutingObservations
from switchyard.observed_routing import observed_cost_inputs


async def test_explicit_outcome_enrichment_drives_existing_weighted_random():
    from switchyard.libsy import Step

    Observation, Store, inputs = RoutingObservation, RoutingObservations, observed_cost_inputs
    store = Store()
    row = Observation(
        scope_id="fixture-scope",
        route_id="route",
        target_id="affordable",
        model="model-a",
        session_id="fixture-session",
        run_id="run-a",
        selection_id="source-a",
        outcome_id=None,
        algorithm="random",
        state="succeeded",
        elapsed_seconds=2,
    )
    store.record(row)
    args = {
        "scope_id": row.scope_id,
        "route_id": row.route_id,
        "task_class": "fixture-review",
        "latency_budget_seconds": 120,
        "minimum_samples": 1,
    }
    with pytest.raises(ValueError, match="complete"):
        inputs(store, eligible_targets=("affordable",), **args)
    measured = replace(row, cost_usd=2.0, solved=True, task_class="fixture-review")
    with pytest.raises(ValueError, match="identity"):
        store.enrich(replace(measured, model="another-model"))
    assert store.enrich(measured) and not store.enrich(measured)
    with pytest.raises(ValueError, match="cannot be replaced"):
        store.enrich(replace(measured, cost_usd=0.0))
    # An unsuccessful task's observed cost remains in the numerator.
    store.record(replace(measured, run_id="run-a-failed", state="failed", solved=False))
    store.record(
        replace(measured, target_id="expensive", model="model-b", run_id="run-b", cost_usd=5.0)
    )
    store.record(
        replace(
            measured,
            target_id="too-slow",
            model="model-c",
            run_id="run-c",
            cost_usd=1.0,
            elapsed_seconds=121,
        )
    )
    store.record(
        replace(measured, target_id="unknown-cost", model="model-d", run_id="run-d", cost_usd=None)
    )
    routed = inputs(
        store, eligible_targets=("affordable", "expensive", "too-slow", "unknown-cost"), **args
    )
    evidence = json.loads(routed.evidence_json)
    assert routed.weights == (1.0, 0.0, 0.0, 0.0)
    assert evidence["targets"]["affordable"]["cost_per_solved_task_usd"] == 4.0
    assert evidence["targets"]["unknown-cost"]["unavailable"] == "incomplete_measurements"
    assert evidence["targets"]["too-slow"]["unavailable"] == "observed_latency_exceeds_budget"
    assert (
        evidence["algorithm_trained"] is False
        and evidence["future_performance_guaranteed"] is False
    )
    async for step in routed.run_stream(
        {"model": "route", "messages": [{"role": "user", "content": []}]}, seed=42
    ):
        match step:
            case Step.CallModel(_):
                raise AssertionError("weighted random requested a model")
            case Step.Done(outcome):
                assert list(outcome.selected_model_ids) == ["affordable"]
                break
    else:
        raise AssertionError("native algorithm produced no routing outcome")
    # Measured eligibility cannot expand the host's current eligible set.
    only_other = inputs(store, eligible_targets=("expensive",), **args)
    assert only_other.targets == ("expensive",) and only_other.weights == (1.0,)
    with pytest.raises(ValueError, match="nonempty"):
        inputs(store, eligible_targets=(), **args)


def test_task_class_scope_and_missing_measurements_refuse():
    Observation, Store, inputs = RoutingObservation, RoutingObservations, observed_cost_inputs
    store = Store()
    store.record(
        Observation(
            scope_id="fixture-owner-one",
            route_id="route",
            target_id="target",
            model="m",
            session_id="session",
            run_id="run",
            selection_id="source",
            outcome_id=None,
            algorithm="random",
            state="succeeded",
            cost_usd=3,
            solved=True,
            task_class="fixture-review",
            elapsed_seconds=2,
        )
    )
    args = {
        "route_id": "route",
        "eligible_targets": ("target",),
        "latency_budget_seconds": 120,
        "minimum_samples": 1,
    }
    for scope, kind in (
        ("fixture-owner-two", "fixture-review"),
        ("fixture-owner-one", "another-class"),
    ):
        with pytest.raises(ValueError, match="complete"):
            inputs(store, scope_id=scope, task_class=kind, **args)
