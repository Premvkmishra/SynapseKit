"""Tests for OrchestrationEvaluator live dashboard integration."""

from __future__ import annotations

from tests.evaluation.orchestration.fixtures import (
    context_loss_run_graph,
    healthy_run_graph,
)

from synapsekit.evaluation.orchestration.evaluator import OrchestrationEvaluator
from synapsekit.live.bus import bus


def test_evaluate_and_publish_when_live_enabled() -> None:
    # Test using the process-wide bus singleton (real object)
    bus.clear()
    bus.enabled = True
    q = bus.subscribe()

    try:
        evaluator = OrchestrationEvaluator()
        graph = healthy_run_graph()
        report = evaluator.evaluate_and_publish(graph, label="Test Run")

        assert report.passed is True

        published_events = []
        while not q.empty():
            published_events.append(q.get_nowait())

        kinds = [e["kind"] for e in published_events]
        assert "orchestration.eval" in kinds
        assert "graph.snapshot" in kinds

        eval_event = next(e for e in published_events if e["kind"] == "orchestration.eval")
        assert eval_event["status"] == "ok"
        assert eval_event["attributes"]["label"] == "Test Run"
        assert eval_event["attributes"]["passed"] is True
    finally:
        bus.unsubscribe(q)
        bus.enabled = False


def test_evaluate_and_publish_when_live_disabled() -> None:
    bus.clear()
    bus.enabled = False

    evaluator = OrchestrationEvaluator()
    graph = context_loss_run_graph()
    report = evaluator.evaluate_and_publish(graph, label="Disabled Run")

    assert report.passed is False
    assert len(bus.history()) == 0  # Zero overhead when off contract
