"""Tests for OrchestrationEvaluator and OrchestrationEvalReport."""

from __future__ import annotations

from tests.evaluation.orchestration.fixtures import (
    context_loss_run_graph,
    healthy_run_graph,
    looping_run_graph,
    misrouting_run_graphs,
)

from synapsekit.evaluation.orchestration.evaluator import OrchestrationEvaluator
from synapsekit.evaluation.orchestration.report import OrchestrationEvalReport


def test_evaluator_healthy_graph_passes() -> None:
    evaluator = OrchestrationEvaluator()
    graph = healthy_run_graph()
    report = evaluator.evaluate(graph)

    assert isinstance(report, OrchestrationEvalReport)
    assert report.passed is True
    assert len(report.findings) == 0
    assert report.summary["loop"] == 0
    assert report.summary["context_loss"] == 0
    assert report.summary["misrouting"] == 0


def test_evaluator_mixed_batch() -> None:
    evaluator = OrchestrationEvaluator()
    graphs = [
        healthy_run_graph(),
        looping_run_graph(),
        context_loss_run_graph(),
    ]
    report = evaluator.evaluate(graphs)

    assert report.passed is False
    assert len(report.findings) >= 2
    assert report.summary["loop"] >= 1
    assert report.summary["context_loss"] >= 1


def test_evaluator_misrouting_batch() -> None:
    evaluator = OrchestrationEvaluator()
    graphs = misrouting_run_graphs()
    report = evaluator.evaluate(graphs)

    assert report.passed is False
    assert report.summary["misrouting"] >= 1


def test_report_to_dict_and_to_markdown() -> None:
    evaluator = OrchestrationEvaluator()
    graphs = [looping_run_graph(), context_loss_run_graph()]
    report = evaluator.evaluate(graphs)

    d = report.to_dict()
    assert d["passed"] is False
    assert "summary" in d
    assert "findings" in d
    assert len(d["findings"]) == len(report.findings)

    md = report.to_markdown()
    assert "# Orchestration Evaluation Report" in md
    assert "Infinite Loops" in md
    assert "Context Loss" in md
