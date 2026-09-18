"""OrchestrationEvaluator — run-graph evaluation and live dashboard publishing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from .detectors import ContextLossDetector, LoopDetector, MisroutingDetector
from .report import OrchestrationEvalReport
from .run_graph import RunGraph


class OrchestrationEvaluator:
    """Evaluates RunGraphs for orchestration failure modes (loops, context loss, misrouting).

    Usage::

        evaluator = OrchestrationEvaluator()
        report = evaluator.evaluate(run_graph)
        if not report.passed:
            print(report.to_markdown())
    """

    def __init__(
        self,
        loop_detector: LoopDetector | None = None,
        context_loss_detector: ContextLossDetector | None = None,
        misrouting_detector: MisroutingDetector | None = None,
        thresholds: dict[str, Any] | None = None,
    ) -> None:
        self.loop_detector = loop_detector or LoopDetector()
        self.context_loss_detector = context_loss_detector or ContextLossDetector()
        self.misrouting_detector = misrouting_detector or MisroutingDetector()
        self.thresholds = thresholds or {}

    def evaluate(self, graphs: RunGraph | Sequence[RunGraph]) -> OrchestrationEvalReport:
        """Run orchestration detectors over one or more RunGraph instances."""
        graph_list = [graphs] if isinstance(graphs, RunGraph) else list(graphs)
        run_ids = [g.run_id for g in graph_list]

        findings = []

        # Per-graph detectors
        for g in graph_list:
            findings.extend(self.loop_detector.detect(g))
            findings.extend(self.context_loss_detector.detect(g))

        # Batch misrouting detector across graphs
        if len(graph_list) > 1:
            findings.extend(self.misrouting_detector.detect_nondeterminism(graph_list))

        summary = {
            "loop": sum(1 for f in findings if f.mode == "loop"),
            "context_loss": sum(1 for f in findings if f.mode == "context_loss"),
            "misrouting": sum(1 for f in findings if f.mode == "misrouting"),
        }

        return OrchestrationEvalReport(
            run_ids=run_ids,
            findings=findings,
            thresholds=self.thresholds,
            summary=summary,
        )

    def evaluate_and_publish(
        self,
        graphs: RunGraph | Sequence[RunGraph],
        *,
        label: str = "",
    ) -> OrchestrationEvalReport:
        """Evaluate graphs and publish findings + graph snapshot to SynapseKit Live."""
        report = self.evaluate(graphs)
        graph_list = [graphs] if isinstance(graphs, RunGraph) else list(graphs)

        try:
            from ...live import publish, publish_graph

            top_findings = [asdict(f) for f in report.findings[:5]]
            publish(
                "orchestration.eval",
                status="blocked" if not report.passed else "ok",
                attributes={
                    "label": label,
                    "passed": report.passed,
                    "summary": report.summary,
                    "findings_count": len(report.findings),
                    "run_ids": report.run_ids,
                    "top_findings": top_findings,
                },
            )

            if graph_list:
                nodes, edges = graph_list[0].to_live_graph(findings=report.findings)
                publish_graph(nodes, edges)
        except Exception:  # pragma: no cover
            pass

        return report
