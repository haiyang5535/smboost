"""Render gate results as Markdown for commit to docs/overnight/."""
from __future__ import annotations

from benchmarks.gates.criteria import GateResult


_PIVOT_BY_GATE = {
    "G1_capability_floor": (
        "Pivot options on G1 failure:\n"
        "- raw_2b < 30%: fall back to plain HumanEval (weaker-test variant) as primary.\n"
        "- raw_9b underperforms: debug quantization/backend; if unfixable, use Qwen 3.5 4B as upper-bound.\n"
    ),
    "G2_harness_lift": (
        "Pivot options on G2 failure:\n"
        "- C1 does not meaningfully beat raw 2B: **Hero-model pivot** to 0.8B hero (lower baseline → more dramatic lift expected).\n"
        "- C1 ≈ C4: **Metric pivot** — switch headline from pass rate to a reliability metric "
        "(malformed-output recovery rate, tool-call retry success).\n"
    ),
    "G3_bfcl_sanity": (
        "Pivot options on G3 failure:\n"
        "- raw_2b near-zero on BFCL: drop BFCL from headline; retain only as demo video scene 2.\n"
    ),
    "G4_widen_confidence": (
        "Pivot options on G4 failure:\n"
        "- Variance too wide: ship with fewer conditions (C1 vs C4 only) and document as v0.1.0 with three-seed follow-up promised.\n"
    ),
}


def _fmt_metrics(metrics: dict) -> str:
    if not metrics:
        return "_(no metrics)_"
    lines = []
    for k, v in metrics.items():
        if isinstance(v, float) and "_rate" in k:
            lines.append(f"- **{k}**: {v:.1%}")
        elif isinstance(v, float):
            lines.append(f"- **{k}**: {v:.3f}")
        else:
            lines.append(f"- **{k}**: {v}")
    return "\n".join(lines)


def render_gate_report(results: list[GateResult], *, run_date: str) -> str:
    lines: list[str] = [f"# Gate Results — {run_date}", ""]

    overall_pass = all(r.passed for r in results)
    lines.append(f"**Overall decision**: {'PROCEED to full run' if overall_pass else 'PIVOT — do not launch full run'}")
    lines.append("")

    for r in results:
        verdict = "**PASS**" if r.passed else "**FAIL**"
        lines.append(f"## {r.name} — {verdict}")
        lines.append("")
        lines.append("### Metrics")
        lines.append(_fmt_metrics(r.metrics))
        lines.append("")
        if r.failed_checks:
            lines.append("### Failed checks")
            for c in r.failed_checks:
                lines.append(f"- {c}")
            lines.append("")
            pivot = _PIVOT_BY_GATE.get(r.name, "")
            if pivot:
                lines.append("### Pivot options")
                lines.append(pivot)
                lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)
