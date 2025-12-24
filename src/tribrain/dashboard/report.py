from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DashboardSummary:
    run_dir: Path
    iters: int
    best_score: float
    best_iter: int
    total_llm_calls: int
    total_llm_tokens: int
    total_iter_time_s: float


def load_trace_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def summarize_trace(rows: list[dict], run_dir: Path) -> DashboardSummary:
    if not rows:
        return DashboardSummary(run_dir=run_dir, iters=0, best_score=0.0, best_iter=-1, total_llm_calls=0, total_llm_tokens=0, total_iter_time_s=0.0)
    best = max(rows, key=lambda r: float(r.get("overall_score", 0.0)))
    budget = rows[-1].get("meta", {}).get("budget", {})
    return DashboardSummary(
        run_dir=run_dir,
        iters=len(rows),
        best_score=float(best.get("overall_score", 0.0)),
        best_iter=int(best.get("iter_idx", -1)),
        total_llm_calls=int(budget.get("llm_calls", rows[-1].get("meta", {}).get("llm_calls", 0))),
        total_llm_tokens=int(budget.get("llm_tokens_used", rows[-1].get("meta", {}).get("llm_tokens_used", 0))),
        total_iter_time_s=float(budget.get("total_iter_time_s", rows[-1].get("meta", {}).get("total_iter_time_s", 0.0))),
    )


def write_markdown(summary: DashboardSummary, rows: list[dict], out_path: Path) -> None:
    lines = []
    lines.append("# TriBrain Run Report\n")
    lines.append(f"- Run dir: `{summary.run_dir}`")
    lines.append(f"- Iterations: **{summary.iters}**")
    lines.append(f"- Best score: **{summary.best_score:.4f}** (iter {summary.best_iter})")
    lines.append(f"- LLM calls (total): **{summary.total_llm_calls}**")
    lines.append(f"- LLM tokens (total): **{summary.total_llm_tokens}**")
    lines.append(f"- Total iteration time (s): **{summary.total_iter_time_s:.1f}**\n")

    lines.append("## Per-iteration\n")
    lines.append("| iter | score | arm | drift | plateau | stop_reason | artifact |")
    lines.append("|---:|---:|---|---|---:|---|---|")
    for r in rows:
        meta = r.get("meta", {})
        stop = str(meta.get("stop_reason", ""))
        arm = str(meta.get("arm", meta.get("chosen_arm", "")))
        drift = "✅" if bool(meta.get("drift_flag", False)) else ""
        plateau = int(meta.get("plateau_steps", 0))
        art = str(r.get("artifact_path", ""))[-60:]
        lines.append(f"| {int(r.get('iter_idx',0))} | {float(r.get('overall_score',0.0)):.4f} | {arm} | {drift} | {plateau} | {stop} | `{art}` |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def try_plot(rows: list[dict], out_dir: Path) -> None:
    """Optional: save simple plots if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-untyped]
    except Exception:
        return
    it = [int(r.get("iter_idx", 0)) for r in rows]
    sc = [float(r.get("overall_score", 0.0)) for r in rows]
    plt.figure()
    plt.plot(it, sc, marker="o")
    plt.xlabel("iteration")
    plt.ylabel("overall_score")
    plt.title("TriBrain: score vs iteration")
    plt.savefig(out_dir / "score.png", dpi=160, bbox_inches="tight")
    plt.close()


def build_dashboard(run_dir: Path, out_dir: Path | None = None) -> Path:
    run_dir = Path(run_dir).resolve()
    trace = run_dir / "trace.jsonl"
    if not trace.exists():
        raise FileNotFoundError(f"trace not found: {trace}")
    out_dir = Path(out_dir or (run_dir / "dashboard")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_trace_jsonl(trace)
    summary = summarize_trace(rows, run_dir)
    md = out_dir / "REPORT.md"
    write_markdown(summary, rows, md)
    try_plot(rows, out_dir)
    return md
