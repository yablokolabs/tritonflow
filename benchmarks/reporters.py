"""Reporting and export utilities for benchmark results."""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from itertools import groupby
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmarks.framework import BenchmarkResult

__all__ = ["BenchmarkReporter"]

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _device_summary() -> str:
    """Return a short string describing the compute device."""
    if _HAS_TORCH and torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        return f"{name} ({mem:.1f} GB)"
    return "CPU (no CUDA)"


class BenchmarkReporter:
    """Generate reports from benchmark results."""

    def __init__(self, results: list[BenchmarkResult]):
        self.results = results

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------

    def to_markdown(self) -> str:
        """Generate a markdown benchmark report with tables."""
        lines: list[str] = []
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        lines.append("# Benchmark Report\n")
        lines.append(f"**Date:** {ts}  ")
        lines.append(f"**Device:** {_device_summary()}  \n")

        sorted_results = sorted(self.results, key=lambda r: r.category)
        for category, group in groupby(sorted_results, key=lambda r: r.category):
            lines.append(f"## {category}\n")
            lines.append(
                "| Name | Size | Dtype | Triton (ms) | PyTorch (ms) | Speedup | Memory |"
            )
            lines.append("|------|------|-------|-------------|--------------|---------|--------|")
            for r in group:
                pt = f"{r.pytorch_ms:.3f}" if r.pytorch_ms is not None else "—"
                sp = f"{r.speedup_vs_pytorch:.2f}×" if r.speedup_vs_pytorch is not None else "—"
                mem = _fmt_bytes(r.memory_bytes) if r.memory_bytes is not None else "—"
                lines.append(
                    f"| {r.name} | {r.input_size} | {r.dtype} "
                    f"| {r.triton_ms:.3f} | {pt} | {sp} | {mem} |"
                )
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    def to_csv(self, path: str) -> None:
        """Export results to CSV."""
        fieldnames = [
            "name",
            "category",
            "input_size",
            "dtype",
            "device",
            "triton_ms",
            "pytorch_ms",
            "numpy_ms",
            "speedup_vs_pytorch",
            "speedup_vs_numpy",
            "memory_bytes",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.results:
                writer.writerow(
                    {
                        "name": r.name,
                        "category": r.category,
                        "input_size": r.input_size,
                        "dtype": r.dtype,
                        "device": r.device,
                        "triton_ms": f"{r.triton_ms:.4f}",
                        "pytorch_ms": f"{r.pytorch_ms:.4f}" if r.pytorch_ms is not None else "",
                        "numpy_ms": f"{r.numpy_ms:.4f}" if r.numpy_ms is not None else "",
                        "speedup_vs_pytorch": (
                            f"{r.speedup_vs_pytorch:.4f}"
                            if r.speedup_vs_pytorch is not None
                            else ""
                        ),
                        "speedup_vs_numpy": (
                            f"{r.speedup_vs_numpy:.4f}"
                            if r.speedup_vs_numpy is not None
                            else ""
                        ),
                        "memory_bytes": r.memory_bytes if r.memory_bytes is not None else "",
                    }
                )

    # ------------------------------------------------------------------
    # Rich console table
    # ------------------------------------------------------------------

    def to_rich_table(self) -> None:
        """Print results using rich Table."""
        from rich.console import Console
        from rich.table import Table

        console = Console()
        sorted_results = sorted(self.results, key=lambda r: r.category)

        for category, group in groupby(sorted_results, key=lambda r: r.category):
            table = Table(title=f"Benchmarks: {category}")
            table.add_column("Name", style="cyan")
            table.add_column("Size", style="green")
            table.add_column("Dtype")
            table.add_column("Triton (ms)", justify="right", style="bold")
            table.add_column("PyTorch (ms)", justify="right")
            table.add_column("Speedup", justify="right", style="magenta")
            table.add_column("Memory", justify="right")

            for r in group:
                pt = f"{r.pytorch_ms:.3f}" if r.pytorch_ms is not None else "—"
                sp = f"{r.speedup_vs_pytorch:.2f}×" if r.speedup_vs_pytorch is not None else "—"
                mem = _fmt_bytes(r.memory_bytes) if r.memory_bytes is not None else "—"
                table.add_row(
                    r.name,
                    r.input_size,
                    r.dtype,
                    f"{r.triton_ms:.3f}",
                    pt,
                    sp,
                    mem,
                )

            console.print(table)
            console.print()

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------

    def generate_charts(self, output_dir: str = "results") -> None:
        """Generate matplotlib benchmark charts (PNG).

        Silently skips if matplotlib is not installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        os.makedirs(output_dir, exist_ok=True)

        # --- Bar chart: Triton vs PyTorch per kernel ---
        paired = [r for r in self.results if r.pytorch_ms is not None]
        if paired:
            names = [f"{r.name}\n({r.input_size})" for r in paired]
            triton_times = [r.triton_ms for r in paired]
            pytorch_times = [r.pytorch_ms for r in paired]

            x = range(len(names))
            width = 0.35
            fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 6))
            ax.bar([i - width / 2 for i in x], triton_times, width, label="Triton")
            ax.bar([i + width / 2 for i in x], pytorch_times, width, label="PyTorch")
            ax.set_ylabel("Time (ms)")
            ax.set_title("Triton vs PyTorch")
            ax.set_xticks(list(x))
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "triton_vs_pytorch.png"), dpi=150)
            plt.close(fig)

        # --- Line chart: scaling with input size ---
        sorted_results = sorted(self.results, key=lambda r: r.name)
        for name, group in groupby(sorted_results, key=lambda r: r.name):
            items = list(group)
            if len(items) < 2:
                continue
            sizes = [r.input_size for r in items]
            triton_times = [r.triton_ms for r in items]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(sizes, triton_times, "o-", label="Triton")
            pytorch_times = [r.pytorch_ms for r in items if r.pytorch_ms is not None]
            if len(pytorch_times) == len(items):
                ax.plot(sizes, pytorch_times, "s--", label="PyTorch")
            ax.set_xlabel("Input Size")
            ax.set_ylabel("Time (ms)")
            ax.set_title(f"Scaling: {name}")
            ax.legend()
            fig.tight_layout()
            safe_name = name.replace(" ", "_").replace("/", "_")
            fig.savefig(os.path.join(output_dir, f"scaling_{safe_name}.png"), dpi=150)
            plt.close(fig)


def _fmt_bytes(b: int) -> str:
    """Format bytes into a human-readable string."""
    if b < 1024:
        return f"{b} B"
    if b < 1024**2:
        return f"{b / 1024:.1f} KB"
    if b < 1024**3:
        return f"{b / 1024**2:.1f} MB"
    return f"{b / 1024**3:.2f} GB"
