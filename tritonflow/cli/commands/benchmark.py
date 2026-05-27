"""Benchmark commands for TritonFlow kernels."""

from pathlib import Path

import typer
from rich.console import Console

__all__ = ["app"]

app = typer.Typer()
console = Console()


@app.command("run")
def run_benchmarks(
    suite: str = typer.Option(
        "all", help="Benchmark suite: all, vector_ops, ml_kernels, similarity"
    ),
    iterations: int = typer.Option(100, help="Number of benchmark iterations"),
    warmup: int = typer.Option(10, help="Number of warmup iterations"),
    export_md: str = typer.Option(None, help="Export results as Markdown to this path"),
    export_csv: str = typer.Option(None, help="Export results as CSV to this path"),
):
    """Run benchmark suite."""
    try:
        from tritonflow.utils.gpu import is_gpu_available

        if not is_gpu_available():
            console.print("[yellow]Warning: No GPU detected. Benchmarks may be limited.[/yellow]")
    except ImportError:
        console.print("[yellow]Warning: GPU utilities not available.[/yellow]")

    console.print(f"[bold]Running benchmark suite:[/bold] {suite}")
    console.print(f"  Iterations: {iterations}, Warmup: {warmup}")

    # Placeholder: import and run actual benchmark suites
    console.print("[yellow]Benchmark runner not yet implemented.[/yellow]")

    if export_md:
        path = Path(export_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]Markdown results would be exported to {export_md}[/green]")

    if export_csv:
        path = Path(export_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]CSV results would be exported to {export_csv}[/green]")
