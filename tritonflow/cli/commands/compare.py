"""Comparison commands for Triton vs PyTorch vs NumPy."""

import typer
from rich.console import Console

__all__ = ["app"]

app = typer.Typer()
console = Console()


@app.command("ops")
def compare_ops(
    kernel: str = typer.Argument(help="Kernel to compare"),
    sizes: str = typer.Option("1024,65536,1048576", help="Comma-separated tensor sizes"),
):
    """Compare Triton kernel against PyTorch and NumPy implementations."""
    try:
        from tritonflow.utils.gpu import is_gpu_available

        if not is_gpu_available():
            console.print("[yellow]Warning: No GPU detected. Triton results unavailable.[/yellow]")
    except ImportError:
        console.print("[yellow]Warning: GPU utilities not available.[/yellow]")

    size_list = [int(s.strip()) for s in sizes.split(",")]
    console.print(f"[bold]Comparing kernel:[/bold] {kernel}")
    console.print(f"  Sizes: {size_list}")

    # Placeholder: import and run actual comparison
    console.print("[yellow]Comparison runner not yet implemented.[/yellow]")
