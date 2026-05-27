"""System and GPU information command."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

__all__ = ["info_command"]

console = Console()


def info_command():
    """Display GPU and system information."""
    import platform
    import sys

    import tritonflow

    console.print(Panel(f"[bold]TritonFlow v{tritonflow.__version__}[/bold]"))

    # System info
    table = Table(title="System Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Python", sys.version.split()[0])
    table.add_row("Platform", platform.platform())

    try:
        import torch

        table.add_row("PyTorch", torch.__version__)
    except ImportError:
        table.add_row("PyTorch", "[red]not installed[/red]")

    try:
        import triton

        table.add_row("Triton", triton.__version__)
    except ImportError:
        table.add_row("Triton", "[red]not installed[/red]")

    try:
        import numpy

        table.add_row("NumPy", numpy.__version__)
    except ImportError:
        table.add_row("NumPy", "[red]not installed[/red]")

    console.print(table)

    # GPU info
    try:
        from tritonflow.utils.gpu import get_device_info, is_gpu_available

        if is_gpu_available():
            info = get_device_info()
            if info is not None:
                gpu_table = Table(title="GPU Information")
                gpu_table.add_column("Property", style="cyan")
                gpu_table.add_column("Value", style="green")
                for k, v in info.items():
                    gpu_table.add_row(k, str(v))
                console.print(gpu_table)
        else:
            console.print("[yellow]No GPU detected. Running in CPU-only mode.[/yellow]")
    except ImportError:
        console.print("[yellow]GPU utilities not available.[/yellow]")
