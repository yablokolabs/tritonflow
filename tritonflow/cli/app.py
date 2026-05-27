"""TritonFlow CLI application."""

import typer

from tritonflow.cli.commands import benchmark, compare, info, profile, visualize

__all__ = ["app"]

app = typer.Typer(
    name="tritonflow",
    help="TritonFlow: High-performance GPU kernels for modern AI workloads.",
    add_completion=False,
)

app.add_typer(benchmark.app, name="benchmark", help="Run kernel benchmarks")
app.add_typer(profile.app, name="profile", help="Profile kernel execution")
app.add_typer(compare.app, name="compare", help="Compare Triton vs PyTorch vs NumPy")
app.add_typer(visualize.app, name="visualize", help="Generate benchmark visualizations")
app.command(name="info")(info.info_command)

if __name__ == "__main__":
    app()
