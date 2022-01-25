import click


@click.group(help="A collection of tools for training, deployment, and orchestration of distributed neural networks.")
def cli() -> None:
    """Root-level `click` group where main CLI commands are attached."""
    pass
