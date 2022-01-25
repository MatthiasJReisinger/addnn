from addnn.cli import cli
import addnn.controller.controller
import addnn.node.cli
import addnn.serve.cli
import addnn.infer
import addnn.example.cli
import addnn.inspect
import addnn.benchmark.cli
import addnn.benchmark.strategies.cli
import addnn.benchmark.end_to_end.cli
import addnn.benchmark.end_to_end.static
import addnn.benchmark.end_to_end.dynamic
import addnn.profile.cli
import addnn.torchhub
import addnn.thresholds
import addnn.validate


def main() -> None:
    """Main entrypoint for the `click` CLI application."""
    cli()


if __name__ == "__main__":
    main()
