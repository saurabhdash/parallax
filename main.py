"""Parallax entry point."""

import argparse
from pathlib import Path
import sys

from parallax.launcher import launch


def print_banner() -> None:
    banner_path = Path(__file__).parent / "assets" / "parallax-banner.ansi"
    sys.stdout.write(banner_path.read_text(encoding="utf-8"))


def main() -> None:
    print_banner()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/qwen3-0.6b-cispo.toml"),
    )
    parser.add_argument("--speedrun", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    assert config_path.is_file()
    assert config_path.suffix == ".toml"
    launch(config_path, speedrun=args.speedrun)


if __name__ == "__main__":
    main()