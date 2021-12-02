from argparse import ArgumentParser
import json

from robust_kolmogorov.solver import solve
from robust_kolmogorov.utils import setup

if __name__ == "__main__":
    parser = ArgumentParser(description="PDE Solver")
    parser.add_argument(
        "--overwrite",
        "-o",
        type=str,
        help="JSON encoded string to update the config.",
    )
    args = parser.parse_args()
    overwrite_config = json.loads(args.overwrite) if args.overwrite else None

    setup(overwrite_config=overwrite_config)
    solve(overwrite_config=overwrite_config)
