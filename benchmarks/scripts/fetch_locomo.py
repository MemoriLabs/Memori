from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch/prepare LoCoMo dataset locally (does not commit data to git)."
    )
    parser.add_argument(
        "--source",
        default="",
        help="Path to an existing LoCoMo JSON file to copy into the cache.",
    )
    parser.add_argument(
        "--dest",
        default="",
        help="Destination file path (defaults to ~/.cache/memori/benchmarks/locomo/locomo.json).",
    )
    parser.add_argument(
        "--print-dest",
        action="store_true",
        help="Print resolved destination path and exit.",
    )
    args = parser.parse_args(argv)

    dest = _default_dest() if not args.dest else Path(args.dest).expanduser()
    dest = dest.resolve()

    if args.print_dest:
        print(str(dest))
        return 0

    if not args.source:
        print(
            "No --source provided.\n\n"
            "Download LoCoMo from Snap Research and pass the JSON path here.\n"
            "Upstream repo: https://github.com/snap-research/locomo\n",
            file=sys.stderr,
        )
        return 2

    src = Path(args.source).expanduser().resolve()
    if not src.exists():
        print(f"Source file not found: {src}", file=sys.stderr)
        return 2

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)

    digest = _sha256(dest)
    print(f"Wrote: {dest}")
    print(f"sha256: {digest}")
    return 0


def _default_dest() -> Path:
    return Path("~/.cache/memori/benchmarks/locomo/locomo.json").expanduser()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
