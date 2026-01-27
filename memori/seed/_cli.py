r"""
 __  __                           _
|  \/  | ___ _ __ ___   ___  _ __(_)
| |\/| |/ _ \ '_ ` _ \ / _ \| '__| |
| |  | |  __/ | | | | | (_) | |  | |
|_|  |_|\___|_| |_| |_|\___/|_|  |_|
                  perfectam memoriam
                       memorilabs.ai
"""

import argparse
import json
import os
import sys
from pathlib import Path

from memori._cli import Cli
from memori._config import Config


class SeedManager:
    def __init__(self, config: Config):
        self.config = config
        self.cli = Cli(config)

    def _create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="python -m memori seed",
            description="Bulk seed conversations for memory creation",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
File format:
  {
    "entity_id": "user-123",
    "conversations": [
      {"id": "conv-1", "messages": [...]}
    ]
  }

Environment variables:
  MEMORI_API_KEY                       Required for AA access
""",
        )
        parser.add_argument(
            "file",
            metavar="<file.json>",
            help="Path to JSON file containing conversations",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=10,
            metavar="N",
            help="Concurrent requests (default: 10)",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Validate file without seeding",
        )
        return parser

    def usage(self):
        self._create_parser().print_help()

    def execute(self):
        parser = self._create_parser()
        args = parser.parse_args(sys.argv[2:])

        path = Path(args.file)
        if not path.exists():
            self.cli.print(f"Error: File not found: {args.file}")
            sys.exit(1)

        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            self.cli.print(f"Error: Invalid JSON: {e}")
            sys.exit(1)

        entity_id = data.get("entity_id")
        if not entity_id:
            self.cli.print("Error: entity_id is required in the JSON file")
            sys.exit(1)

        process_id = data.get("process_id")

        conversations = data.get("conversations", [])
        if not conversations:
            self.cli.print("Error: No conversations found in file")
            sys.exit(1)

        total_messages = sum(len(c.get("messages", [])) for c in conversations)

        self.cli.print(f"File: {args.file}")
        self.cli.print(f"Entity ID: {entity_id}")
        self.cli.print(f"Process ID: {process_id or 'None'}")
        self.cli.print(f"Conversations: {len(conversations)}")
        self.cli.print(f"Total messages: {total_messages}")
        self.cli.print(f"Batch size: {args.batch_size}\n")

        if args.dry_run:
            self.cli.print("Dry run - validation complete, no data seeded.")
            self._print_preview(conversations[:3])
            return

        if not os.environ.get("MEMORI_API_KEY"):
            self.cli.print("Error: MEMORI_API_KEY is required for seeding.")
            self.cli.print("Set your API key: export MEMORI_API_KEY=<your_key>")
            sys.exit(1)

        self._run_seeding(
            file_path=args.file,
            entity_id=entity_id,
            process_id=process_id,
            batch_size=args.batch_size,
        )

    def _print_preview(self, conversations):
        self.cli.print("Preview of conversations:")
        self.cli.print("-" * 50)

        for i, conv in enumerate(conversations):
            conv_id = conv.get("id", f"[index-{i}]")
            messages = conv.get("messages", [])
            self.cli.print(f"  {conv_id}: {len(messages)} messages")

            if messages:
                first_msg = messages[0]
                content = first_msg.get("content", "")[:50]
                self.cli.print(f"    First: [{first_msg.get('role')}] {content}...")

        if len(conversations) < 3:
            self.cli.print("  ... and more")

    def _run_seeding(
        self,
        file_path: str,
        entity_id: str,
        process_id: str | None,
        batch_size: int,
    ):
        from memori import Memori
        from memori.seed import seed_from_file

        self.cli.print("Starting seeding...\n")

        last_percent = -1

        def on_progress(processed, total, result):
            nonlocal last_percent
            percent = int((processed / total) * 100)

            if percent != last_percent:
                bar = "=" * (percent // 2) + ">" + " " * (50 - percent // 2)
                print(f"\r[{bar}] {percent}% ({processed}/{total})", end="", flush=True)
                last_percent = percent

        try:
            m = Memori()

            storage = m.config.storage
            if storage is None:
                self.cli.print("Error: No database connection configured")
                sys.exit(1)

            driver = getattr(storage, "driver", None)
            if driver is None:
                self.cli.print("Error: No database driver configured")
                sys.exit(1)

            result = seed_from_file(
                config=m.config,
                driver=driver,
                file_path=file_path,
                entity_id=entity_id,
                process_id=process_id,
                batch_size=batch_size,
                on_progress=on_progress,
            )

            print("")  # newline after progress bar
            self.cli.print("=" * 50)
            self.cli.print("SEEDING COMPLETE")
            self.cli.print("=" * 50)
            self.cli.print(f"Total conversations: {result.total}")
            self.cli.print(f"Successful: {result.successful}")
            self.cli.print(f"Failed: {result.failed}")
            self.cli.print(f"Total triples extracted: {result.total_triples}")
            self.cli.print(f"Duration: {result.duration_ms / 1000:.1f}s")
            self.cli.print(f"Success rate: {result.success_rate:.1f}%")

            if result.failed > 0:
                self.cli.print("")
                self.cli.print("Failed conversations:")
                for conv in result.conversations:
                    if not conv.success:
                        self.cli.print(f"  - {conv.conversation_id}: {conv.error}")

        except Exception as e:
            self.cli.print(f"\nError during seeding: {e}")
            raise
