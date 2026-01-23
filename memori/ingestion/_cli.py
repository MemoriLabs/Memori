r"""
 __  __                           _
|  \/  | ___ _ __ ___   ___  _ __(_)
| |\/| |/ _ \ '_ ` _ \ / _ \| '__| |
| |  | |  __/ | | | | | (_) | |  | |
|_|  |_|\___|_| |_| |_|\___/|_|  |_|
                  perfectam memoriam
                       memorilabs.ai
"""

import json
import os
import sys
from pathlib import Path

from memori._cli import Cli
from memori._config import Config


class Manager:
    def __init__(self, config: Config):
        self.config = config
        self.cli = Cli(config)

    def usage(self):
        self.cli.print("Usage: python -m memori ingest <file.json> [options]")
        self.cli.print("")
        self.cli.print("Arguments:")
        self.cli.print("  <file.json>       Path to JSON file containing conversations")
        self.cli.print("")
        self.cli.print("Options:")
        self.cli.print("  --entity-id ID    Entity ID (overrides file value)")
        self.cli.print("  --process-id ID   Process ID (overrides file value)")
        self.cli.print("  --batch-size N    Concurrent requests (default: 10)")
        self.cli.print("  --dry-run         Validate file without ingesting")
        self.cli.print("")
        self.cli.print("File format:")
        self.cli.print("  {")
        self.cli.print('    "entity_id": "user-123",')
        self.cli.print('    "conversations": [')
        self.cli.print('      {"id": "conv-1", "messages": [...]}')
        self.cli.print("    ]")
        self.cli.print("  }")
        self.cli.print("")
        self.cli.print("Environment:")
        self.cli.print("  MEMORI_API_KEY                    Required for AA access")
        self.cli.print("  MEMORI_COCKROACHDB_CONNECTION_STRING  Database connection")

    def execute(self):
        args = sys.argv[2:]

        if not args:
            self.usage()
            return

        file_path = None
        entity_id = None
        process_id = None
        batch_size = 10
        dry_run = False

        i = 0
        while i < len(args):
            arg = args[i]

            if arg == "--entity-id" and i + 1 < len(args):
                entity_id = args[i + 1]
                i += 2
            elif arg == "--process-id" and i + 1 < len(args):
                process_id = args[i + 1]
                i += 2
            elif arg == "--batch-size" and i + 1 < len(args):
                batch_size = int(args[i + 1])
                i += 2
            elif arg == "--dry-run":
                dry_run = True
                i += 1
            elif arg.startswith("--"):
                self.cli.print(f"Unknown option: {arg}")
                self.usage()
                return
            else:
                file_path = arg
                i += 1

        if not file_path:
            self.cli.print("Error: No file specified")
            self.usage()
            return

        path = Path(file_path)
        if not path.exists():
            self.cli.print(f"Error: File not found: {file_path}")
            return

        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            self.cli.print(f"Error: Invalid JSON: {e}")
            return

        final_entity_id = entity_id or data.get("entity_id")
        if not final_entity_id:
            self.cli.print(
                "Error: entity_id must be provided via --entity-id or in file"
            )
            return

        final_process_id = process_id or data.get("process_id")

        conversations = data.get("conversations", [])
        if not conversations:
            self.cli.print("Error: No conversations found in file")
            return

        total_messages = sum(len(c.get("messages", [])) for c in conversations)

        self.cli.print(f"File: {file_path}")
        self.cli.print(f"Entity ID: {final_entity_id}")
        self.cli.print(f"Process ID: {final_process_id or 'None'}")
        self.cli.print(f"Conversations: {len(conversations)}")
        self.cli.print(f"Total messages: {total_messages}")
        self.cli.print(f"Batch size: {batch_size}")
        self.cli.print("")

        if dry_run:
            self.cli.print("Dry run - validation complete, no data ingested.")
            self._print_preview(conversations[:3])
            return

        if not os.environ.get("MEMORI_API_KEY"):
            self.cli.print(
                "Warning: MEMORI_API_KEY not set - running in anonymous mode"
            )
            self.cli.print("")

        if not os.environ.get("MEMORI_COCKROACHDB_CONNECTION_STRING"):
            self.cli.print("Error: MEMORI_COCKROACHDB_CONNECTION_STRING not set")
            self.cli.print(
                "Set this environment variable to your database connection string."
            )
            return

        self._run_ingestion(
            file_path=file_path,
            entity_id=final_entity_id,
            process_id=final_process_id,
            batch_size=batch_size,
            total_conversations=len(conversations),
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

    def _run_ingestion(
        self,
        file_path: str,
        entity_id: str,
        process_id: str,
        batch_size: int,
        total_conversations: int,
    ):
        from memori import Memori
        from memori.ingestion import ingest_from_file

        self.cli.print("Starting ingestion...")
        self.cli.print("")

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

            driver = m.config.storage.driver

            result = ingest_from_file(
                config=m.config,
                driver=driver,
                file_path=file_path,
                entity_id=entity_id,
                process_id=process_id,
                batch_size=batch_size,
                on_progress=on_progress,
            )

            print("")
            self.cli.print("")
            self.cli.print("=" * 50)
            self.cli.print("INGESTION COMPLETE")
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
            self.cli.print(f"\nError during ingestion: {e}")
            raise
