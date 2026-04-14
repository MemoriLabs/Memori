"""
SQLite BYODB + Rust core smoke test.

This mirrors examples/sqlite/main.py but forces MEMORI_USE_RUST_CORE=1
and fails fast if the Rust core extension is not active.
"""

import os

from openai import OpenAI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from memori import Memori


def main() -> None:
    print("Starting rust_core_main...", flush=True)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("Set OPENAI_API_KEY before running this example.")

    os.environ["MEMORI_USE_RUST_CORE"] = "1"
    os.environ["MEMORI_TEST_MODE"] = "1"

    client = OpenAI(api_key=openai_api_key, timeout=30.0)
    engine = create_engine("sqlite:///memori_rust_core.db")
    session = sessionmaker(bind=engine)

    print("Initializing Memori (BYODB + Rust core)...", flush=True)
    mem = Memori(conn=session).llm.register(client)
    mem.attribution(entity_id="rust-core-user", process_id="sqlite-example")
    mem.config.storage.build()

    if mem.config.rust_core is None:
        raise RuntimeError(
            "Rust core is not active. Build/load memori_python and rerun."
        )

    print("Rust core active:", type(mem.config.rust_core).__name__, flush=True)

    print("\nYou: What is my favorite season?", flush=True)
    first = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is my favorite season?"}],
    )
    print("AI:", first.choices[0].message.content, flush=True)

    print("\nYou: What's my favorite season?", flush=True)
    second = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What's my favorite season?"}],
    )
    print("AI:", second.choices[0].message.content, flush=True)

    print("\nYou: What season do I like for the weather?", flush=True)
    third = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "What season do I like for the weather?"}
        ],
    )
    print("AI:", third.choices[0].message.content, flush=True)

    mem.augmentation.wait(timeout=10)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
