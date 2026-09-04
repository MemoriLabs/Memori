"""
Memori + LlamaIndex: Customer Support Bot Example
==================================================

Demonstrates how Memori adds persistent cross-session memory to a
LlamaIndex chat engine.

Requirements:
    pip install memori llama-index-core llama-index-llms-openai python-dotenv

Environment variables:
    MEMORI_API_KEY   — Memori Cloud API key (or use BYODB connection)
    OPENAI_API_KEY   — OpenAI API key
"""

import os

from dotenv import load_dotenv

load_dotenv()

from llama_index.core.chat_engine import SimpleChatEngine  # noqa: E402
from llama_index.llms.openai import OpenAI  # noqa: E402

from memori import Memori  # noqa: E402
from memori.integrations.llamaindex import MemoriChatEngine  # noqa: E402


def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. Initialise Memori (Cloud mode — needs MEMORI_API_KEY)            #
    #    For local / BYODB mode, pass a SQLAlchemy Session instead:       #
    #       from sqlalchemy import create_engine                          #
    #       from sqlalchemy.orm import sessionmaker                       #
    #       engine = create_engine("sqlite:///demo.db")                   #
    #       Session = sessionmaker(bind=engine)                           #
    #       mem = Memori(conn=Session)                                    #
    #       mem.config.storage.build()                                    #
    # ------------------------------------------------------------------ #
    mem = Memori()

    # ------------------------------------------------------------------ #
    # 2. Build a vanilla LlamaIndex SimpleChatEngine                      #
    # ------------------------------------------------------------------ #
    llm = OpenAI(model="gpt-4o-mini", system_prompt=(
        "You are a helpful customer support agent. "
        "Remember customer preferences and history from previous conversations."
    ))
    base_engine = SimpleChatEngine.from_defaults(llm=llm)

    # ------------------------------------------------------------------ #
    # 3. Wrap it with MemoriChatEngine — one line of code                 #
    # ------------------------------------------------------------------ #
    engine = MemoriChatEngine(
        chat_engine=base_engine,
        memori=mem,
        entity_id="customer-alice",       # identifies the user
        process_id="support-agent",       # identifies this agent
    )

    # ------------------------------------------------------------------ #
    # 4. Simulate a first session                                         #
    # ------------------------------------------------------------------ #
    print("=== Session 1 ===\n")

    print("Customer: Hi, I'd like to order a large pepperoni pizza, extra cheese.")
    r1 = engine.chat("Hi, I'd like to order a large pepperoni pizza, extra cheese.")
    print(f"Agent:    {r1}\n")

    print("Customer: My address is 42 Maple Street, Springfield.")
    r2 = engine.chat("My address is 42 Maple Street, Springfield.")
    print(f"Agent:    {r2}\n")

    # ------------------------------------------------------------------ #
    # 5. Simulate a second session (reset → new Memori session)           #
    # ------------------------------------------------------------------ #
    engine.reset()
    print("=== Session 2 (new session — Memori recalls past conversation) ===\n")

    print("Customer: What did I order last time?")
    r3 = engine.chat("What did I order last time?")
    print(f"Agent:    {r3}\n")

    print("Customer: And what was my address on file?")
    r4 = engine.chat("And what was my address on file?")
    print(f"Agent:    {r4}\n")

    # ------------------------------------------------------------------ #
    # 6. Wait for background augmentation to finish before exiting        #
    # ------------------------------------------------------------------ #
    print("Waiting for Memori augmentation to finish...")
    mem.augmentation.wait()
    print("Done.")


if __name__ == "__main__":
    main()
