"""
Quickstart: Memori + OpenAI Responses API + SQLite

Demonstrates how Memori adds memory using OpenAI's new Responses API.
"""

import os

from openai import OpenAI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

from memori import Memori

# Setup OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "<your_api_key_here>"))

# Setup SQLite
engine = create_engine("sqlite:///memori.db")
Session = sessionmaker(bind=engine)

# Setup Memori - that's it!
mem = Memori(conn=Session).llm.register(client)
mem.attribution(entity_id="user-123", process_id="my-app")
mem.config.storage.build()

if __name__ == "__main__":
    # First conversation - establish facts using Responses API
    print("You: My favorite color is blue and I live in Paris")
    response1 = client.responses.create(
        model="gpt-4o-mini",
        input="My favorite color is blue and I live in Paris",
        instructions="You are a helpful assistant. Remember what the user tells you.",
        max_output_tokens=100,
    )
    print(f"AI: {response1.output_text}\n")

    # Second conversation - Memori recalls context automatically
    print("You: What's my favorite color?")
    response2 = client.responses.create(
        model="gpt-4o-mini",
        input="What's my favorite color?",
        instructions="Answer based on what you know about the user.",
        max_output_tokens=100,
    )
    print(f"AI: {response2.output_text}\n")

    # Third conversation - context is maintained
    print("You: What city do I live in?")
    response3 = client.responses.create(
        model="gpt-4o-mini",
        input="What city do I live in?",
        instructions="Answer based on what you know about the user.",
        max_output_tokens=100,
    )
    print(f"AI: {response3.output_text}")

    # Advanced Augmentation runs asynchronously to efficiently
    # create memories. For this example, a short lived command
    # line program, we need to wait for it to finish.
    mem.augmentation.wait()
