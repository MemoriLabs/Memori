"""
Quickstart: Memori + OpenAI + OceanBase

Demonstrates how Memori adds memory across conversations.
"""

import os

from openai import OpenAI
from sqlalchemy import create_engine
from sqlalchemy.dialects import registry
from sqlalchemy.orm import sessionmaker

from memori import Memori

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

registry.register("mysql.oceanbase", "pyobvector.schema.dialect", "OceanBaseDialect")

database_connection_string = os.getenv("DATABASE_CONNECTION_STRING")
if not database_connection_string:
    raise ValueError("DATABASE_CONNECTION_STRING must be set in the environment")

engine = create_engine(database_connection_string, pool_pre_ping=True)
Session = sessionmaker(bind=engine)

mem = Memori(conn=Session).llm.register(client)
mem.attribution(entity_id="user-123", process_id="my-app")
mem.config.storage.build()

if __name__ == "__main__":
    model = os.getenv("OPENAI_MODEL", "qwen-plus")
    print("You: My favorite color is blue and I live in Paris")
    response1 = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "My favorite color is blue and I live in Paris"}
        ],
    )
    print(f"AI: {response1.choices[0].message.content}\n")

    print("You: What's my favorite color?")
    response2 = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "What's my favorite color?"}],
    )
    print(f"AI: {response2.choices[0].message.content}\n")

    print("You: What city do I live in?")
    response3 = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "What city do I live in?"}],
    )
    print(f"AI: {response3.choices[0].message.content}")

    # Wait for background augmentation in short-lived scripts.
    mem.augmentation.wait()
