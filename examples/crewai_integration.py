"""Example: CrewAI agents with Memori for persistent memory.

This example demonstrates how to integrate Memori with CrewAI to enable
agents to automatically capture and recall structured information across
conversations and executions.

Prerequisites:
    pip install crewai memori python-dotenv
    Set OPENAI_API_KEY or MEMORI_API_KEY in .env
"""

import os
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def example_single_agent_with_memory():
    """Example: Single agent with Memori memory."""
    try:
        from crewai import Agent, Crew, Task
        from memori import Memori
    except ImportError:
        print("CrewAI not installed. Install with: pip install crewai")
        return

    print("\n📚 Example 1: Single Agent with Memori Memory\n")
    print("-" * 60)

    # Initialize Memori
    mem = Memori().attribution(
        entity_id="research_team",
        process_id="crewai_researcher",
    )

    # Create a research agent
    researcher = Agent(
        role="Research Analyst",
        goal="Find and analyze information about AI trends",
        backstory=(
            "An experienced research analyst specializing in "
            "artificial intelligence and emerging technologies."
        ),
        extra_context={"memori": mem},
    )

    # Create a simple task
    task = Task(
        description="Research the latest trends in AI and LLMs",
        agent=researcher,
        expected_output="A summary of current AI trends",
    )

    # Create crew
    crew = Crew(
        agents=[researcher],
        tasks=[task],
        verbose=True,
    )

    # Register with Memori for automatic capturing
    mem.crewai.register(crew=crew)

    # Execute
    print("\n🚀 Executing crew...\n")
    result = crew.kickoff()

    print(f"\n✅ Execution complete!")
    print(f"Result: {result}\n")

    # Demonstrate memory recall
    print("\n🔍 Recalling related memories...\n")
    memories = mem.recall("AI trends and LLMs")
    print(f"Found {len(memories)} related memories")
    for i, memory in enumerate(memories[:3], 1):
        print(f"  {i}. {memory}")


def example_multi_agent_crew_with_memory():
    """Example: Multi-agent crew with shared Memori."""
    try:
        from crewai import Agent, Crew, Task
        from memori import Memori
    except ImportError:
        print("CrewAI not installed. Install with: pip install crewai")
        return

    print("\n👥 Example 2: Multi-Agent Crew with Shared Memory\n")
    print("-" * 60)

    # Initialize Memori with team context
    mem = Memori().attribution(
        entity_id="product_team",
        process_id="crewai_product_dev",
    )

    # Create specialized agents
    researcher = Agent(
        role="Market Researcher",
        goal="Identify market opportunities",
        backstory="Expert in market analysis and trends",
        extra_context={"memori": mem},
    )

    strategist = Agent(
        role="Product Strategist",
        goal="Develop product strategies based on research",
        backstory="Experienced product strategist",
        extra_context={"memori": mem},
    )

    # Create interdependent tasks
    research_task = Task(
        description="Analyze the market for AI-powered tools",
        agent=researcher,
        expected_output="Market analysis report",
    )

    strategy_task = Task(
        description="Based on research, create a go-to-market strategy",
        agent=strategist,
        expected_output="Strategy document",
    )

    # Create crew
    crew = Crew(
        agents=[researcher, strategist],
        tasks=[research_task, strategy_task],
        verbose=True,
    )

    # Register with Memori
    mem.crewai.register(crew=crew)

    print("\n🚀 Executing multi-agent crew...\n")
    result = crew.kickoff()

    print(f"\n✅ Crew execution complete!")
    print("\n💾 Agent interactions captured in Memori")
    print("The team's decisions and findings are now persistent")


def example_memory_lifecycle():
    """Example: Full memory lifecycle with CrewAI."""
    try:
        from crewai import Agent, Crew, Task
        from memori import Memori
    except ImportError:
        print("CrewAI not installed. Install with: pip install crewai")
        return

    print("\n🔄 Example 3: Memory Lifecycle\n")
    print("-" * 60)

    # First session: Create memories
    print("\nSession 1: Creating initial memories...\n")
    mem1 = Memori().attribution(
        entity_id="learning_bot",
        process_id="session_1",
    )

    agent = Agent(
        role="Learning Assistant",
        goal="Learn from tasks and retain knowledge",
        backstory="A helpful learning assistant",
    )

    task = Task(
        description="Learn about Python programming concepts",
        agent=agent,
        expected_output="Understanding of Python concepts",
    )

    crew = Crew(agents=[agent], tasks=[task])
    mem1.crewai.register(crew=crew)
    crew.kickoff()
    print("✅ Session 1 complete - memories stored\n")

    # Second session: Recall and build on memories
    print("Session 2: Recalling and building on memories...\n")
    mem2 = Memori().attribution(
        entity_id="learning_bot",
        process_id="session_2",
    )

    print("🔍 Recalling previous learning...")
    memories = mem2.recall("Python programming")
    print(f"Found {len(memories)} previous memories\n")

    print("✅ Session 2 complete - using augmented memories\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("🎯 CrewAI + Memori Integration Examples")
    print("=" * 60)

    # Check for required API key
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("MEMORI_API_KEY"):
        print("\n⚠️  No API key configured")
        print("Set OPENAI_API_KEY or MEMORI_API_KEY in .env")
        return

    try:
        example_single_agent_with_memory()
        example_multi_agent_crew_with_memory()
        example_memory_lifecycle()

        print("\n" + "=" * 60)
        print("✅ All examples completed!")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
