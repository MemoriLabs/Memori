from crewai import Agent, Crew, Process, Task
from memori.proxy.crewai import MemoriCrewAIAdapter

# Initialize the Memori adapter
# Ensure OPENAI_API_KEY is set in your environment
adapter = MemoriCrewAIAdapter()


def main():
    # 1. Create your agents
    researcher = Agent(
        role="Senior Researcher",
        goal="Discover facts about AI memory systems",
        backstory="You are an expert at analyzing AI memory capabilities.",
        verbose=True,
        allow_delegation=False,
    )

    writer = Agent(
        role="Tech Writer",
        goal="Write a summary based on research",
        backstory="You transform complex facts into easy-to-understand summaries.",
        verbose=True,
        allow_delegation=False,
    )

    # 2. Create tasks
    research_task = Task(
        description="Research the benefits of long-term memory in AI agents.",
        expected_output="A list of 3 benefits.",
        agent=researcher,
    )

    writing_task = Task(
        description="Write a short paragraph summarizing the benefits found by the researcher.",
        expected_output="A short paragraph.",
        agent=writer,
    )

    # 3. Initialize Crew
    crew = Crew(agents=[researcher, writer], tasks=[research_task, writing_task], process=Process.sequential)

    # 4. Inject Memori telemetry / observability adapter
    user_id = "test_user_crew"
    session_id = "run_001"
    crew = adapter.setup_crew(crew, user_id=user_id, run_id=session_id)

    # 5. Kickoff the crew
    print("Starting Crew execution...")
    result = crew.kickoff()
    print("Crew execution finished.")
    print("Result:", result)

    # 6. Retrieve the automatically captured memories
    print("\n--- Memori Captured Memories ---")
    memories = adapter.client.get_all(user_id=user_id, run_id=session_id)
    for mem in memories:
        print(f"- [Agent: {mem.get('agent_id')}] {mem.get('memory')}")


if __name__ == "__main__":
    main()
