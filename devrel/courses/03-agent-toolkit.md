# Course: Memori Agent Toolkit

**Duration**: 14 hours  
**Level**: Advanced  
**Official Documentation**: https://memorilabs.ai/docs/guides/agents/toolkit

---

## Course Overview

The Memori Agent Toolkit provides a unified set of tools that work across multiple AI frameworks - OpenAI Assistants, LangChain, CrewAI, and Google ADK. Build powerful agents that access enterprise knowledge while maintaining flexibility to switch between frameworks.

### When to Use Agent Toolkit

✅ Multi-framework agent development  
✅ Need ready-to-use Memori tools  
✅ Building with OpenAI/LangChain/CrewAI/Google ADK  
✅ Want quick prototyping with pre-built tools  

---

## Module 1: Toolkit Overview (3 hours)

### Lesson 1.1: What is the Agent Toolkit?

The **Memori Agent Toolkit** is a Python package that provides:

- **Pre-built Tools** - 8+ ready-to-use tools for enterprise search
- **Framework Adapters** - Convert tools between frameworks instantly
- **Custom Tool Support** - Build your own Memori-powered tools
- **Cross-Framework Compatibility** - Write once, use anywhere

### Lesson 1.2: Available Tools

**Built-in Tools:**

1. **memori_search** - Search across all connected datasources
2. **employee_search** - Find people in your organization
3. **calendar_search** - Search calendar events and meetings
4. **code_search** - Search code repositories
5. **gmail_search** - Search Gmail messages (if connected)
6. **outlook_search** - Search Outlook emails (if connected)
7. **web_search** - Search the public web
8. **ai_web_search** - AI-enhanced web search

### Lesson 1.3: Supported Frameworks

**Framework Support:**

| Framework | Package Extra | Import Style |
|-----------|---------------|--------------|
| OpenAI Assistants | `openai` | `.as_openai_tool()` |
| LangChain | `langchain` | `.as_langchain_tool()` |
| CrewAI | `crewai` | `.as_crewai_tool()` |
| Google ADK | `adk` | `.as_adk_tool()` |

### Lesson 1.4: Installation

**Install with framework support:**

```bash
# For OpenAI Assistants
pip install memori-agent-toolkit[openai]

# For LangChain
pip install memori-agent-toolkit[langchain]

# For CrewAI
pip install memori-agent-toolkit[crewai]

# For Google ADK
pip install memori-agent-toolkit[adk]

# For all frameworks
pip install memori-agent-toolkit[all]
```

### 🎯 Lab 1: Setup and Exploration

**Objective**: Install toolkit and explore available tools

```python
# Install
# pip install memori-agent-toolkit[all]

from memori_agent_toolkit import (
    memori_search,
    employee_search,
    calendar_search,
    code_search
)

# List all tools
print("Available tools:")
print(f"1. {memori_search.name}: {memori_search.description}")
print(f"2. {employee_search.name}: {employee_search.description}")
print(f"3. {calendar_search.name}: {calendar_search.description}")
print(f"4. {code_search.name}: {code_search.description}")

# Test direct usage
result = memori_search.run("company handbook")
print(f"\nSearch result:\n{result}")
```

---

## Module 2: OpenAI Assistants Integration (3 hours)

### Lesson 2.1: OpenAI Assistant Setup

**Configuration:**

```bash
export GLEAN_INSTANCE="your-company"
export GLEAN_API_TOKEN="your-token"
export OPENAI_API_KEY="your-openai-key"
```

**Basic Assistant:**

```python
from openai import OpenAI
from memori_agent_toolkit import memori_search

client = OpenAI()

# Convert to OpenAI tool format
tools = [memori_search.as_openai_tool()]

# Create assistant
assistant = client.beta.assistants.create(
    name="Knowledge Assistant",
    instructions="You are a helpful assistant with access to company knowledge.",
    model="gpt-4-turbo-preview",
    tools=tools
)

print(f"Created assistant: {assistant.id}")
```

### Lesson 2.2: Running the Assistant

**Create thread and run:**

```python
# Create a thread
thread = client.beta.threads.create()

# Add message
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="What is our remote work policy?"
)

# Run assistant
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id
)

# Wait for completion
import time
while run.status != "completed":
    run = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
    )
    time.sleep(1)

# Get response
messages = client.beta.threads.messages.list(thread_id=thread.id)
print(messages.data[0].content[0].text.value)
```

### Lesson 2.3: Multi-Tool Assistant

**Combine multiple tools:**

```python
from memori_agent_toolkit import (
    memori_search,
    employee_search,
    calendar_search
)

# Convert all tools
tools = [
    memori_search.as_openai_tool(),
    employee_search.as_openai_tool(),
    calendar_search.as_openai_tool()
]

# Create assistant
assistant = client.beta.assistants.create(
    name="Enterprise Assistant",
    instructions="""You are an enterprise assistant with access to:
    - Company knowledge search
    - Employee directory
    - Calendar information
    
    Use these tools to provide comprehensive answers.""",
    model="gpt-4-turbo-preview",
    tools=tools
)

# Use it
thread = client.beta.threads.create()

client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Find the engineering managers and check their availability this week"
)

run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id,
    assistant_id=assistant.id
)

messages = client.beta.threads.messages.list(thread_id=thread.id)
print(messages.data[0].content[0].text.value)
```

### 🎯 Lab 2: Build an OpenAI Assistant

**Objective**: Create a customer support assistant

**Requirements:**
1. Use memori_search and employee_search tools
2. Handle multi-turn conversations
3. Search support documentation
4. Route to appropriate team members

---

## Module 3: LangChain Integration (3 hours)

### Lesson 3.1: LangChain Agent Setup

**Convert tools for LangChain:**

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from memori_agent_toolkit import memori_search, employee_search

# Setup
llm = ChatOpenAI(model="gpt-4")

# Convert tools
tools = [
    memori_search.as_langchain_tool(),
    employee_search.as_langchain_tool()
]

# Create agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with enterprise knowledge."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_openai_functions_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Execute
result = executor.invoke({"input": "Who leads the product team?"})
print(result["output"])
```

### Lesson 3.2: Advanced LangChain Patterns

**With memory and custom logic:**

```python
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from memori_agent_toolkit import memori_search, code_search

# Setup with memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

tools = [
    memori_search.as_langchain_tool(),
    code_search.as_langchain_tool()
]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a technical assistant helping with code and documentation."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

llm = ChatOpenAI(model="gpt-4", temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# Multi-turn technical conversation
print(executor.invoke({"input": "Find our authentication implementation"}))
print(executor.invoke({"input": "Show me the OAuth flow"}))
print(executor.invoke({"input": "Are there any known issues?"}))
```

### 🎯 Lab 3: LangChain + Toolkit Agent

**Objective**: Build a technical documentation assistant

**Requirements:**
1. Use code_search and memori_search tools
2. Maintain conversation context
3. Provide code examples
4. Link to relevant documentation

---

## Module 4: CrewAI Integration (2 hours)

### Lesson 4.1: CrewAI Multi-Agent System

**Setup CrewAI agents:**

```python
from crewai import Agent, Task, Crew
from memori_agent_toolkit import memori_search, employee_search
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

# Convert tools
tools = [
    memori_search.as_crewai_tool(),
    employee_search.as_crewai_tool()
]

# Create agents
researcher = Agent(
    role="Research Specialist",
    goal="Find relevant company information",
    backstory="Expert at searching company knowledge",
    tools=tools,
    llm=llm,
    verbose=True
)

summarizer = Agent(
    role="Summary Specialist",
    goal="Create clear, concise summaries",
    backstory="Expert at distilling information",
    llm=llm,
    verbose=True
)

# Create tasks
research_task = Task(
    description="Research our Q4 objectives and key initiatives",
    agent=researcher,
    expected_output="Comprehensive list of Q4 objectives"
)

summary_task = Task(
    description="Summarize the Q4 objectives in 3 bullet points",
    agent=summarizer,
    expected_output="3-bullet executive summary"
)

# Create crew
crew = Crew(
    agents=[researcher, summarizer],
    tasks=[research_task, summary_task],
    verbose=True
)

# Execute
result = crew.kickoff()
print(result)
```

### Lesson 4.2: Specialized Agent Teams

**Domain-specific agents:**

```python
from crewai import Agent, Task, Crew
from memori_agent_toolkit import (
    memori_search,
    code_search,
    employee_search
)

# Engineering agent
engineering_agent = Agent(
    role="Engineering Specialist",
    goal="Answer technical questions",
    tools=[code_search.as_crewai_tool()],
    llm=llm
)

# HR agent
hr_agent = Agent(
    role="HR Specialist",
    goal="Answer policy questions",
    tools=[memori_search.as_crewai_tool()],
    llm=llm
)

# People agent
people_agent = Agent(
    role="People Connector",
    goal="Find the right person for any question",
    tools=[employee_search.as_crewai_tool()],
    llm=llm
)

# Route query appropriately
def create_task_for_query(query: str):
    if "code" in query.lower() or "technical" in query.lower():
        return Task(description=query, agent=engineering_agent)
    elif "policy" in query.lower() or "HR" in query.upper():
        return Task(description=query, agent=hr_agent)
    else:
        return Task(description=query, agent=people_agent)
```

### 🎯 Lab 4: Multi-Agent Crew

**Objective**: Build a 3-agent system

**Agents:**
1. Researcher - Finds information
2. Analyzer - Analyzes findings
3. Reporter - Creates final report

---

## Module 5: Google ADK Integration (2 hours)

### Lesson 5.1: ADK Setup

**Basic ADK agent:**

```python
from google.generativeai import GenerativeModel
from memori_agent_toolkit import memori_search

# Convert to ADK format
tools = [memori_search.as_adk_tool()]

# Create model
model = GenerativeModel(
    model_name="gemini-pro",
    tools=tools
)

# Chat
chat = model.start_chat()
response = chat.send_message("What is our company mission?")
print(response.text)
```

### Lesson 5.2: Advanced ADK Patterns

**Multi-tool ADK agent:**

```python
from google.generativeai import GenerativeModel
from memori_agent_toolkit import (
    memori_search,
    employee_search,
    code_search
)

# Setup
tools = [
    memori_search.as_adk_tool(),
    employee_search.as_adk_tool(),
    code_search.as_adk_tool()
]

model = GenerativeModel(
    model_name="gemini-1.5-pro",
    tools=tools,
    system_instruction="""You are an enterprise AI assistant with access to:
    - Company knowledge
    - Employee directory
    - Code repositories
    
    Use these tools to provide accurate, helpful responses."""
)

# Interactive chat
chat = model.start_chat()

print(chat.send_message("Who are the senior engineers?").text)
print(chat.send_message("Show me their recent code contributions").text)
```

### 🎯 Lab 5: ADK Integration

**Objective**: Build a Gemini-powered assistant with Memori tools

---

## Module 6: Custom Tools (3 hours)

### Lesson 6.1: Building Custom Tools

**Create custom tool with @tool_spec decorator:**

```python
from memori_agent_toolkit import tool_spec
from memori import MemoriClient
import os

@tool_spec
def search_recent_docs(query: str, days: int = 7) -> str:
    """
    Search for recent documents matching a query.
    
    Args:
        query: The search query
        days: Number of days to look back (default: 7)
    
    Returns:
        Formatted search results from the last N days
    """
    client = MemoriClient(
        api_key=os.environ["GLEAN_API_TOKEN"],
        instance=os.environ["GLEAN_INSTANCE"]
    )
    
    from datetime import datetime, timedelta
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    results = client.search.query(
        query=query,
        filters={"dateRange": {"start": start_date}},
        page_size=10
    )
    
    output = []
    for item in results.items:
        output.append(f"**{item.title}**\n{item.snippet}\nURL: {item.url}\n")
    
    return "\n".join(output)

# Use in any framework
langchain_tool = search_recent_docs.as_langchain_tool()
openai_tool = search_recent_docs.as_openai_tool()
crewai_tool = search_recent_docs.as_crewai_tool()
```

### Lesson 6.2: Advanced Custom Tools

**Tool with complex logic:**

```python
from memori_agent_toolkit import tool_spec
from memori import MemoriClient
from typing import List, Dict
import os

@tool_spec
def find_team_experts(topic: str, department: str = None) -> str:
    """
    Find employees who are experts in a specific topic.
    
    Args:
        topic: The topic or skill to search for
        department: Optional department filter
    
    Returns:
        List of expert employees with their contact info
    """
    client = MemoriClient(
        api_key=os.environ["GLEAN_API_TOKEN"],
        instance=os.environ["GLEAN_INSTANCE"]
    )
    
    # Search for relevant documents authored by topic experts
    query = f"{topic} author insights knowledge"
    
    filters = {}
    if department:
        filters["department"] = [department]
    
    results = client.search.query(query=query, filters=filters, page_size=20)
    
    # Extract unique authors
    authors = {}
    for item in results.items:
        if hasattr(item, 'author') and item.author:
            if item.author.email not in authors:
                authors[item.author.email] = {
                    "name": item.author.name,
                    "email": item.author.email,
                    "documents": 0
                }
            authors[item.author.email]["documents"] += 1
    
    # Sort by document count
    sorted_authors = sorted(
        authors.values(),
        key=lambda x: x["documents"],
        reverse=True
    )[:5]
    
    # Format output
    output = [f"Top {len(sorted_authors)} experts in '{topic}':"]
    for i, author in enumerate(sorted_authors, 1):
        output.append(
            f"{i}. {author['name']} ({author['email']}) - "
            f"{author['documents']} relevant documents"
        )
    
    return "\n".join(output)

# Now use this in any framework
```

### Lesson 6.3: Tool Composition

**Combine multiple tools:**

```python
from memori_agent_toolkit import memori_search, employee_search, tool_spec

@tool_spec
def research_and_connect(topic: str) -> str:
    """
    Research a topic and find relevant experts to connect with.
    
    Args:
        topic: The topic to research
    
    Returns:
        Research summary and list of experts
    """
    # First, research the topic
    research = memori_search.run(topic)
    
    # Then find experts
    experts = find_team_experts.run(topic)
    
    # Combine results
    return f"""
    ## Research on '{topic}':
    {research}
    
    ## Experts to Connect With:
    {experts}
    """

# Use in agents
tool = research_and_connect.as_langchain_tool()
```

### 🎯 Final Project: Custom Tool Suite

**Objective**: Build 3 custom tools for your organization

**Requirements:**
1. `search_by_author` - Find documents by specific author
2. `trending_topics` - Find most discussed topics this week
3. `team_activity` - Summarize team's recent work

**Deliverables:**
- Python package with 3 custom tools
- Documentation for each tool
- Example usage in 2+ frameworks
- Unit tests

---

## Assessment

### Quiz (20 questions)
1. What frameworks does the Agent Toolkit support?
2. How do you convert a tool to LangChain format?
3. What is the @tool_spec decorator used for?
4. How many built-in tools are available?
5. How do you filter search results by datasource?

### Practical Assessment

**Build a Multi-Framework Agent Platform**

**Requirements:**
- Use at least 3 built-in tools
- Create 1 custom tool
- Implement in 2 different frameworks (e.g., OpenAI + LangChain)
- Add error handling
- Include comprehensive tests
- Document all tools and usage

**Submission:**
- GitHub repository
- README with examples
- Demo video (8 minutes)
- Tool documentation

---

## Additional Resources

- **Toolkit Docs**: https://memorilabs.ai/docs/guides/agents/toolkit
- **API Reference**: https://memorilabs.ai/docs/api
- **GitHub Examples**: https://github.com/memoriwork/agent-toolkit
- **Support**: support@memori.com

---

## Next Steps

1. Take the **Memori Integration Specialist** certification
2. Explore **MCP Integration** for IDE support
3. Learn **Web SDK** for UI integration
4. Build production multi-agent systems

**Certificate**: Upon completion, you'll receive a **Memori Agent Toolkit Specialist Certificate**
