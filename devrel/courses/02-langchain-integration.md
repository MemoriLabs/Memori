# Course: LangChain Integration with Memori

**Duration**: 12 hours  
**Level**: Intermediate  
**Official Documentation**: https://memorilabs.ai/docs/guides/agents/langchain

---

## Course Overview

Build AI agents using LangChain with Memori's enterprise search and knowledge capabilities. The `langchain-memori` package provides seamless integration between LangChain's agent framework and Memori's Work AI Platform.

### When to Use LangChain Integration

✅ Building agents with LangChain framework  
✅ Need pre-built agent patterns and chains  
✅ Want extensive ecosystem of tools and integrations  
✅ Prefer declarative agent configuration  

---

## Module 1: LangChain Fundamentals (3 hours)

### Lesson 1.1: What is LangChain?

**LangChain** is a framework for developing applications powered by language models. It provides:

- **Chains** - Sequences of operations (e.g., search → format → generate)
- **Agents** - Systems that use LLMs to decide which actions to take
- **Tools** - Functions that agents can call (e.g., Memori search)
- **Memory** - Conversation history management
- **Prompts** - Reusable templates for LLM interactions

### Lesson 1.2: LangChain Components

```python
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

# LLM
llm = ChatOpenAI(model="gpt-4")

# Tool
search_tool = Tool(
    name="search",
    func=lambda x: f"Results for: {x}",
    description="Useful for searching information"
)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("human", "{input}"),
    ("ai", "{agent_scratchpad}")
])

# Memory
memory = ConversationBufferMemory()

# Agent
agent = create_openai_functions_agent(llm, [search_tool], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[search_tool], memory=memory)

# Execute
response = agent_executor.invoke({"input": "What is the capital of France?"})
print(response["output"])
```

### Lesson 1.3: Installation

**Install LangChain and Memori integration:**

```bash
pip install -U langchain-memori
```

**Additional dependencies:**
```bash
# For OpenAI models
pip install openai

# For Anthropic models
pip install anthropic

# For other LangChain features
pip install langchain
```

### 🎯 Lab 1: LangChain Basics

**Objective**: Create a simple LangChain agent without Memori

```python
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate
import os

# Setup
os.environ["OPENAI_API_KEY"] = "your-key"

llm = ChatOpenAI(model="gpt-3.5-turbo")

# Create a simple calculator tool
def add(input_str):
    """Add two numbers"""
    a, b = input_str.split(",")
    return float(a) + float(b)

calculator = Tool(
    name="Calculator",
    func=add,
    description="Useful for adding two numbers. Input format: 'a,b'"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_openai_functions_agent(llm, [calculator], prompt)
executor = AgentExecutor(agent=agent, tools=[calculator], verbose=True)

# Test
result = executor.invoke({"input": "What is 123 + 456?"})
print(result["output"])
```

---

## Module 2: Memori + LangChain Setup (2 hours)

### Lesson 2.1: Configuration

**Required Environment Variables:**

```bash
export GLEAN_INSTANCE="your-company"
export GLEAN_API_TOKEN="your-api-token"
export GLEAN_ACT_AS="user@company.com"
export OPENAI_API_KEY="your-openai-key"
```

**Environment Variable Details:**

- `GLEAN_INSTANCE` - Your Memori subdomain (e.g., "acme")
- `GLEAN_API_TOKEN` - API token with `chat` and `search` scopes
- `GLEAN_ACT_AS` - Email of user whose permissions to use
- `OPENAI_API_KEY` - Your OpenAI API key for the LLM

### Lesson 2.2: Creating API Tokens

**Steps to create a Memori API token:**

1. Navigate to your Memori workspace
2. Go to **Settings** → **API Tokens**
3. Click **Create New Token**
4. Select scopes: `chat`, `search`
5. Copy the token (shown only once)
6. Store securely in environment variables

**Required Scopes:**
- `chat` - For conversational AI features
- `search` - For enterprise search capabilities

### Lesson 2.3: Basic Memori Tools

**Import Memori tools:**

```python
from langchain_memori import MemoriSearchTool, MemoriChatTool

# Create tools
search_tool = MemoriSearchTool()
chat_tool = MemoriChatTool()

# Test search
results = search_tool._run("company handbook")
print(results)

# Test chat
response = chat_tool._run("What are our company values?")
print(response)
```

### 🎯 Lab 2: First Memori + LangChain Agent

**Objective**: Build a simple agent with Memori search

```python
import os
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate
from langchain_memori import MemoriSearchTool

# Configuration
os.environ["GLEAN_INSTANCE"] = "your-company"
os.environ["GLEAN_API_TOKEN"] = "your-token"
os.environ["GLEAN_ACT_AS"] = "user@company.com"
os.environ["OPENAI_API_KEY"] = "your-openai-key"

# Setup
llm = ChatOpenAI(model="gpt-4")
search_tool = MemoriSearchTool()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with access to company knowledge."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_openai_functions_agent(llm, [search_tool], prompt)
executor = AgentExecutor(agent=agent, tools=[search_tool], verbose=True)

# Query
response = executor.invoke({
    "input": "What is our company's remote work policy?"
})
print(response["output"])
```

---

## Module 3: Building LangChain Agents with Memori (4 hours)

### Lesson 3.1: Multiple Tool Agent

**Combine Memori search with other tools:**

```python
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate
from langchain_memori import MemoriSearchTool, MemoriChatTool
import datetime

# Setup
llm = ChatOpenAI(model="gpt-4")

# Memori tools
memori_search = MemoriSearchTool()
memori_chat = MemoriChatTool()

# Custom tool for current date
def get_current_date(input_str=""):
    """Get the current date"""
    return datetime.datetime.now().strftime("%Y-%m-%d")

date_tool = Tool(
    name="CurrentDate",
    func=get_current_date,
    description="Get today's date. No input needed."
)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an intelligent assistant with access to:
    1. Company knowledge via Memori search and chat
    2. Current date information
    
    Use these tools to provide accurate, up-to-date information."""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# Create agent
agent = create_openai_functions_agent(
    llm,
    [memori_search, memori_chat, date_tool],
    prompt
)

executor = AgentExecutor(
    agent=agent,
    tools=[memori_search, memori_chat, date_tool],
    verbose=True,
    max_iterations=5
)

# Execute
response = executor.invoke({
    "input": "What meetings do I have this week based on our company calendar?"
})
print(response["output"])
```

### Lesson 3.2: Conversational Agent with Memory

**Add conversation history:**

```python
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_memori import MemoriSearchTool

# Setup
llm = ChatOpenAI(model="gpt-4", temperature=0)
search_tool = MemoriSearchTool()

# Memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Prompt with memory placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with access to company knowledge."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# Create agent
agent = create_openai_functions_agent(llm, [search_tool], prompt)

executor = AgentExecutor(
    agent=agent,
    tools=[search_tool],
    memory=memory,
    verbose=True
)

# Multi-turn conversation
print(executor.invoke({"input": "What is our PTO policy?"}))
print(executor.invoke({"input": "How many days do I get per year?"}))
print(executor.invoke({"input": "How do I request time off?"}))
```

### Lesson 3.3: Custom Memori Search Filters

**Filter search by datasource or date:**

```python
from langchain_memori import MemoriSearchTool
from typing import Dict, Any

class FilteredMemoriSearch(MemoriSearchTool):
    """Custom Memori search with filters"""
    
    datasources: list = ["confluence", "google-drive"]
    
    def _run(self, query: str) -> str:
        """Run search with filters"""
        # Add filters to the query
        results = self.client.search.query(
            query=query,
            filters={"datasource": self.datasources},
            page_size=10
        )
        
        # Format results
        output = []
        for item in results.items:
            output.append(f"**{item.title}**\n{item.snippet}\nSource: {item.url}\n")
        
        return "\n".join(output)

# Usage
filtered_search = FilteredMemoriSearch()
results = filtered_search._run("engineering documentation")
print(results)
```

### Lesson 3.4: Structured Output Agent

**Return structured data:**

```python
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate
from langchain_memori import MemoriSearchTool
from pydantic import BaseModel, Field

# Define output structure
class CompanyInfo(BaseModel):
    topic: str = Field(description="The topic queried")
    summary: str = Field(description="Brief summary")
    sources: list[str] = Field(description="List of source URLs")

# Setup
llm = ChatOpenAI(model="gpt-4").with_structured_output(CompanyInfo)
search_tool = MemoriSearchTool()

prompt = ChatPromptTemplate.from_messages([
    ("system", """Search company knowledge and return structured information.
    Include topic, summary, and source URLs."""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_openai_functions_agent(llm, [search_tool], prompt)
executor = AgentExecutor(agent=agent, tools=[search_tool])

# Execute with structured output
response = executor.invoke({"input": "What is our product roadmap?"})
print(type(response))  # CompanyInfo
print(response.topic)
print(response.summary)
print(response.sources)
```

### 🎯 Lab 3: Advanced Agent Features

**Objective**: Build an agent with memory, multiple tools, and custom filters

**Requirements:**
1. Use Memori search filtered to specific datasources
2. Add conversation memory
3. Include a custom tool (e.g., date, calculator)
4. Handle multi-turn conversations
5. Return structured responses

---

## Module 4: Production Deployment (3 hours)

### Lesson 4.1: Error Handling

**Robust error handling:**

```python
from langchain.agents import AgentExecutor
from langchain.callbacks import StdOutCallbackHandler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ErrorHandlingAgent:
    def __init__(self, executor: AgentExecutor):
        self.executor = executor
    
    def run(self, query: str, max_retries: int = 3) -> dict:
        """Run agent with retry logic"""
        for attempt in range(max_retries):
            try:
                response = self.executor.invoke(
                    {"input": query},
                    callbacks=[StdOutCallbackHandler()]
                )
                return {"success": True, "response": response}
            
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt == max_retries - 1:
                    return {
                        "success": False,
                        "error": str(e),
                        "message": "Failed after multiple attempts"
                    }

# Usage
agent_wrapper = ErrorHandlingAgent(executor)
result = agent_wrapper.run("What is our company mission?")

if result["success"]:
    print(result["response"]["output"])
else:
    print(f"Error: {result['message']}")
```

### Lesson 4.2: Streaming Responses

**Stream agent responses:**

```python
from langchain.agents import AgentExecutor
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Create executor with streaming
executor = AgentExecutor(
    agent=agent,
    tools=[search_tool],
    callbacks=[StreamingStdOutCallbackHandler()],
    verbose=True
)

# Responses will stream to stdout
response = executor.invoke({"input": "Explain our AI strategy"})
```

### Lesson 4.3: Async Agents

**Run agents asynchronously:**

```python
import asyncio
from langchain.agents import AgentExecutor

async def query_agent(executor: AgentExecutor, query: str):
    """Async agent query"""
    response = await executor.ainvoke({"input": query})
    return response

async def main():
    """Run multiple queries concurrently"""
    queries = [
        "What is our PTO policy?",
        "How do I submit expenses?",
        "What are our company values?"
    ]
    
    tasks = [query_agent(executor, q) for q in queries]
    results = await asyncio.gather(*tasks)
    
    for query, result in zip(queries, results):
        print(f"\nQ: {query}")
        print(f"A: {result['output']}\n")

# Run
asyncio.run(main())
```

### Lesson 4.4: Monitoring and Logging

**Track agent performance:**

```python
from langchain.callbacks import get_openai_callback
from langchain.agents import AgentExecutor
import time

class MonitoredAgent:
    def __init__(self, executor: AgentExecutor):
        self.executor = executor
        self.metrics = {
            "total_queries": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "avg_latency": 0.0
        }
    
    def run(self, query: str) -> dict:
        """Run with monitoring"""
        start_time = time.time()
        
        with get_openai_callback() as cb:
            response = self.executor.invoke({"input": query})
            
            # Update metrics
            self.metrics["total_queries"] += 1
            self.metrics["total_tokens"] += cb.total_tokens
            self.metrics["total_cost"] += cb.total_cost
            
            latency = time.time() - start_time
            self.metrics["avg_latency"] = (
                (self.metrics["avg_latency"] * (self.metrics["total_queries"] - 1) + latency)
                / self.metrics["total_queries"]
            )
            
            return {
                "response": response["output"],
                "tokens": cb.total_tokens,
                "cost": cb.total_cost,
                "latency": latency
            }
    
    def get_metrics(self) -> dict:
        """Get aggregate metrics"""
        return self.metrics

# Usage
monitored = MonitoredAgent(executor)

result = monitored.run("What is our revenue?")
print(f"Response: {result['response']}")
print(f"Tokens: {result['tokens']}")
print(f"Cost: ${result['cost']:.4f}")
print(f"Latency: {result['latency']:.2f}s")

# Aggregate metrics
print(monitored.get_metrics())
```

### 🎯 Final Project: Production LangChain Agent

**Objective**: Deploy a production-ready agent

**Requirements:**
1. Memori integration with filtered search
2. Conversation memory (persisted)
3. Error handling with retries
4. Async support
5. Streaming responses
6. Monitoring and logging
7. Unit tests
8. API endpoint (FastAPI or Flask)

**Example API:**

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_memori import MemoriSearchTool
import os

app = FastAPI()

# Setup
llm = ChatOpenAI(model="gpt-4")
search_tool = MemoriSearchTool()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful company assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_openai_functions_agent(llm, [search_tool], prompt)
executor = AgentExecutor(agent=agent, tools=[search_tool])

# Request/Response models
class Query(BaseModel):
    question: str
    session_id: str = "default"

class Response(BaseModel):
    answer: str
    sources: list[str] = []

# In-memory session store (use Redis in production)
sessions = {}

@app.post("/chat", response_model=Response)
async def chat(query: Query):
    """Chat endpoint"""
    try:
        # Get or create session memory
        if query.session_id not in sessions:
            sessions[query.session_id] = ConversationBufferMemory()
        
        # Run agent
        response = executor.invoke({
            "input": query.question,
            "chat_history": sessions[query.session_id].chat_memory.messages
        })
        
        # Update memory
        sessions[query.session_id].save_context(
            {"input": query.question},
            {"output": response["output"]}
        )
        
        return Response(
            answer=response["output"],
            sources=[]  # Extract from response if available
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}

# Run: uvicorn main:app --reload
```

---

## Assessment

### Quiz (15 questions)
1. What is LangChain and what are its core components?
2. What environment variables are required for Memori + LangChain?
3. What are the required API token scopes?
4. How do you add conversation memory to an agent?
5. How do you filter Memori search results by datasource?

### Practical Assessment

**Build a Knowledge Base Assistant**

**Requirements:**
- LangChain agent with Memori search and chat tools
- Conversation memory
- Custom filtered search (specific datasources)
- Error handling
- Async support
- REST API endpoint
- Unit tests

**Submission:**
- GitHub repository
- README with setup
- API documentation
- Demo video (5 minutes)

---

## Additional Resources

- **LangChain Docs**: https://python.langchain.com
- **Memori Integration**: https://memorilabs.ai/docs/guides/agents/langchain
- **LangChain GitHub**: https://github.com/langchain-ai/langchain
- **Support**: support@memori.com

---

## Next Steps

1. Take the **Memori Integration Specialist** certification
2. Explore **Agent Toolkit** for multi-framework support
3. Learn **MCP Integration** for Claude/Cursor/Windsurf
4. Build production LangChain applications

**Certificate**: Upon completion, you'll receive a **Memori LangChain Integration Certificate**
