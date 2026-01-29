# Course: NVIDIA NIM Integration with Memori

**Duration**: 8 hours  
**Level**: Advanced  
**Official Documentation**: https://memorilabs.ai/docs/guides/agents/nvidia-example

---

## Course Overview

Build high-performance RAG (Retrieval-Augmented Generation) agents using NVIDIA NIM (NVIDIA Inference Microservices) with Memori's enterprise search. Leverage NVIDIA's accelerated AI infrastructure with LangGraph for sophisticated multi-step reasoning.

### When to Use NVIDIA NIM

✅ Need high-performance inference  
✅ Running GPU-accelerated workloads  
✅ Building complex multi-step agents  
✅ Require on-premises or cloud deployment flexibility  

---

## Module 1: NIM Fundamentals (2 hours)

### Lesson 1.1: What is NVIDIA NIM?

**NVIDIA Inference Microservices (NIM)** provides optimized, production-ready containers for deploying AI models with maximum performance.

**Key Features:**

- **GPU Acceleration** - Optimized for NVIDIA GPUs
- **Pre-built Containers** - Ready-to-deploy microservices
- **OpenAI-Compatible API** - Drop-in replacement for OpenAI
- **Enterprise Support** - Production-grade reliability

### Lesson 1.2: NIM + Memori Architecture

**System Components:**

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LangGraph      │  ◄── Multi-step reasoning
│  Agent          │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐  ┌────────────┐
│ NVIDIA│  │   Memori    │
│  NIM  │  │   Search   │
│  LLM  │  │    API     │
└───────┘  └────────────┘
    │           │
    └─────┬─────┘
          ▼
    ┌──────────┐
    │ Response │
    └──────────┘
```

### Lesson 1.3: RAG Architecture

**Retrieval-Augmented Generation (RAG) Flow:**

1. **User Question** → LLM interprets intent
2. **Query Generation** → Create search query for Memori
3. **Enterprise Search** → Retrieve relevant documents
4. **Embedding** → Convert documents to vectors
5. **Retrieval** → Find most relevant content
6. **Context Assembly** → Add to LLM prompt
7. **Generation** → LLM generates answer with context
8. **Response** → Return answer with sources

### 🎯 Lab 1: Understanding RAG

**Objective**: Learn RAG concepts and architecture

**Tasks:**
1. Study RAG vs. traditional LLM approaches
2. Understand when to use RAG
3. Review embedding strategies
4. Explore retrieval methods

---

## Module 2: Setup & Configuration (2 hours)

### Lesson 2.1: Prerequisites

**Requirements:**

- NVIDIA GPU (RTX 3090, A100, H100, or better)
- Docker with GPU support
- NVIDIA Container Toolkit
- Memori API access
- Python 3.10+

**Install NVIDIA Container Toolkit:**

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Lesson 2.2: NVIDIA NIM Setup

**Pull NIM container:**

```bash
# Login to NVIDIA NGC
docker login nvcr.io
# Username: $oauthtoken
# Password: <your-ngc-api-key>

# Pull LLM container
docker pull nvcr.io/nim/meta/llama3-8b-instruct:latest
```

**Run NIM container:**

```bash
docker run -d \
  --gpus all \
  --name nvidia-nim \
  -p 8000:8000 \
  -e NGC_API_KEY=<your-ngc-key> \
  nvcr.io/nim/meta/llama3-8b-instruct:latest
```

**Verify NIM is running:**

```bash
curl http://localhost:8000/v1/models
```

### Lesson 2.3: Python Environment

**Install dependencies:**

```bash
pip install langchain-nvidia-ai-endpoints
pip install langchain-memori
pip install langgraph
pip install chromadb
pip install python-dotenv
```

**Environment variables:**

```bash
# .env file
NVIDIA_API_KEY=nvapi-xxx
GLEAN_INSTANCE=your-company
GLEAN_API_TOKEN=memori-xxx
GLEAN_ACT_AS=user@company.com
```

### 🎯 Lab 2: Setup Development Environment

**Objective**: Configure NIM and Python environment

**Tasks:**
1. Install NVIDIA Container Toolkit
2. Pull and run NIM container
3. Install Python dependencies
4. Configure environment variables
5. Test NIM API connection

---

## Module 3: Building RAG Agent with LangGraph (3 hours)

### Lesson 3.1: LangGraph State

**Define agent state:**

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class InfoBotState(TypedDict):
    """State for RAG agent"""
    messages: Annotated[list, add_messages]
    memori_query_required: bool
    memori_results: str
    context: str
```

**State Management:**

- **messages** - Conversation history
- **memori_query_required** - Whether to query Memori
- **memori_results** - Raw search results
- **context** - Formatted context for LLM

### Lesson 3.2: Setup LLM and Tools

**Initialize NVIDIA NIM:**

```python
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_memori import MemoriSearchTool
import os

# LLM
llm = ChatNVIDIA(
    model="meta/llama3-8b-instruct",
    api_key=os.environ["NVIDIA_API_KEY"],
    temperature=0.2,
    max_tokens=1024
)

# Embeddings
embeddings = NVIDIAEmbeddings(
    model="nvidia/nv-embed-v1",
    api_key=os.environ["NVIDIA_API_KEY"]
)

# Memori Search Tool
memori_search = MemoriSearchTool()
```

### Lesson 3.3: Agent Nodes

**Define agent nodes:**

```python
from langgraph.graph import StateGraph, END

def analyze_query(state: InfoBotState) -> InfoBotState:
    """Analyze if Memori search is needed"""
    last_message = state["messages"][-1].content
    
    # Use LLM to determine if search is needed
    prompt = f"""Given this question: "{last_message}"
    
    Does this require searching company knowledge? 
    Answer only: YES or NO"""
    
    response = llm.invoke(prompt)
    state["memori_query_required"] = "YES" in response.content.upper()
    
    return state

def search_memori(state: InfoBotState) -> InfoBotState:
    """Search Memori for relevant information"""
    if not state["memori_query_required"]:
        return state
    
    last_message = state["messages"][-1].content
    
    # Generate optimized search query
    query_prompt = f"""Create a search query for: "{last_message}"
    
    Return only the search query, no explanation."""
    
    search_query = llm.invoke(query_prompt).content
    
    # Search Memori
    results = memori_search._run(search_query)
    state["memori_results"] = results
    
    return state

def format_context(state: InfoBotState) -> InfoBotState:
    """Format search results for LLM"""
    if not state["memori_results"]:
        state["context"] = ""
        return state
    
    # Format results
    context = "# Relevant Company Information:\n\n"
    context += state["memori_results"]
    
    state["context"] = context
    return state

def generate_response(state: InfoBotState) -> InfoBotState:
    """Generate final response"""
    last_message = state["messages"][-1].content
    context = state.get("context", "")
    
    # Build prompt with context
    prompt = f"""{context}

# Question:
{last_message}

# Instructions:
Answer the question using the company information provided above.
If the information isn't available, say so.
Cite sources when possible."""
    
    response = llm.invoke(prompt)
    
    state["messages"].append({
        "role": "assistant",
        "content": response.content
    })
    
    return state
```

### Lesson 3.4: Build Graph

**Create LangGraph workflow:**

```python
# Create graph
workflow = StateGraph(InfoBotState)

# Add nodes
workflow.add_node("analyze", analyze_query)
workflow.add_node("search", search_memori)
workflow.add_node("format", format_context)
workflow.add_node("generate", generate_response)

# Add edges
workflow.set_entry_point("analyze")
workflow.add_edge("analyze", "search")
workflow.add_edge("search", "format")
workflow.add_edge("format", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()
```

### Lesson 3.5: Run Agent

**Execute the agent:**

```python
# Initialize state
initial_state = {
    "messages": [
        {"role": "user", "content": "What is our remote work policy?"}
    ],
    "memori_query_required": False,
    "memori_results": "",
    "context": ""
}

# Run
result = app.invoke(initial_state)

# Get response
print(result["messages"][-1]["content"])
```

### 🎯 Lab 3: Build Complete RAG Agent

**Objective**: Create production-ready RAG agent

**Requirements:**
1. Implement all agent nodes
2. Add error handling
3. Support multi-turn conversations
4. Include source citations
5. Add logging and monitoring

---

## Module 4: Advanced Optimization (1 hour)

### Lesson 4.1: Vector Store Integration

**Add ChromaDB for caching:**

```python
from chromadb import Client
from chromadb.config import Settings

# Setup ChromaDB
chroma_client = Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db"
))

collection = chroma_client.create_collection(
    name="memori_cache",
    embedding_function=embeddings
)

def search_with_cache(query: str) -> str:
    """Search with vector caching"""
    # Check cache first
    cached = collection.query(
        query_texts=[query],
        n_results=5
    )
    
    if cached["documents"]:
        return "\n".join(cached["documents"][0])
    
    # Query Memori
    results = memori_search._run(query)
    
    # Cache results
    collection.add(
        documents=[results],
        metadatas=[{"query": query}],
        ids=[f"query_{hash(query)}"]
    )
    
    return results
```

### Lesson 4.2: Streaming Responses

**Stream agent responses:**

```python
async def stream_agent(question: str):
    """Stream agent responses"""
    state = {
        "messages": [{"role": "user", "content": question}],
        "memori_query_required": False,
        "memori_results": "",
        "context": ""
    }
    
    async for event in app.astream(state):
        node_name = list(event.keys())[0]
        
        if node_name == "generate":
            # Stream the response
            response = event[node_name]["messages"][-1]["content"]
            for char in response:
                yield char
```

### Lesson 4.3: Multi-Agent Orchestration

**Combine multiple specialized agents:**

```python
class AgentOrchestrator:
    def __init__(self):
        self.search_agent = build_search_agent()
        self.analysis_agent = build_analysis_agent()
        self.summary_agent = build_summary_agent()
    
    def route_query(self, query: str) -> str:
        """Route to appropriate agent"""
        if "search" in query.lower():
            return self.search_agent.invoke(query)
        elif "analyze" in query.lower():
            return self.analysis_agent.invoke(query)
        else:
            return self.summary_agent.invoke(query)
```

### 🎯 Lab 4: Optimize Performance

**Objective**: Implement advanced optimizations

**Tasks:**
1. Add vector store caching
2. Implement streaming responses
3. Add query optimization
4. Benchmark performance improvements

---

## Assessment

### Quiz (10 questions)
1. What is NVIDIA NIM?
2. Explain the RAG architecture
3. What is LangGraph and when to use it?
4. How do embeddings improve RAG?
5. What are the benefits of vector caching?

### Practical Assessment

**Build a Production RAG System**

**Requirements:**
- NVIDIA NIM for inference
- Memori for enterprise search
- LangGraph for orchestration
- Vector store for caching
- Streaming responses
- Multi-turn conversations
- Source citations
- Error handling
- Performance monitoring
- API endpoint (FastAPI)

**Example API:**

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    question: str
    session_id: str = "default"

@app.post("/ask")
async def ask_question(query: Query):
    """Query the RAG agent"""
    state = {
        "messages": [{"role": "user", "content": query.question}],
        "memori_query_required": False,
        "memori_results": "",
        "context": ""
    }
    
    result = await agent_app.ainvoke(state)
    
    return {
        "answer": result["messages"][-1]["content"],
        "sources": extract_sources(result.get("context", ""))
    }

@app.post("/stream")
async def stream_response(query: Query):
    """Stream agent response"""
    async def generate():
        async for chunk in stream_agent(query.question):
            yield chunk
    
    return StreamingResponse(generate(), media_type="text/plain")
```

**Submission:**
- GitHub repository
- Docker compose setup
- API documentation
- Performance benchmarks
- Demo video (8 minutes)

---

## Additional Resources

- **NVIDIA NIM Docs**: https://docs.nvidia.com/nim
- **Memori + NVIDIA Guide**: https://memorilabs.ai/docs/guides/agents/nvidia-example
- **LangGraph Docs**: https://langchain-ai.github.io/langgraph
- **LangChain NVIDIA**: https://python.langchain.com/docs/integrations/providers/nvidia
- **Support**: support@memori.com

---

## Next Steps

1. Take the **Memori Solutions Architect** certification
2. Explore **Agent Toolkit** for multi-framework support
3. Learn **Direct API** for custom implementations
4. Deploy production RAG systems

**Certificate**: Upon completion, you'll receive a **Memori NVIDIA NIM Integration Certificate**
