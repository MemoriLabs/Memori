# Course: Direct API Integration with Memori

**Duration**: 16 hours  
**Level**: Intermediate  
**Official Documentation**: https://memorilabs.ai/docs/guides/agents/direct-api

---

## Course Overview

Build custom agents using Memori's REST APIs with our official client libraries. This approach gives you full control over the agent logic while leveraging Memori's search and chat capabilities.

### When to Use Direct API Integration

✅ Custom business logic requirements  
✅ Integration with existing systems  
✅ Fine-grained control over agent behavior  
✅ Building on non-standard frameworks  

---

## Module 1: API Fundamentals (3 hours)

### Lesson 1.1: Understanding Memori's APIs

**Memori provides two primary API surfaces:**

#### Client API (User-Facing Operations)
- **Chat API** - Conversational AI with enterprise context
- **Search API** - Enterprise search capabilities
- **Agents API** - Manage pre-built agents
- **Documents API** - Access and manipulate documents
- **Entities API** - People, projects, organizational data

#### Indexing API (Data Ingestion)
- **Documents** - Index custom content and files
- **People** - Sync organizational directories
- **Permissions** - Fine-grained access control
- **Datasources** - Configure data connectors
- **Activity** - Track user interactions

### Lesson 1.2: Authentication

**API Token Types:**

1. **Service Account Tokens** - For backend integrations
2. **User-Scoped Tokens** - Act on behalf of users
3. **OAuth Tokens** - For user-interactive applications

**Required Scopes:**
- `chat` - For Chat API access
- `search` - For Search API access
- `agents` - For Agents API access

**Setup:**
```bash
export GLEAN_INSTANCE="your-company"
export GLEAN_API_TOKEN="your-api-token"
```

### Lesson 1.3: Rate Limits

**Default Limits:**
- 100 requests per minute per token
- 1000 requests per hour per token

**Best Practices:**
- Implement exponential backoff
- Cache responses when appropriate
- Use batch operations where available

### 🎯 Lab 1: Setup Your Environment

**Objective**: Configure authentication and make your first API call

```python
import os
import requests

# Configuration
GLEAN_INSTANCE = os.environ["GLEAN_INSTANCE"]
GLEAN_API_TOKEN = os.environ["GLEAN_API_TOKEN"]

# Test connection
url = f"https://{GLEAN_INSTANCE}-be.memori.com/api/v1/ping"
headers = {
    "Authorization": f"Bearer {GLEAN_API_TOKEN}",
    "Content-Type": "application/json"
}

response = requests.get(url, headers=headers)
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
```

---

## Module 2: Client Libraries (4 hours)

### Lesson 2.1: Python SDK

**Installation:**
```bash
pip install memori-api-client
```

**Basic Usage:**
```python
from memori import MemoriClient
import os

client = MemoriClient(
    api_key=os.environ["GLEAN_API_TOKEN"],
    instance=os.environ["GLEAN_INSTANCE"]
)

# Test the connection
print("Connected to Memori!")
```

### Lesson 2.2: TypeScript SDK

**Installation:**
```bash
npm install @memoriwork/api-client
```

**Basic Usage:**
```typescript
import { MemoriClient } from '@memoriwork/api-client';

const client = new MemoriClient({
  apiKey: process.env.GLEAN_API_TOKEN!,
  instance: process.env.GLEAN_INSTANCE!
});

// Test the connection
async function test() {
  console.log('Connected to Memori!');
}

test();
```

### Lesson 2.3: Go SDK

**Installation:**
```bash
go get github.com/memoriwork/api-client-go
```

**Basic Usage:**
```go
package main

import (
    "os"
    "github.com/memoriwork/api-client-go"
)

func main() {
    client := memori.NewClient(
        os.Getenv("GLEAN_API_TOKEN"),
        os.Getenv("GLEAN_INSTANCE"),
    )
    
    // Test the connection
    println("Connected to Memori!")
}
```

### Lesson 2.4: Java SDK

**Maven Configuration:**
```xml
<dependency>
    <groupId>com.memori</groupId>
    <artifactId>api-client</artifactId>
    <version>1.0.0</version>
</dependency>
```

**Basic Usage:**
```java
import com.memori.MemoriClient;

public class Main {
    public static void main(String[] args) {
        MemoriClient client = new MemoriClient(
            System.getenv("GLEAN_API_TOKEN"),
            System.getenv("GLEAN_INSTANCE")
        );
        
        System.out.println("Connected to Memori!");
    }
}
```

### 🎯 Lab 2: Client Library Setup

**Objective**: Install and configure your preferred SDK

**Tasks:**
1. Choose your language (Python, TypeScript, Go, or Java)
2. Install the SDK
3. Create a simple connection test
4. Verify authentication works

---

## Module 3: Core APIs (5 hours)

### Lesson 3.1: Chat API

**Create a chat session:**

```python
from memori import MemoriClient
import models

client = MemoriClient(
    api_key=os.environ["GLEAN_API_TOKEN"],
    instance=os.environ["GLEAN_INSTANCE"]
)

# Simple chat request
response = client.client.chat.create(
    messages=[
        models.ChatMessageFragment(
            role="user",
            content="What are our Q4 objectives?"
        )
    ]
)

print(response.message.content)
```

**Streaming responses:**

```python
# Stream chat responses
stream = client.client.chat.create(
    messages=[
        models.ChatMessageFragment(
            role="user",
            content="Explain our company's AI strategy"
        )
    ],
    stream=True
)

for chunk in stream:
    print(chunk.content, end='', flush=True)
```

### Lesson 3.2: Search API

**Basic search:**

```python
# Search company knowledge
results = client.client.search.query(
    query="customer onboarding process",
    page_size=10
)

for item in results.items:
    print(f"Title: {item.title}")
    print(f"URL: {item.url}")
    print(f"Snippet: {item.snippet}")
    print("---")
```

**Advanced search with filters:**

```python
# Search with filters
results = client.client.search.query(
    query="product roadmap",
    filters={
        "datasource": ["confluence", "google-drive"],
        "dateRange": {
            "start": "2024-01-01",
            "end": "2024-12-31"
        }
    },
    page_size=20
)
```

### Lesson 3.3: Agents API

**List available agents:**

```python
# Get all agents
agents = client.client.agents.list()

for agent in agents.items:
    print(f"Agent: {agent.name}")
    print(f"Description: {agent.description}")
```

**Run an agent:**

```python
# Execute an agent
response = client.client.agents.create_and_stream_run(
    agent_id="support-agent-001",
    messages=[
        models.ChatMessageFragment(
            role="user",
            content="How do I reset my password?"
        )
    ]
)

for chunk in response:
    print(chunk.content, end='', flush=True)
```

### Lesson 3.4: Documents API

**Retrieve a document:**

```python
# Get document by ID
document = client.client.documents.get(
    document_id="doc-12345"
)

print(f"Title: {document.title}")
print(f"Content: {document.body}")
print(f"Author: {document.author}")
```

### Lesson 3.5: Entities API

**Search for people:**

```python
# Find employees
people = client.client.entities.search_people(
    query="engineering manager",
    filters={
        "department": ["Engineering"],
        "location": ["San Francisco"]
    }
)

for person in people.items:
    print(f"Name: {person.name}")
    print(f"Title: {person.title}")
    print(f"Email: {person.email}")
```

### 🎯 Lab 3: API Exploration

**Objective**: Use each core API

**Tasks:**
1. Perform a chat request
2. Execute a search query with filters
3. List available agents
4. Retrieve a document
5. Search for people in your organization

---

## Module 4: Building Your First Agent (4 hours)

### Lesson 4.1: Agent Architecture

**Components of a Memori Agent:**

1. **Input Handler** - Receives user queries
2. **Search Context** - Retrieves relevant information
3. **Response Generator** - Builds conversational responses
4. **History Manager** - Tracks conversation state
5. **Error Handler** - Manages failures gracefully

### Lesson 4.2: Customer Support Agent

**Project: Build a customer support agent**

```python
from memori import MemoriClient
import models
from typing import List, Dict

class SupportAgent:
    def __init__(self, memori_client: MemoriClient):
        self.client = memori_client
        self.conversation_history: List[Dict] = []
    
    def search_knowledge_base(self, query: str) -> List[Dict]:
        """Search for relevant support articles"""
        results = self.client.client.search.query(
            query=query,
            filters={
                "datasource": ["zendesk", "confluence"]
            },
            page_size=5
        )
        
        return [
            {
                "title": item.title,
                "url": item.url,
                "snippet": item.snippet
            }
            for item in results.items
        ]
    
    def generate_response(self, user_query: str) -> str:
        """Generate a response using Memori Chat API"""
        # First, search for relevant context
        context = self.search_knowledge_base(user_query)
        
        # Build context message
        context_text = "\n".join([
            f"- {item['title']}: {item['snippet']}"
            for item in context
        ])
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_query
        })
        
        # Generate response with context
        response = self.client.client.chat.create(
            messages=[
                models.ChatMessageFragment(
                    role="system",
                    content=f"You are a customer support agent. Use this context to answer:\n{context_text}"
                ),
                *[
                    models.ChatMessageFragment(**msg)
                    for msg in self.conversation_history
                ]
            ]
        )
        
        # Save assistant response
        self.conversation_history.append({
            "role": "assistant",
            "content": response.message.content
        })
        
        return response.message.content
    
    def reset_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []

# Usage
client = MemoriClient(
    api_key=os.environ["GLEAN_API_TOKEN"],
    instance=os.environ["GLEAN_INSTANCE"]
)

agent = SupportAgent(client)

# Interact with the agent
response = agent.generate_response("How do I reset my password?")
print(response)

response = agent.generate_response("What about two-factor authentication?")
print(response)
```

### Lesson 4.3: Error Handling

**Implement robust error handling:**

```python
import time
from requests.exceptions import RequestException

def with_retry(func, max_retries=3, backoff=2):
    """Retry decorator with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func()
        except RequestException as e:
            if attempt == max_retries - 1:
                raise
            wait_time = backoff ** attempt
            print(f"Retry {attempt + 1}/{max_retries} after {wait_time}s")
            time.sleep(wait_time)

# Usage
response = with_retry(
    lambda: client.client.search.query(query="test")
)
```

### Lesson 4.4: Conversation Management

**Track conversation state:**

```python
class ConversationManager:
    def __init__(self, max_history=10):
        self.history = []
        self.max_history = max_history
    
    def add_message(self, role: str, content: str):
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        
        # Keep only recent messages
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_messages(self):
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self.history
        ]
    
    def clear(self):
        self.history = []
```

### 🎯 Final Project: Production-Ready Agent

**Objective**: Build a complete agent with all features

**Requirements:**
1. Search context retrieval
2. Conversational responses
3. Conversation history management
4. Error handling with retries
5. Logging and monitoring
6. Unit tests

**Example Complete Agent:**

```python
import logging
from typing import List, Dict, Optional
from memori import MemoriClient
import models

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionAgent:
    """Production-ready Memori agent with full features"""
    
    def __init__(
        self,
        memori_client: MemoriClient,
        agent_name: str = "Default Agent",
        max_context_items: int = 5,
        max_history: int = 10
    ):
        self.client = memori_client
        self.name = agent_name
        self.max_context_items = max_context_items
        self.conversation_manager = ConversationManager(max_history)
        logger.info(f"Initialized {self.name}")
    
    def search_context(
        self,
        query: str,
        datasources: Optional[List[str]] = None
    ) -> List[Dict]:
        """Search for relevant context with error handling"""
        try:
            results = with_retry(
                lambda: self.client.client.search.query(
                    query=query,
                    filters={"datasource": datasources} if datasources else {},
                    page_size=self.max_context_items
                )
            )
            
            logger.info(f"Found {len(results.items)} context items")
            return [
                {
                    "title": item.title,
                    "url": item.url,
                    "snippet": item.snippet
                }
                for item in results.items
            ]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def respond(
        self,
        user_query: str,
        datasources: Optional[List[str]] = None
    ) -> Dict:
        """Generate a response with full context"""
        try:
            # Get relevant context
            context = self.search_context(user_query, datasources)
            
            # Build context text
            context_text = "\n\n".join([
                f"**{item['title']}**\n{item['snippet']}\nSource: {item['url']}"
                for item in context
            ])
            
            # Add user message to history
            self.conversation_manager.add_message("user", user_query)
            
            # Generate response
            response = self.client.client.chat.create(
                messages=[
                    models.ChatMessageFragment(
                        role="system",
                        content=f"You are {self.name}. Use this context:\n\n{context_text}"
                    ),
                    *[
                        models.ChatMessageFragment(**msg)
                        for msg in self.conversation_manager.get_messages()
                    ]
                ]
            )
            
            # Add assistant message to history
            assistant_message = response.message.content
            self.conversation_manager.add_message("assistant", assistant_message)
            
            logger.info(f"Generated response ({len(assistant_message)} chars)")
            
            return {
                "response": assistant_message,
                "context": context,
                "conversation_id": id(self.conversation_manager)
            }
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return {
                "response": "I'm sorry, I encountered an error. Please try again.",
                "context": [],
                "error": str(e)
            }
    
    def reset(self):
        """Reset conversation state"""
        self.conversation_manager.clear()
        logger.info("Conversation reset")

# Usage example
if __name__ == "__main__":
    client = MemoriClient(
        api_key=os.environ["GLEAN_API_TOKEN"],
        instance=os.environ["GLEAN_INSTANCE"]
    )
    
    agent = ProductionAgent(
        memori_client=client,
        agent_name="Support Agent"
    )
    
    # Interact
    result = agent.respond(
        "How do I configure SSO?",
        datasources=["confluence", "zendesk"]
    )
    
    print(f"Response: {result['response']}")
    print(f"Context sources: {len(result['context'])}")
```

---

## Assessment

### Quiz (20 questions)
1. What are the two primary API surfaces in Memori?
2. Which API would you use to build a conversational interface?
3. What authentication scopes are required for chat access?
4. How do you implement rate limit handling?
5. What's the difference between service account and user-scoped tokens?

### Practical Assessment

**Build a Sales Assistant Agent**

**Requirements:**
- Search Salesforce and Confluence for customer information
- Answer questions about deals and accounts
- Track conversation history
- Include error handling
- Add logging

**Submission**:
- Code repository (GitHub)
- README with setup instructions
- Unit tests (minimum 80% coverage)
- Demo video (5 minutes)

---

## Additional Resources

- **API Reference**: https://memorilabs.ai/docs/api/client-api
- **Client Libraries**: https://memorilabs.ai/docs/libraries/api-clients
- **GitHub Examples**: https://github.com/memoriwork
- **Support**: support@memori.com

---

## Next Steps

After completing this course, you can:
1. Take the **Memori API Developer** certification
2. Move to **LangChain Integration** course
3. Explore **Agent Toolkit** for multi-framework support
4. Build production applications with Memori APIs

**Certificate**: Upon completion and passing assessment, you'll receive a **Memori Direct API Integration Certificate**
