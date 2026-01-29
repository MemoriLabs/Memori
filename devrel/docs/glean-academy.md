# Memori Academy - API & Integration Training

## Overview

Memori Academy provides comprehensive training on building AI agents, apps, and integrations using Memori's Work AI Platform. All courses are aligned with our official developer documentation at [memorilabs.ai/docs](https://memorilabs.ai/docs).

**Access**: [learning-paths.html](../learning-paths.html)

---

## 🎓 Learning Paths

### Path 1: Direct API Integration (16 hours)
**Build custom agents using Memori's REST APIs with official client libraries**

[Official Guide](https://memorilabs.ai/docs/guides/agents/direct-api)

#### Module 1: API Fundamentals (3 hours)
- Understanding Memori's Client API vs Indexing API
- Authentication and API tokens
- Rate limits and best practices
- Setting up your development environment

#### Module 2: Client Libraries (4 hours)
- **Python SDK**: `pip install memori-api-client`
- **TypeScript SDK**: `npm install @memoriwork/api-client`
- **Go SDK**: `go get github.com/memoriwork/api-client-go`
- **Java SDK**: Maven/Gradle configuration
- [Client Libraries Documentation](https://memorilabs.ai/docs/libraries/api-clients)

#### Module 3: Core APIs (5 hours)
- **Chat API**: Conversational AI with context
- **Search API**: Enterprise search capabilities
- **Agents API**: Manage pre-built agents
- **Documents API**: Access and manipulate documents
- **Entities API**: People, projects, and organizational data

#### Module 4: Building Your First Agent (4 hours)
- Project: Customer Support Agent
- Implementing search context retrieval
- Building conversational responses
- Managing conversation history
- Error handling and fallbacks

**Assessment**: Build a production-ready agent with custom business logic

---

### Path 2: LangChain Integration (12 hours)
**Build AI agents using Python and the LangChain framework**

[Official Guide](https://memorilabs.ai/docs/guides/agents/langchain)

#### Module 1: Setup (2 hours)
- Installing `langchain-memori`
```bash
pip install -U langchain-memori
```
- API token configuration
- Environment variables setup
- User-scoped tokens with required scopes (`chat`, `search`)

#### Module 2: LangChain Basics (3 hours)
- LangChain architecture overview
- Chains and agents
- Memory and conversation state
- Tool integration patterns

#### Module 3: Memori Tools in LangChain (4 hours)
- Using Memori as a retriever
- Implementing RAG patterns
- Multi-step reasoning
- Context management

#### Module 4: Production Deployment (3 hours)
- FastAPI integration
- Streaming responses
- Error handling
- Monitoring and logging

**Assessment**: Build an AI assistant that answers questions using company knowledge

---

### Path 3: Agent Toolkit (14 hours)
**Use pre-built tools across multiple agent frameworks**

[Official Guide](https://memorilabs.ai/docs/guides/agents/toolkit)

#### Module 1: Toolkit Overview (2 hours)
- When to use the Agent Toolkit
- Supported frameworks
- Installation for each framework
```bash
# OpenAI
pip install memori-agent-toolkit[openai]
# LangChain
pip install memori-agent-toolkit[langchain]
# CrewAI
pip install memori-agent-toolkit[crewai]
# Google ADK  
pip install memori-agent-toolkit[adk]
```

#### Module 2: Available Tools (4 hours)
- **memori_search**: Company knowledge base search
- **employee_search**: Find employees by attributes
- **calendar_search**: Meetings and events
- **code_search**: Source code repositories
- **gmail_search**: Gmail integration
- **outlook_search**: Outlook integration

#### Module 3: Framework Integration (4 hours)
- OpenAI Assistants integration
- LangChain tools
- CrewAI multi-agent systems
- Google ADK implementation

#### Module 4: Custom Tools (4 hours)
- Creating custom Memori tools
- Tool chaining and composition
- Cross-framework compatibility
- Testing and validation

**Assessment**: Build a multi-agent system using CrewAI with Memori tools

---

### Path 4: MCP Integration (10 hours)
**Connect AI tools to Memori's enterprise knowledge with zero setup**

[Official Guide](https://memorilabs.ai/docs/guides/mcp)

#### Module 1: MCP Fundamentals (2 hours)
- Model Context Protocol overview
- Remote vs Local MCP servers
- Supported host applications
- Authentication methods

#### Module 2: Remote MCP Server (3 hours)
- Opening the MCP Configurator
- OAuth setup
- Configuration for Claude Desktop
- Configuration for Cursor IDE
- Configuration for Windsurf
- [Administrator Guide](https://docs.memori.com/admin/mcp)

#### Module 3: MCP Tools (3 hours)
- **company_search**: Query enterprise content
- **chat**: Conversational AI
- **people_profile_search**: Employee directory
- **read_documents**: Retrieve documents by ID/URL

#### Module 4: Local MCP Server (2 hours)
- Self-hosted implementation
- `@memoriwork/local-mcp-server`
- Custom tool development
- Air-gapped environments

**Assessment**: Configure MCP for your organization and build a custom tool

---

### Path 5: Web SDK Integration (12 hours)
**Embed AI-powered search and chat into your intranet portal**

[Official Guide](https://memorilabs.ai/docs/libraries/web-sdk/overview)

#### Module 1: SDK Setup (2 hours)
- Installation and configuration
- Third-party cookie management
- Authentication setup
- Component overview

#### Module 2: Available Components (4 hours)
- **Memori Chat**: Embed full chat functionality
- **Autocomplete + Search Results**: Custom search page
- **Modal Search**: Overlay dialog implementation
- **Sidebar Search**: Contextual recommendations
- **Recommendations Component**: Embedded suggestions

#### Module 3: Customization (3 hours)
- UI customization
- Theme integration
- Component props and configuration
- Event handling

#### Module 4: Production Deployment (3 hours)
- Performance optimization
- Security considerations
- Analytics integration
- A/B testing

**Assessment**: Build a custom search page with Memori components

---

### Path 6: NVIDIA NIM Integration (8 hours)
**Build agents with NVIDIA NIM microservices and Memori**

[Official Guide](https://memorilabs.ai/docs/guides/agents/nvidia-example)

#### Module 1: Setup (2 hours)
- NVIDIA API key configuration
- LangChain NVIDIA endpoints
- Memori API setup
- Environment configuration
```python
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
```

#### Module 2: RAG Architecture (3 hours)
- Understanding retrieval-augmented generation
- Using Memori as a retriever
- Embedding models
- Context injection

#### Module 3: LangGraph Agent (2 hours)
- Agent state management
- Multi-step reasoning
- Tool integration
- Response generation

#### Module 4: Production Optimization (1 hour)
- Performance tuning
- Cost optimization
- Monitoring and observability
- Scaling considerations

**Assessment**: Build a customer support agent with NVIDIA NIM + Memori

---

## 🏆 Certifications

### Memori API Developer (Beginner)
- **Duration**: 10 hours
- **Prerequisites**: Basic Python or TypeScript knowledge
- **Topics**: Client APIs, Authentication, Basic agent building
- **Assessment**: Build a simple search integration
- **Cost**: Free

### Memori Integration Specialist (Intermediate)
- **Duration**: 20 hours
- **Prerequisites**: API Developer certification
- **Topics**: LangChain, Agent Toolkit, Web SDK, MCP
- **Assessment**: Build a production agent with one framework
- **Cost**: $299 (Free for Enterprise)

### Memori Solutions Architect (Advanced)
- **Duration**: 40 hours
- **Prerequisites**: Integration Specialist certification
- **Topics**: Multi-framework integration, custom tools, enterprise architecture
- **Assessment**: Design and implement a complete solution
- **Cost**: $999 (50% off for Enterprise)

---

## 📚 Course Catalog

### API Fundamentals
- **Duration**: 4 hours
- **Level**: Beginner
- **Format**: Self-paced
- **Topics**: Client API, Indexing API, Authentication, Rate limits

### Building Agents with Direct API
- **Duration**: 8 hours
- **Level**: Intermediate
- **Format**: Instructor-led + labs
- **Topics**: Chat API, Search API, Agents API, Custom logic

### LangChain for Enterprise AI
- **Duration**: 12 hours
- **Level**: Intermediate
- **Format**: Project-based
- **Topics**: LangChain integration, RAG patterns, Production deployment

### Multi-Agent Systems with CrewAI
- **Duration**: 10 hours
- **Level**: Advanced
- **Format**: Hands-on workshop
- **Topics**: Agent Toolkit, CrewAI, Orchestration, Testing

### MCP Configuration & Management
- **Duration**: 6 hours
- **Level**: Intermediate
- **Format**: Live workshop
- **Topics**: Remote MCP, OAuth, Host configuration, Custom tools

### Web SDK Deep Dive
- **Duration**: 8 hours
- **Level**: Intermediate
- **Format**: Code-along
- **Topics**: All components, Customization, Production deployment

### NVIDIA NIM + Memori Integration
- **Duration**: 6 hours
- **Level**: Advanced
- **Format**: Technical workshop
- **Topics**: NIM microservices, RAG, LangGraph, Optimization

---

## 🎯 Enterprise Training

### Custom Training Programs

**Half-Day Workshop** (4 hours)
- Focus on one specific topic
- Up to 30 participants
- Includes hands-on lab time
- $5,000

**Full-Day Intensive** (8 hours)
- Cover multiple topics
- Up to 25 participants
- Includes project work
- $10,000

**Multi-Day Bootcamp** (2-5 days)
- Comprehensive training program
- Certification preparation
- Custom curriculum
- $25,000 - $75,000

### Training Delivery Options

1. **Virtual Instructor-Led**
   - Live online sessions
   - Interactive labs
   - Q&A and support

2. **On-Site**
   - At your office
   - Hands-on collaboration
   - Custom environments

3. **Hybrid**
   - Mix of virtual and on-site
   - Flexible scheduling
   - Ongoing support

4. **Self-Paced Online**
   - Access anytime
   - Video lessons
   - Automated labs
   - Support forums

---

## 📅 Live Webinars (Monthly)

### Upcoming Schedule

**January 2026: Getting Started with Memori APIs**
- Introduction to Client and Indexing APIs
- Your first API integration
- Best practices and common patterns

**February 2026: Building AI Agents**
- Direct API vs LangChain vs Agent Toolkit
- When to use each approach
- Live coding demonstration

**March 2026: MCP Deep Dive**
- Remote vs Local MCP servers
- Configuring for different hosts
- Building custom MCP tools

**April 2026: Production Deployment**
- Security and authentication
- Scaling and performance
- Monitoring and observability

[Register for Webinars](https://academy.memorilabs.ai/webinars)

---

## 🔗 Quick Links

- **Developer Portal**: https://memorilabs.ai/docs
- **Learning Paths**: [learning-paths.html](../learning-paths.html)
- **API Reference**: https://memorilabs.ai/docs/api/client-api
- **GitHub Examples**: https://github.com/memoriwork
- **Community Slack**: https://memorilabs.ai/slack

---

## 💡 Getting Started

1. **Choose Your Path**: Select a learning path based on your needs
2. **Complete Prerequisites**: Ensure you have the required knowledge
3. **Enroll in Courses**: Self-paced or instructor-led options
4. **Complete Projects**: Hands-on assessments
5. **Get Certified**: Validate your expertise

[Start Learning Now →](../learning-paths.html)

---

## 📞 Contact

**Enterprise Training Inquiries**: academy@memori.com  
**Certification Support**: cert@memori.com  
**Technical Questions**: support@memori.com

**Office Hours**: Every Tuesday & Thursday, 2-4 PM PT  
[Schedule Office Hours](https://memorilabs.ai/office-hours)
