# Memori Memory Types Guide

This document explains the seven memory types that Memori extracts and indexes, and provides examples of how to work with them.

## Quick Reference

| Memory Type | Purpose | Example | Scope |
|-------------|---------|---------|-------|
| **Facts** | Absolute truths | "User's name is John" | Entity |
| **Events** | Time-based actions | "User completed signup" | Session |
| **Relationships** | Connections | "John manages Sarah" | Entity |
| **Skills** | Capabilities | "Knows Python" | Entity |
| **Preferences** | User choices | "Prefers email contact" | Session |
| **Rules** | Constraints | "Budget limit $5000" | Entity |
| **People** | Individual info | "John from Sales" | Entity |

---

## Memory Type Details

### 1. Facts

**What it captures**: Definitive, unchanging information

**Characteristics**:
- Single source of truth
- Rarely change
- High recall relevance
- Entity-scoped (apply across all sessions)

**Examples**:
```
Input: "My name is John and I work at Acme Corp."

Facts extracted:
- user has name "John"
- user works at "Acme Corp"
- user position "employee"

Usage:
facts = mem.recall("What is the user's name?")
# → Returns: "John"

facts = mem.recall("Where does the user work?")
# → Returns: "Acme Corp"
```

**Best for**:
- Personal information (name, email, phone)
- Organizational data (department, role, company)
- System configuration
- Static preferences that don't change

---

### 2. Events

**What it captures**: Actions or occurrences tied to specific times

**Characteristics**:
- Point-in-time records
- Historical audit trail
- Include temporal context
- Session-scoped (time-limited)

**Examples**:
```
Input: "I just completed the onboarding process and received my API key."

Events extracted:
- "user completed onboarding" (timestamp: 2024-06-27T10:30:00Z)
- "user received API key" (timestamp: 2024-06-27T10:30:01Z)

Usage:
events = mem.recall("What did the user do today?")
# → Returns recent events with timestamps

events = mem.recall("When did onboarding complete?", limit=1)
# → Returns event with timestamp
```

**Best for**:
- Action tracking (completed task, started project)
- System events (error occurred, deployment failed)
- User interactions (logged in, clicked button)
- Change tracking (updated profile, changed settings)

---

### 3. Relationships

**What it captures**: Connections between entities

**Characteristics**:
- Bi-directional connections
- Build knowledge graphs
- Enable inference and reasoning
- Entity-scoped (persistent)

**Examples**:
```
Input: "I manage a team of 5 engineers. Alice leads the frontend team. 
Bob is our infrastructure specialist."

Relationships extracted:
- user → manages → team (5 engineers)
- Alice → works on → frontend
- Bob → specializes in → infrastructure

Usage:
relationships = mem.recall("Who works for this user?")
# → Returns: Alice, Bob, and other team members

relationships = mem.recall("What is Alice's role?")
# → Returns: "frontend team lead"
```

**Best for**:
- Organizational hierarchies (manager-employee)
- Project teams
- Customer relationships
- Software dependencies
- Supply chain connections

---

### 4. Skills

**What it captures**: Capabilities, expertise, or proficiency areas

**Characteristics**:
- Proficiency levels (beginner, intermediate, expert)
- Track experience duration
- Enable capability matching
- Entity-scoped

**Examples**:
```
Input: "I have 10 years of Python experience and am learning Go. 
I'm fluent in Spanish and intermediate with Mandarin."

Skills extracted:
- Python: 10 years, expert level
- Go: beginner level (just learning)
- Spanish: fluent level
- Mandarin: intermediate level

Usage:
skills = mem.recall("What programming languages can this user work with?")
# → Returns: Python (expert), Go (beginner)

skills = mem.recall("What languages does the user speak?")
# → Returns: Spanish (fluent), Mandarin (intermediate)
```

**Best for**:
- Employee skill inventories
- Freelancer profiles
- Agent capabilities
- Permission/authorization levels
- Resource allocation

---

### 5. Preferences

**What it captures**: User or process preferences for this interaction

**Characteristics**:
- Session-specific (usually)
- Personalization data
- Can influence behavior
- Short-lived or explicitly refreshed

**Examples**:
```
Input: "I prefer detailed technical explanations, and I want all responses 
in bullet points. Please avoid jargon."

Preferences extracted:
- Communication style: detailed and technical
- Format preference: bullet points
- Jargon preference: avoid

Usage:
prefs = mem.recall("How should we communicate with this user?")
# → Returns communication preferences

# Next request can use these prefs for better responses
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": 
            "Communicate with bullet points, detailed and technical, avoid jargon"
        },
        {"role": "user", "content": "Explain how caching works"}
    ]
)
```

**Best for**:
- UI/UX preferences (dark mode, language, timezone)
- Communication style preferences
- Content format preferences (markdown, JSON, etc)
- Privacy preferences
- Notification settings

---

### 6. Rules

**What it captures**: Constraints, guidelines, and business logic

**Characteristics**:
- Business constraints
- Safety/compliance rules
- Operational guidelines
- Entity or process scoped
- Often mandatory

**Examples**:
```
Input: "We have a strict policy: never approve expenses above $10,000 
without VP sign-off. Budget allocation for Q3 is $50,000. 
All travel requires manager approval."

Rules extracted:
- expense > $10,000 requires VP approval (mandatory)
- budget Q3 = $50,000 (constraint)
- travel requires manager approval (process rule)

Usage:
rules = mem.recall("What are the approval requirements?")
# → Returns: VP approval for >$10k, manager approval for travel

# Can be used to validate actions
if expense_amount > 10000:
    rules = mem.recall("What approval is needed for large expenses?")
    if "VP" in rules:
        notify_vp(expense)  # Trigger rule-based action
```

**Best for**:
- Compliance rules
- Business policies
- Access control rules
- Budget constraints
- Operational guidelines
- Safety constraints

---

### 7. People

**What it captures**: Information about individuals mentioned in conversations

**Characteristics**:
- Individual profiles
- Build organization directory
- Track roles and departments
- Entity-scoped
- Relationships to other people

**Examples**:
```
Input: "John from Sales reported a critical bug. Sarah from DevOps is 
investigating. We're waiting for Bob's code review from the backend team."

People extracted:
- John: department Sales, action "reported bug"
- Sarah: department DevOps, action "investigating"
- Bob: department Backend, role "code reviewer"

Usage:
people = mem.recall("Who reported the bug?")
# → Returns: John from Sales

people = mem.recall("Who is available to help?")
# → Returns: Sarah (DevOps), Bob (Backend)

# Build organization graph
people = mem.recall("List all team members")
# → Returns all people mentioned
```

**Best for**:
- Organization directories
- Contact lists
- Team hierarchies
- Responsibility tracking
- Escalation paths

---

## Memory Type Interactions

Memory types often work together to build rich context:

### Example: Support Ticket System

```
Conversation:
"Hi, I'm Alice from Acme Corp. I have a critical API issue. 
My team uses Python heavily. We need this fixed by tomorrow."

Multiple memory types extracted:

1. FACT
   └─ company = "Acme Corp"
   └─ primary_language = "Python"

2. PEOPLE
   └─ Alice: role "team lead", company "Acme Corp"

3. SKILL
   └─ team_skill_Python = "heavy usage"

4. EVENT
   └─ "critical API issue reported" (timestamp: now)

5. RULE
   └─ priority = "critical"
   └─ deadline = "tomorrow"

6. PREFERENCE
   └─ communication_urgency = "high"

Later recall:
mem.recall("What issues do we have?")
# Returns: Critical API issue (EVENT)

mem.recall("Who reported this?")
# Returns: Alice from Acme Corp (PEOPLE)

mem.recall("What's the deadline?")
# Returns: Tomorrow (RULE)

mem.recall("Can we use Python in the fix?")
# Returns: Yes, team uses Python heavily (SKILL)
```

---

## Working with Memory Types in Code

### Recall by Type

```python
from memori import Memori

mem = Memori()
mem.attribution(entity_id="alice", process_id="support_agent")

# Query-based recall returns relevant type
facts = mem.recall("What is Alice's email?")
# → Returns FACT type

events = mem.recall("What happened yesterday?")
# → Returns EVENT type

skills = mem.recall("What can Alice do?")
# → Returns SKILL type

rules = mem.recall("What are the requirements?")
# → Returns RULE type
```

### Advanced Filtering

```python
# Get facts only (BYODB mode)
facts = mem.recall(
    query="Tell me about the user",
    limit=10
)

# Facts are returned with metadata
for fact in facts:
    print(f"Fact: {fact.content}")
    print(f"Confidence: {fact.rank_score}")
    print(f"Memory ID: {fact.id}")

# Cloud mode includes more metadata
result = mem.recall("What are the constraints?")

for fact in result['facts']:
    print(f"Rule: {fact.get('content')}")
    print(f"Type: {fact.get('memory_type')}")
    print(f"Score: {fact.get('rank_score')}")
```

### Using Memory Types for Agent Behavior

```python
# Example: Smart response generation using memory types

def respond_to_user(user_query):
    # Recall relevant memories
    memories = mem.recall(user_query, limit=20)
    
    # Organize by type
    facts = [m for m in memories if m.get('type') == 'fact']
    rules = [m for m in memories if m.get('type') == 'rule']
    skills = [m for m in memories if m.get('type') == 'skill']
    prefs = [m for m in memories if m.get('type') == 'preference']
    
    # Build context
    context = ""
    if facts:
        context += f"Known facts: {facts}\n"
    if rules:
        context += f"Apply these rules: {rules}\n"
    if skills:
        context += f"Available capabilities: {skills}\n"
    if prefs:
        context += f"User preferences: {prefs}\n"
    
    # Generate response with full context
    response = llm.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"{context}"},
            {"role": "user", "content": user_query}
        ]
    )
    
    return response.choices[0].message.content
```

---

## Best Practices

### ✅ DO

- **Use specific queries** to get relevant memory types
- **Update facts** when information changes
- **Create new sessions** when context fundamentally changes
- **Set entity_id and process_id** for proper scoping
- **Use preferences** to personalize interactions
- **Document rules** for compliance and safety

### ❌ DON'T

- **Mix memory types** in a single recall query
- **Rely on stale events** without timestamp checking
- **Forget to set attribution** - memories won't be stored
- **Store sensitive data** as preferences (use encrypted field)
- **Assume global scope** - remember entity/process/session levels
- **Ignore low confidence scores** - filter out uncertain recalls

---

## Troubleshooting Memory Types

### No Results Returned

**Problem**: `mem.recall()` returns empty list

**Causes**:
- Entity not created yet (no previous interactions)
- Query too specific (no matching memories)
- Relevance threshold too high

**Solution**:
```python
# Check if entity exists
if not mem.config.entity_id:
    mem.attribution(entity_id="new_user")

# Try broader query
results = mem.recall("Tell me about this user")

# Lower threshold temporarily
original_threshold = mem.config.recall_relevance_threshold
mem.config.recall_relevance_threshold = 0.3
results = mem.recall(query)
mem.config.recall_relevance_threshold = original_threshold
```

### Wrong Memory Type Returned

**Problem**: Query for fact returns event or vice versa

**Causes**:
- Ambiguous query phrasing
- Similar semantic meaning across types
- Hybrid search returning multiple types

**Solution**:
```python
# Be more specific in query
results = mem.recall("What is the user's exact name?")
# More likely to return FACT

results = mem.recall("When did the user sign up?")
# More likely to return EVENT
```

### Low Confidence Scores

**Problem**: All recalled memories have low scores (< 0.7)

**Causes**:
- Few similar memories in database
- Query poorly phrased
- Memory types don't match query semantics

**Solution**:
```python
# Rephrase query with domain language
results = mem.recall("SLA requirements?")  # Better for RULE type

# Check if memories exist first
all_memories = mem.recall("everything about this user", limit=100)
print(f"Found {len(all_memories)} memories")

# Adjust re-ranking weights
import os
os.environ['MEMORI_RECALL_LEX_WEIGHT'] = '0.25'  # Increase keyword matching
```

---

## See Also

- [Architecture Overview](./ARCHITECTURE.md) - System design
- [Attribution & Sessions](./ARCHITECTURE.md#attribution-system) - Scoping
- [Recall Pipeline](./ARCHITECTURE.md#recall-pipeline) - How retrieval works
