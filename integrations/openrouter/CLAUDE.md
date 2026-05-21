# Memori Memory Skill

You MUST use the Memori skill via Bash for any memory-related requests. Do not answer from context alone.

## You MUST run this at the start of every session:
```bash
bun .claude/skills/memori/index.ts recall.summary
```

## When the user asks what you remember, know about them, or anything from past sessions:
```bash
bun .claude/skills/memori/index.ts recall
```

## When the user asks to save or capture something:
```bash
bun .claude/skills/memori/index.ts capture \
  --sessionId <sessionId> \
  --userMessage "<what the user said>" \
  --assistantMessage "<your response>"
```

## When resuming after compaction:
```bash
bun .claude/skills/memori/index.ts compaction --projectId <projectId>
```

## When the user asks to send feedback about memory:
```bash
bun .claude/skills/memori/index.ts feedback --content "<feedback>"
```
