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

## At the end of every assistant turn:
```bash
bun .claude/skills/memori/index.ts advanced-augmentation \
  --sessionId <sessionId> \
  --userMessage "<what the user said>" \
  --assistantMessage "<your response>"
```

This stores the turn and waits for the augmentation response. Do this every turn
so advanced augmentation runs consistently. Also use this immediately when the
user asks to save or remember something.

## When resuming after compaction:
```bash
bun .claude/skills/memori/index.ts compaction --projectId <projectId>
```

## When the user asks to send feedback about memory:
```bash
bun .claude/skills/memori/index.ts feedback --content "<feedback>"
```
