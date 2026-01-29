# SDK/DX Feedback Note: Add a "memori doctor" end-to-end self-check

## Summary
Shipping a single, guided CLI command that validates a user's Memori setup and runs a tiny write-and-recall loop would materially reduce time-to-first-success and support load. The command would be called `memori doctor` (or `python -m memori doctor`) and would surface precise, actionable errors for the most common failure points: missing API keys, unsupported LLM client registration, missing attribution, missing storage schema, and background augmentation not completing before exit.

## Proposal
Add a new CLI command:

```
memori doctor
```

Behavior:
1. Detect runtime environment and configuration.
   - Python version
   - Required env vars (e.g., OpenAI key if an OpenAI client is detected)
   - DB connection factory is callable
2. Validate Memori wiring.
   - Confirm `Memori(...).llm.register(...)` produces a supported client
   - Confirm `mem.attribution(...)` is set
3. Run an end-to-end smoke test.
   - Create a temporary SQLite DB (or use the provided connection)
   - Build the schema
   - Send a single test message
   - Wait for augmentation completion
   - Recall a known fact
4. Print a simple pass/fail summary with next-step links to docs and cookbook.

## Why this is the right lever
- It addresses the most frequent first-run failures (schema not built, attribution missing, provider mis-registered, short-lived scripts exiting before augmentation completes).
- It makes problem diagnosis explicit and fast, rather than asking developers to interpret stack traces.
- It would reduce repeated issues in GitHub and Discord and provide consistent, testable onboarding steps.

## Rationale with evidence
Recent issues show onboarding friction stemming from platform-specific setup and configuration mismatches (for example, Windows + Docker Compose onboarding) and provider-specific ingestion gaps (for example, Azure OpenAI multi-turn memory). A single, guided diagnostic would catch these classes of errors early and provide a clear path forward.

## Trade-offs considered
- Additional CLI surface area: another command to maintain and document.
- Extra runtime dependencies: must keep doctor lightweight and avoid pulling in optional providers unnecessarily.
- Risk of confusion if it modifies user data: mitigated by defaulting to a temporary SQLite DB and requiring explicit flags to touch production data.

## Rollout plan
1. Ship `memori doctor` behind a feature flag (or as a beta subcommand) for one release.
2. Instrument simple metrics: run count, pass rate, top error categories.
3. Promote in the README and docs Quickstart.
4. Iterate based on the top three failure modes.

## Success metrics
- Reduction in "first-run" issues and Discord support requests.
- Improved time-to-first-success (measured by quickstart completion in docs analytics).
- Increased adoption of the cookbook examples.
