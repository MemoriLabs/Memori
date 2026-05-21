#!/bin/bash
set -e

SESSION_ID="test-session-1"
USER_MSG_1="My name is Ryan. I'm a software engineer who loves TypeScript and hates standups."
ASSISTANT_MSG_1="Got it — you're Ryan, a software engineer. TypeScript fan, standup hater. Noted."
USER_MSG_2="I also prefer dark mode and my favorite food is tacos."
ASSISTANT_MSG_2="Dark mode and tacos. I'll remember that."

# load env for display
source .env 2>/dev/null || true

echo "======================================"
echo "  MEMORI OPENROUTER SKILL SMOKE TEST"
echo "======================================"
echo "MEMORI_API_KEY:    ${MEMORI_API_KEY:0:8}..."
echo "MEMORI_ENTITY_ID:  $MEMORI_ENTITY_ID"
echo "MEMORI_PROJECT_ID: ${MEMORI_PROJECT_ID:-<not set>}"
echo ""

echo "--- [capture] turn 1 ---"
echo "  sessionId:        $SESSION_ID"
echo "  userMessage:      $USER_MSG_1"
echo "  assistantMessage: $ASSISTANT_MSG_1"
echo ""
bun --env-file=.env index.ts capture \
  --sessionId "$SESSION_ID" \
  --userMessage "$USER_MSG_1" \
  --assistantMessage "$ASSISTANT_MSG_1"

echo ""
echo "--- [capture] turn 2 ---"
echo "  sessionId:        $SESSION_ID"
echo "  userMessage:      $USER_MSG_2"
echo "  assistantMessage: $ASSISTANT_MSG_2"
echo ""
bun --env-file=.env index.ts capture \
  --sessionId "$SESSION_ID" \
  --userMessage "$USER_MSG_2" \
  --assistantMessage "$ASSISTANT_MSG_2"

echo ""
echo "--- sleeping 3s for augmentation to process ---"
sleep 3

echo ""
echo "--- [recall] ---"
echo "  entity_id:  $MEMORI_ENTITY_ID"
echo "  project_id: ${MEMORI_PROJECT_ID:-<not set>}"
echo ""
bun --env-file=.env index.ts recall

echo ""
echo "--- [recall.summary] ---"
echo "  project_id: ${MEMORI_PROJECT_ID:-<not set>}"
echo "  date range: last 24h (default)"
echo ""
bun --env-file=.env index.ts recall.summary

echo ""
echo "--- [feedback] ---"
echo "  content: test feedback from openrouter skill smoke test"
echo ""
bun --env-file=.env index.ts feedback --content "test feedback from openrouter skill smoke test"

echo ""
echo "======================================"
echo "  done"
echo "======================================"
