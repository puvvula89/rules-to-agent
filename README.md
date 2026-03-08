# Telco ADK + FSM POC

A phone-upgrade workflow agent built with Google ADK, a YAML-driven Python FSM, and MCP tools. The LLM handles conversation; the FSM enforces deterministic routing.

---

## Architecture

```
User → ADK Agent (TelcoManager / Gemini 2.5 Pro)
              ↕ before_model: filter tools by FSM state
              ↕ after_tool:   update ledger, advance FSM
       MCP Tools (mock_mcp_server) ← 12 telco tools
       FSM (config/phone_upgrade.yaml) ← YAML state machine
       ADK Session State ← ledger + fsm_state persisted automatically
```

**FSM States:** `Auth → AccountStandingCheck → LineToUpgrade → CheckLineUpgradeEligibility → VerifyTradeIn → DeviceTradeInChecks → TradeInPricing → NewUpgradeDeviceSelection → NewUpgradeDevicePricing → FinalPricing → ProcessOrder → EndSuccess`

Error terminals: `EndUnauthorized`, `EndBadStanding`, `EndNotEligible`, `EndOrderFailed`

---

## Setup

### Requirements
- Python 3.14+
- Poetry ([install](https://install.python-poetry.org))
- Google API key **or** gcloud with Vertex AI access

### Install
```bash
git clone <repo-url>
cd rules-to-agent
poetry install
```

### Credentials (pick one)

**Option A — Google AI Studio (simplest)**
```bash
export GOOGLE_API_KEY=your_key_here
# Get a key at: https://aistudio.google.com/app/apikey
```

**Option B — Vertex AI**
```bash
gcloud auth application-default login
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=tmeg-working-demos
export GOOGLE_CLOUD_LOCATION=us-central1
```

---

## Running Locally

### Terminal 1 — Start the MCP server
```bash
poetry run uvicorn mock_mcp_server.server:app --port 8080
# Expected: "StreamableHTTP session manager started"
```

### Terminal 2 — Start the ADK web UI
```bash
poetry run adk web src/agents/
# Opens http://localhost:8000 — chat UI + trace inspector
```

Walk through the flow in the browser chat:
- Account `1234`, any PIN → happy path → `EndSuccess` with `order_id: ORD-999888777`
- Account `9999` → `EndUnauthorized`
- Say "change my device" at FinalPricing → FSM rewinds to `NewUpgradeDeviceSelection`

---

## Testing

### Unit tests — FSM logic only (no LLM, no server)
```bash
poetry run pytest tests/test_fsm.py -v
```

### Integration tests — callbacks + MCP server (no LLM)
Requires the MCP server running on port 8080.
```bash
poetry run pytest tests/test_agent_flow.py -v
```

### E2E live tests — real LLM + real MCP server + real FSM
Requires credentials (Step above) and MCP server running.

```bash
# All 4 scenarios
poetry run pytest tests/test_e2e_live.py -v -s

# Individual scenarios
poetry run pytest tests/test_e2e_live.py::test_happy_path_with_trade_in -v -s
poetry run pytest tests/test_e2e_live.py::test_no_trade_in_path -v -s
poetry run pytest tests/test_e2e_live.py::test_unauthorized_path -v -s
poetry run pytest tests/test_e2e_live.py::test_change_of_mind -v -s

# Interactive menu (standalone)
poetry run python tests/test_e2e_live.py
```

**Expected output per turn:**
```
[USER]  Hi, I'd like to upgrade my phone. Account 1234, PIN 5678.
  [TOOLS] verify_auth → check_standing
  [FSM]   LineToUpgrade
[AGENT] Your account is verified! Which line would you like to upgrade?
```

**Expected final output:**
```
FINAL FSM STATE: EndSuccess
✓ Happy path with trade-in PASSED
```

---

## Project Structure

```
config/
  phone_upgrade.yaml        # FSM states, transitions, objectives
mock_mcp_server/
  server.py                 # FastMCP ASGI app — 12 telco tools
src/
  agents/manager.py         # ADK root_agent + before_model/after_tool callbacks
  orchestrator/fsm.py       # Stateless FSM engine
deploy/
  app.py                    # Phase 3: AdkApp wrapper for Agent Engine
tests/
  test_fsm.py               # 20 FSM unit tests
  test_agent_flow.py        # 48 MCP + callback integration tests
  test_e2e_live.py          # 4 live E2E scenarios (real LLM)
```

---

## GCP Deployment (Phase 3)

```bash
# Deploy to Agent Engine
poetry run python deploy/app.py
```

Project: `tmeg-working-demos` · Location: `us-central1` · Model: `gemini-2.5-pro`
