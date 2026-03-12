# Telco ADK + FSM POC — Claude Context

## Project Goal
Replace legacy telco rules engines with a Google ADK + YAML-driven Python FSM for a phone-upgrade workflow. Single-agent ADK architecture with deterministic FSM guardrails preventing LLM hallucination.

**Key Demo**: Zero Python changes required when business rules evolve or tool response field names change. The LLM semantically bridges tool outputs to structured JSON; the FSM routes deterministically from YAML conditions.

## Architecture (Three Layers)

```
┌─────────────────────────────────────────────────┐
│  ORCHESTRATOR (LLM — Gemini 2.5 Pro)            │
│  • Talks to user, calls any MCP tool it needs   │
│  • Translates tool responses into clean JSON    │
│  • Explicitly calls fsm_advance after each tool │
│  • Guided by: objective + extract_variables     │
└──────────────────────┬──────────────────────────┘
                       │ structured JSON via fsm_advance
┌──────────────────────▼──────────────────────────┐
│  BRIDGE (Python — fully domain-agnostic)        │
│  • before_model: injects dynamic system prompt  │
│  • fsm_advance tool: updates ledger, fires FSM  │
│  • after_tool: handles detect_intent only       │
│  • after_model: fallback JSON block parser      │
└──────────────────────┬──────────────────────────┘
                       │ ledger dict
┌──────────────────────▼──────────────────────────┐
│  ROUTER (FSM via `transitions` library)         │
│  • All conditions in YAML as condition_string   │
│  • Per-transition closures evaluate at runtime  │
│  • Global intents identified by transition_type │
│  • Zero knowledge of specific states or fields  │
└─────────────────────────────────────────────────┘
```

## Key Files

| File | Role |
|------|------|
| `src/agents/agent.py` | ADK agent + 3 callbacks + fsm_advance tool; fully domain-agnostic |
| `src/agents/orchestrator/fsm.py` | WorkflowFSM + FlowController; uses `transitions` library |
| `config/phone_upgrade.yaml` | ALL business logic: states, transitions, conditions, objectives |
| `mock_mcp_server/server.py` | FastMCP ASGI mock API server (12 tools) |
| `deploy/app.py` | AdkApp wrapper for Agent Engine deployment |
| `tests/test_fsm.py` | 21 FSM unit tests (no agent/LLM needed) |
| `tests/test_agent_flow.py` | 68 callback integration tests (no LLM needed) |
| `tests/test_e2e_live.py` | 4 E2E scenarios with real Gemini LLM + MCP server |

## What Lives Where

| Concern | Location |
|---|---|
| States, objectives, extract_variables | YAML only |
| Transition conditions (condition_string) | YAML only |
| Global intents, clear_keys, descriptions | YAML only (`transition_type: global`) |
| Tool selection per state | **Nowhere** — LLM infers from objective + tool descriptions |
| Ledger key names | Derived from YAML `extract_variables` at FSM init |
| Memory-wipe logic | Generated dynamically from YAML `clear_keys` — no hardcoded Python methods |
| Global intent names in LLM prompt | Loaded from YAML at startup via `get_global_intents()` |
| FSM routing logic | `transitions` library + `simpleeval`, driven by YAML |

## Critical Implementation Notes

### Three Callbacks + fsm_advance Tool

- `before_model` — injects dynamic system instruction: `BRAND_INSTRUCTION` (constant Verizon/Alex persona) + per-state objective + `fsm_advance` call example + global intents list (loaded from YAML). No tool filtering.
- `after_tool` — **only handles `detect_intent`**; calls `fsm.fire_intent()` with the full trigger name. All other tools: passes response through unchanged.
- `after_model` — **fallback only**. Parses ` ```json``` ` block from LLM text if no function calls were made; deep-merges into ledger; cascade-advances FSM.
- `fsm_advance` (ADK FunctionTool) — **primary FSM advancement mechanism**. LLM calls this explicitly after every domain tool call. Normalises boolean strings, deep-merges data into ledger, fires `fsm.evaluate()`, returns `{workflow_advanced_to, next_objective, data_still_needed}`.

### Slot-Filling Loop
The LLM is instructed to loop `tool → fsm_advance → tool → fsm_advance` within a single turn, advancing as long as it has the data. It pauses only when `data_still_needed` contains fields the user hasn't provided yet.

### FSM API
- `fsm.evaluate(current_state, ledger)` → new_state
- `fsm.fire_intent(current_state, trigger_name, ledger)` → new_state (takes **full trigger name**, e.g. `intent_change_new_device`; clears ledger keys in-place)
- `fsm.get_objective(state_name)` → str
- `fsm.get_extract_variables(state_name)` → list[str]
- `fsm.get_global_intents()` → list of `{trigger, description}` for all `transition_type: global` transitions

### Session State
Stored in `tool_context.state` / `callback_context.state`:
- `state["fsm_state"]` — current FSM state name (str)
- `state["ledger"]` — starts as `{}`, grows via deep-merge as LLM calls `fsm_advance`

### YAML Global Transition Schema
Global (change-of-mind) transitions use `transition_type: global`. Memory-wipe methods are generated dynamically from `clear_keys` — no hardcoded Python methods.
```yaml
- trigger: intent_change_new_device
  source: "*"
  dest: NewUpgradeDeviceSelection
  transition_type: global
  clear_keys: [new_device_context, order_context]
  description: "User wants to select a different new device"
```

### YAML Condition String Constraints (simpleeval)
- No `{}` dict literals → use subscript access: `context['account_context'].get('is_authorized')`
- No tuple literals → use `and`/`or` chains
- All ledger keys are normalised into `context` at evaluation time so subscript access is always safe

### transitions Library (0.9.x) Constraints
- `State.__init__()` rejects custom kwargs → pass only state name strings to Machine; keep metadata in `_state_meta` dict
- `Transition.__init__()` rejects custom kwargs → strip `condition_string`, `transition_type`, `clear_keys`, `description` before passing to Machine; attach closures via `setattr`
- `send_event=True` — EventData passed to all callbacks
- `ignore_invalid_triggers=True` — silently stay in terminal states

### Boolean Normalisation
LLM occasionally emits `"true"`/`"false"` as strings. `_normalize_booleans()` is called in both `fsm_advance` and `after_model` before FSM evaluation. FSM conditions use `== True` / `== False` so string values cause silent failures without this.

### MCP Server
`app = mcp.streamable_http_app()` — do NOT wrap in `Starlette(Mount(...))`.

### Poetry Path
`/usr/local/bin/poetry` (not in PATH by default on this machine).

## How to Run Locally

```bash
# Terminal 1: MCP server
/usr/local/bin/poetry run uvicorn mock_mcp_server.server:app --port 8080

# Terminal 2: ADK web UI
/usr/local/bin/poetry run adk web src/
# Opens http://localhost:8000
```

```bash
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=tmeg-working-demos
export GOOGLE_CLOUD_LOCATION=us-central1
```

## Running Tests

```bash
# FSM unit tests (no credentials needed)
/usr/local/bin/poetry run pytest tests/test_fsm.py -v

# Integration tests (no LLM; skip MCP smoke tests if server not running)
/usr/local/bin/poetry run pytest tests/test_agent_flow.py -v -k "not TestMCPServer"

# Integration tests with MCP smoke tests (MCP server must be on :8080)
/usr/local/bin/poetry run pytest tests/test_agent_flow.py -v

# E2E live tests — MCP server + GCP credentials required
/usr/local/bin/poetry run pytest tests/test_e2e_live.py -v -s
# Logs written to: logs/flow-test.log (overwritten on each run)
```

## GCP Project
- **Project**: `tmeg-working-demos` | **Location**: `us-central1` | **Model**: `gemini-2.5-pro`
- **Python**: 3.14+ | **Package manager**: Poetry (`/usr/local/bin/poetry`)

## FSM States (15 total)
Happy path: `Auth → AccountStandingCheck → LineToUpgrade → CheckLineUpgradeEligibility → VerifyTradeIn → DeviceTradeInChecks → TradeInPricing → NewUpgradeDeviceSelection → NewUpgradeDevicePricing → FinalPricing → ProcessOrder → EndSuccess`

Error terminals: `EndUnauthorized`, `EndBadStanding`, `EndNotEligible`, `EndOrderFailed`

Global intents (change-of-mind, full trigger names): `intent_change_line`, `intent_change_trade_in_device`, `intent_change_new_device`

## Current Status
All phases complete. 89 tests pass (21 FSM unit + 68 agent flow integration). E2E live tests pass (3/4 verified; happy path with trade-in assertion fixed).

**Completed work:**
- Phase 1–4: FSM stateless, MCP server, ADK callbacks, dynamic FSM-LLM architecture
- `fsm_advance` explicit tool for slot-filling loop
- `BRAND_INSTRUCTION` constant — Verizon/Alex persona injected every turn
- Boolean normalisation (`"true"` string → `True` bool)
- `transition_type: global` in YAML to identify change-of-mind transitions
- `clear_keys` in YAML drives memory-wipe closures — no hardcoded Python methods
- `get_global_intents()` loads intent names + descriptions from YAML at startup
- `fire_intent` uses full trigger name directly (no `intent_` prefix prepend)
- `_GLOBAL_INTENTS_TEXT` injected into system prompt dynamically

**Next steps:**
- Deploy to Agent Engine (Phase 3 — `deploy/app.py` ready)
- Dynamism demo: change `verify_auth` to return `{"authorization_status": "GRANTED"}` → zero Python/YAML changes, LLM maps semantically
