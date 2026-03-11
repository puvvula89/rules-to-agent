# Technical Design Document: Agentic Rules Engine (Telco POC)

## 1. Objective

Build an agentic orchestration system using the Google Agent Development Kit (ADK) that replaces deterministic legacy rules engines (like Pega). The system guides users through complex Telco workflows (e.g., Phone Upgrade with Trade-in) using a hybrid architecture: an LLM for conversational intent, slot-filling, and data extraction, combined with a Python Finite State Machine (FSM) for deterministic routing.

**Key design principle:** Zero Python changes required when business rules evolve. The LLM semantically bridges tool outputs to structured data; the FSM routes deterministically from YAML conditions. Adding a state, changing a transition condition, or introducing a new change-of-mind intent requires only a YAML edit.

---

## 2. High-Level Architecture

The system is composed of three distinct layers that never bleed into each other:

```
┌─────────────────────────────────────────────────┐
│  ORCHESTRATOR (LLM — Gemini 2.5 Pro)            │
│  • Manages the user conversation                │
│  • Calls domain MCP tools to fulfill objectives │
│  • Translates tool responses into structured    │
│    JSON and passes them to fsm_advance          │
│  • Guided by: objective + extract_variables     │
└──────────────────────┬──────────────────────────┘
                       │ structured JSON via fsm_advance
┌──────────────────────▼──────────────────────────┐
│  BRIDGE (Python — fully domain-agnostic)        │
│  • before_model: reads FSM state → injects      │
│    objective + fsm_advance example into prompt  │
│  • after_tool: handles detect_intent only       │
│  • after_model: fallback JSON parser            │
│  • fsm_advance: explicit ADK tool the LLM calls │
│    after every domain tool; updates ledger and  │
│    fires FSM                                    │
└──────────────────────┬──────────────────────────┘
                       │ ledger dict
┌──────────────────────▼──────────────────────────┐
│  ROUTER (FSM — transitions + simpleeval)        │
│  • All conditions defined in YAML               │
│  • Per-transition closures evaluate at runtime  │
│  • Global intent transitions driven by          │
│    transition_type: global in YAML              │
│  • Zero knowledge of specific fields or states  │
└─────────────────────────────────────────────────┘
```

---

## 3. Core Components

### 3.1 The ADK Agent (`src/agents/manager.py`)

A single `google.adk.Agent` named `TelcoManager` backed by Gemini 2.5 Pro.

**Tools available to the LLM (always, no filtering):**
- All domain MCP tools via `McpToolset` (12 tools: `verify_auth`, `check_standing`, `set_line`, `check_eligibility`, `set_trade_in_preference`, `record_condition`, `pricing`, `select_device`, `confirm_order`, `decline_order`, `submit_order`, `detect_intent`)
- `fsm_advance` — an internal ADK `FunctionTool` registered directly on the agent

> **Why no tool filtering?** Filtering was considered and rejected. Injecting the current objective and `extract_variables` into the system prompt gives the LLM sufficient guidance to pick the right tool. Filtering adds fragile coupling between Python and YAML with no meaningful safety benefit — the FSM itself is the authoritative guardrail.

### 3.2 The Three ADK Callbacks

#### `before_model_callback`
Fires before every LLM request. Injects a two-part dynamic system instruction:

1. **`BRAND_INSTRUCTION`** (constant) — Verizon brand persona ("Alex"), tone guidelines, empathy rules. Identical across all states.
2. **Per-state dynamic block** — reads `fsm_state` from session, fetches `objective` and `extract_variables` from the FSM, builds a concrete `fsm_advance` call example, and appends the global intent list (loaded from YAML at startup).

The LLM always receives:
```
CURRENT STATE: <state_name>
CURRENT OBJECTIVE: <objective text>
WORKFLOW RULES:
1. Call the appropriate domain tool(s)...
2. After each domain tool call, call fsm_advance(data=<example>)
3. fsm_advance returns next_objective and data_still_needed...
CHANGE OF MIND — call detect_intent(intent) with one of:
  - intent_change_line: User wants to change the phone line they are upgrading
  - intent_change_new_device: User wants to select a different new device
  ...
```

No tool filtering occurs here. The only output is a mutated `system_instruction`.

#### `after_tool_callback`
Fires after every tool call. **Handles `detect_intent` only.** For all other tools, returns the response unchanged (the LLM is responsible for calling `fsm_advance` explicitly).

When `detect_intent` fires:
1. Parses `detected_intent` (a full trigger name, e.g., `intent_change_new_device`) from the MCP response
2. Calls `fsm.fire_intent(current_state, trigger_name, ledger)` which:
   - Wipes the relevant ledger keys (defined in YAML `clear_keys`)
   - Jumps the FSM to the target state

#### `after_model_callback`
**Fallback only.** Handles the case where the LLM produces a ` ```json``` ` block in its text response instead of calling `fsm_advance` (e.g., for terminal states where no tool call is needed).

If the response contains function calls, this callback skips entirely. Otherwise it:
1. Parses the first ` ```json``` ` block from the LLM text
2. Deep-merges it into the ledger
3. Runs a cascade loop (`evaluate` → `evaluate` → ...) until the FSM stabilises

### 3.3 The `fsm_advance` Tool (Primary FSM Advancement)

An internal ADK `FunctionTool` the LLM is instructed to call after every domain tool call. This is the primary mechanism by which the FSM advances.

```
LLM: call verify_auth(account_number="1234", pin="5678")
MCP: {"is_authorized": true}
LLM: call fsm_advance(data={"account_context": {"is_authorized": true}})
FSM: Auth → AccountStandingCheck
fsm_advance returns: {
    "workflow_advanced_to": "AccountStandingCheck",
    "next_objective": "Check if the account standing is good...",
    "data_still_needed": ["account_context.standing"]
}
LLM: call check_standing(account_number="1234")   ← immediately, no user input needed
...
```

Internally `fsm_advance`:
1. Normalises boolean strings (`"true"` → `True`, `"false"` → `False`) — the LLM occasionally emits booleans as strings
2. Deep-merges the data into the ledger (additive, never overwrites unrelated fields)
3. Calls `fsm.evaluate(current_state, ledger)` → new state
4. Returns the new state name, its objective, and its `extract_variables` list

### 3.4 The Python FSM (`src/orchestrator/fsm.py`)

A stateless FSM built on the `transitions` library (0.9.x) and driven entirely by YAML.

**`WorkflowFSM.__init__`** parses the YAML and:
- Derives ledger keys from `extract_variables` across all states (no hardcoding)
- For each transition with `condition_string`: generates a `simpleeval` closure and attaches it to `FlowController`
- For each transition with `transition_type: global` + `clear_keys`: generates a memory-wipe closure dynamically (no hardcoded method names) and wires it as the `before` callback

**Key API:**
| Method | Purpose |
|---|---|
| `evaluate(state, ledger)` | Fire `advance` trigger; return new state (or same if no condition matches) |
| `fire_intent(state, trigger_name, ledger)` | Fire a global intent trigger by full name; wipes ledger keys in-place |
| `get_objective(state)` | Return objective string for a state |
| `get_extract_variables(state)` | Return list of `context.field` paths the LLM must populate |
| `get_global_intents()` | Return `[{trigger, description}]` for all `transition_type: global` transitions |

**`simpleeval` constraint:** Condition strings cannot use `{}` dict literals. All conditions use subscript access: `context['account_context'].get('is_authorized') == True`.

### 3.5 The YAML Schema (`config/phone_upgrade.yaml`)

All business logic lives here. Python has zero knowledge of state names, field names, or condition logic.

```yaml
name: phone_upgrade_flow
initial: Auth

states:
  - name: Auth
    objective: "Verify the user is an authorized account holder using the verify_auth tool."
    extract_variables: ["account_context.is_authorized"]

  # ... 14 more states

transitions:
  # Global intents — identified by transition_type: global
  # clear_keys drives memory wipes (no hardcoded Python methods)
  # description is injected into the LLM system prompt dynamically
  - trigger: intent_change_new_device
    source: "*"
    dest: NewUpgradeDeviceSelection
    transition_type: global
    clear_keys: [new_device_context, order_context]
    description: "User wants to select a different new device"

  # Local transitions — all use the universal trigger 'advance'
  - trigger: advance
    source: Auth
    dest: AccountStandingCheck
    condition_string: "context['account_context'].get('is_authorized') == True"

  # ...
```

**What lives in YAML only:**
- State names and objectives
- `extract_variables` (tells LLM what data to collect per state)
- Transition `condition_string` (evaluated by `simpleeval`)
- Global intent trigger names, `clear_keys`, and `description`

**What does NOT live anywhere in Python or YAML:**
- Which tool to call per state — the LLM infers this from the objective + tool docstrings
- Ledger key names — derived at runtime from `extract_variables`

### 3.6 The Ledger (Session State)

A hierarchical JSON dict stored in ADK session state (`tool_context.state["ledger"]`). Grows additively across turns via `_deep_merge`. Never overwritten entirely — only targeted keys are cleared on change-of-mind.

Structure mirrors the `extract_variables` context groups:
```json
{
  "account_context":    {"is_authorized": true, "standing": "GOOD"},
  "line_context":       {"selected_number": "555-1234", "is_eligible": true},
  "trade_in_context":   {"wants_trade_in": true, "final_condition": "Good", "quote_value": 200},
  "new_device_context": {"selection": "iPhone 16", "price": 1000},
  "order_context":      {"user_confirmed": true, "order_id": "ORD-999888777", "error": false}
}
```

### 3.7 The MCP Server (`mock_mcp_server/server.py`)

A FastMCP ASGI server (`mcp.streamable_http_app()`) exposing 12 tools over Streamable HTTP. Returns plain JSON dicts. The LLM calls these tools naturally; the MCP envelope is unwrapped by `_parse_mcp_response` in the Python bridge when needed.

---

## 4. Execution Flow

### 4.1 Normal Turn (Slot-Filling)

The LLM is instructed to loop `tool → fsm_advance → tool → fsm_advance` within a single turn, continuing as long as it has the data needed to advance. It only pauses to ask the user when `data_still_needed` contains fields it cannot fill from the conversation.

```
User: "Hi, I'd like to upgrade my phone. Account 1234, PIN 5678."

before_model → injects: CURRENT STATE: Auth, CURRENT OBJECTIVE: verify_auth...
LLM → verify_auth(account_number="1234", pin="5678")
MCP → {"is_authorized": true}
LLM → fsm_advance(data={"account_context": {"is_authorized": true}})
FSM → Auth → AccountStandingCheck
      returns: next_objective="Check standing...", data_still_needed=["account_context.standing"]
LLM → check_standing(account_number="1234")     ← no user input needed
MCP → {"standing": "GOOD"}
LLM → fsm_advance(data={"account_context": {"standing": "GOOD"}})
FSM → AccountStandingCheck → LineToUpgrade
      returns: next_objective="Determine which phone line...", data_still_needed=["line_context.selected_number"]
LLM → "Great! Your account is in good standing. Which phone number would you like to upgrade?"
      ← pauses: needs line number from user
```

### 4.2 Change-of-Mind Flow

```
User: "Actually, I want a different phone."

LLM → detect_intent(intent="intent_change_new_device")
MCP → {"detected_intent": "intent_change_new_device"}
after_tool → fsm.fire_intent("FinalPricing", "intent_change_new_device", ledger)
           → wipes new_device_context and order_context from ledger
           → FSM jumps to NewUpgradeDeviceSelection
LLM → "Of course! What new device would you like instead?"
```

The LLM knows which intents are available because `before_model` injects them dynamically from `fsm.get_global_intents()` — loaded from YAML at startup.

---

## 5. FSM States (15 total)

**Happy path:**
`Auth → AccountStandingCheck → LineToUpgrade → CheckLineUpgradeEligibility → VerifyTradeIn → DeviceTradeInChecks → TradeInPricing → NewUpgradeDeviceSelection → NewUpgradeDevicePricing → FinalPricing → ProcessOrder → EndSuccess`

**Error terminals:** `EndUnauthorized`, `EndBadStanding`, `EndNotEligible`, `EndOrderFailed`

**Global intents (change-of-mind, fire from any state):**
- `intent_change_line` → `LineToUpgrade` (clears line + all downstream)
- `intent_change_trade_in_device` → `DeviceTradeInChecks` (clears trade-in + downstream)
- `intent_change_new_device` → `NewUpgradeDeviceSelection` (clears device + order)

---

## 6. Testing Pyramid

```
tests/
  test_fsm.py           — 21 unit tests: FSM transitions, intent resets, get_global_intents
                          No LLM, no MCP server. Tests the YAML→FSM routing in isolation.

  test_agent_flow.py    — 68 integration tests: callback pipeline simulation
                          No real LLM. Uses mock ADK objects to exercise before_model,
                          after_tool, after_model, and fsm_advance with the real FSM.
                          Includes MCP smoke tests (require server on :8080).

  test_e2e_live.py      — 4 E2E scenarios: real Gemini LLM + real MCP server
                          Happy path with trade-in, no trade-in, unauthorized account,
                          change-of-mind. Requires GCP credentials + MCP server.
```

---

## 7. How to Run

```bash
# Terminal 1 — MCP server
/usr/local/bin/poetry run uvicorn mock_mcp_server.server:app --port 8080

# Terminal 2 — ADK web UI
/usr/local/bin/poetry run adk web src/agents/
# Opens http://localhost:8000

# GCP credentials (Vertex AI)
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=tmeg-working-demos
export GOOGLE_CLOUD_LOCATION=us-central1
```

```bash
# Run tests (no credentials needed for unit + integration)
/usr/local/bin/poetry run pytest tests/test_fsm.py tests/test_agent_flow.py -v -k "not TestMCPServer"

# Full E2E (MCP server + GCP credentials required)
/usr/local/bin/poetry run pytest tests/test_e2e_live.py -v -s
# Logs written to: logs/flow-test.log
```

---

## 8. Technical Debt & Future Work

### V2: Sub-Agent Architecture (Context Isolation)

In the POC, all tool responses accumulate in the root agent's context window across turns. For production, a sub-agent pattern is recommended:

- **Root agent:** Manages all user conversation, knows FSM state, infers what to ask the user from objective + tool docstrings, writes user inputs to shared session, calls a `SubAgent` as an `AgentTool`
- **Sub-agent:** Receives the current state objective as its system prompt, reads user inputs from session, selects and calls the appropriate MCP tool, returns a ` ```json``` ` block
- **Root agent:** Parses the sub-agent reply, feeds JSON to FSM via `fsm_advance`, advances state

This ensures tool responses never accumulate in the root agent context. Root agent context only grows with user conversation + small JSON blocks per state. Sub-agent context is always fresh (one objective + one tool call per invocation).

ADK `AgentTool` supports this natively. Session state is shared between parent and child agents.

### Auto-Generating `clear_keys` (Graph Traversal)

Business users should not manually map memory wipes in YAML. In V2, the Python backend will parse the YAML into a Directed Acyclic Graph (DAG) and automatically calculate the downstream data variables that must be cleared when a state regresses. The `clear_keys` field would be removed from YAML entirely.

### Human-in-the-Loop (HITL)

Serialize the JSON ledger and current FSM state to a live human dashboard on unresolvable workflow errors (e.g., hard failure on account standing, repeated auth failures).

### Dynamic Agent Topologies

Expand the YAML to support `execution_mode: parallel` so the manager can spin up ADK `ParallelAgents` for concurrent execution of multi-line audits.
