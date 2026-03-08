# Agentic Rules Engine (Telco POC) Implementation Plan

## Goal Description
Build an agentic orchestration POC that replaces deterministic legacy rules engines. The system natively uses Google ADK in a **Single-Agent Architecture**, leveraging Pydantic hooks (`before_model` and `after_tool`) to enforce strict compliance with a Python Finite State Machine (FSM). This eliminates routing hallucinations by completely controlling the tools exposed to the ADK Agent at any given moment based on the FSM State.

## Proposed Changes

We will refine the codebase to ensure robust handling of MCP TextContent payloads.

---

### Phase 1: Robust Payload Parsing in Pydantic Hooks

Currently, the `after_tool` hook receives a raw string payload from the Model Context Protocol (MCP) tool response, but MCP v1.0 standardizes returning a list of `TextContent` objects. The manager agent must cleanly unwrap this to update the JSON ledger.

#### [MODIFY] src/agents/manager.py
- Update the `after_tool` hook to safely parse `tool_result` whether it arrives as a string or structurally as a list of MCP `TextContent` objects.
- Ensure the `json.loads` reliably extracts the dictionary to update `session.ledger.[context_key]`.

---

### Phase 2: mock_mcp_server Refinements

The baseline mock tools are implemented, but they need to ensure valid state transitions can occur uninterrupted.

#### [MODIFY] mock_mcp_server/server.py
- Audit all mock tools (`check_eligibility`, `pricing`, `submit_order`) to verify they definitively return `list[TextContent]` objects.
- Ensure no Python execution paths lack a `return` statement, which would crash the stdio server connection.

---

## Verification Plan

### Manual Verification
1. Open a terminal and run `poetry run python src/main.py`.
2. Follow the standard "Upgrade Phone" flow:
   - Provide "I want to upgrade my phone"
   - Provide account number and PIN.
3. Observe the internal logs to verify:
   - The MCP `verify_auth` tool returns successfully.
   - The FSM pointer advances from `Auth` to `AccountStandingCheck`.
   - The Ledger prints correct variables.
