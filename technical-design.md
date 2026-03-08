# Technical Design Document: Agentic Rules Engine (Telco POC)

## 1. Objective

To build an agentic orchestration system using the Google Agent Development Kit (ADK) that replaces deterministic legacy rules engines (like Pega). The system guides users through complex Telco workflows (e.g., Phone Upgrade with Trade-in) utilizing a hybrid architecture: LLMs for conversational intent, slot-filling, and data extraction, combined with a strict Python Finite State Machine (FSM) for deterministic routing.

## 2. High-Level Architecture Pattern

The system uses a **Single-Agent ADK Architecture** strictly bound to a **Dynamic FSM Factory**. Logic (YAML Rules), Memory (JSON Ledger), and Execution (LLMs) are interconnected through high-performance `Pydantic` callbacks (`before_model_callback` and `after_tool_callback`).

### 2.1 The Single Agent Orchestrator

1.  **Tier 1: The Manager Agent (The Orchestrator)**
    * **Role:** Acts as both the conversational greeter and tool executor.
    * **Tool Belt:** The Manager has a single `google.adk.tools.McpToolset` mapped to an internal MCP mock server.
    * **Context:** Extremely small. The available tools are physically altered before every single LLM request using ADK hooks.

### 2.2 The Execution Guardrails (ADK Hooks)

To eliminate hallucinations and guarantee math compliance, the system injects strict callbacks into the `Manager` agent.

1.  **`before_model_callback`**: Before the ADK contacts Vertex AI, this python function intercepts the request. It reads `session.current_state` (e.g. `Auth`) and looks at the allowed FSM mapping. It dynamically scrubs all invalid tool schemas from the `llm_request.config.tools` payload. This guarantees the LLM physically cannot execute actions that contradict the Backend Rules Engine.
2.  **`after_tool_callback`**: When the LLM calls an MCP tool and the response comes back, this function intercepts the response. It takes the MCP `TextContent`, updates the memory Ledger, triggers the Python FSM to slide the pointer forward to the next state, and updates the state.

## 3. Core System Components

### 3.1 The Python FSM Factory (The Routing Engine)

* **Role:** Translates the YAML file into a strict mathematical state machine using a Python library (e.g., `transitions`).
* **Mechanics:** Uses a safe evaluation library (e.g., `simpleeval`) to check YAML string conditions against the JSON Ledger.
* **Handoff:** Called deterministically at the end of the `after_tool_callback` to advance the logic based on the tool data.

### 3.2 The State & Memory Layer (Anti-Bloat Architecture)

Memory is divided to ensure the LLM never suffers from context fatigue:

1.  **The Ledger (Single Source of Truth):** A hierarchical JSON object in Python memory. Stores factual data extracted by tools. *Crucial Hook:* The `after_tool_callback` intercepts MCP tool returns and updates this ledger automatically before advancing the FSM.
2.  **The Transcript (Sliding Window):** The raw chat history between the user and Manager. Strictly truncated to the last 3–5 turns.

### 3.3 The Lean YAML Schema (Business Logic)

The YAML is explicitly "DRY" (Don't Repeat Yourself). It does not define required API fields (the Manager infers those directly from the MCP tool schemas). It only defines the business flow, objectives, and transition conditions.

```yaml
use_case: phone_upgrade_flow
initial_state: Auth

# Evaluated by Python FSM BEFORE local states to handle "Change of Mind"
global_transitions:
  - intent: "change_line"
    next_state: LineToUpgrade
    # Wiping the line invalidates EVERYTHING downstream
    clear_memory: ["line_context", "trade_in_context", "new_device_context", "order_context"]
    
  - intent: "change_trade_in_device"
    next_state: DevicetradeInChecks
    # Wiping the trade-in invalidates the final math, but keeps the new device choice
    clear_memory: ["trade_in_context", "order_context"] 
    
  - intent: "change_new_device"
    next_state: NewUpgradeDeviceSelection
    # Wiping the new device invalidates the final math, but keeps the trade-in quote intact
    clear_memory: ["new_device_context", "order_context"]

states:
  Auth:
    objective: "Verify the user is an authorized account holder using the verify_auth tool."
    transitions:
      - condition: "account_context.is_authorized == True"
        next_state: AccountStandingCheck
      - condition: "account_context.is_authorized == False"
        next_state: EndUnauthorized

  AccountStandingCheck:
    objective: "Check if the account standing is good using the check_standing tool."
    transitions:
      - condition: "account_context.standing == 'GOOD'"
        next_state: LineToUpgrade
      - condition: "account_context.standing != 'GOOD'"
        next_state: EndBadStanding

  LineToUpgrade:
    objective: "Determine which phone line the user wants to upgrade."
    transitions:
      - condition: "line_context.selected_number != null"
        next_state: CheckLineUpgradeEligibility

  CheckLineUpgradeEligibility:
    objective: "Check if the selected line is eligible for an upgrade using the backend tool."
    transitions:
      - condition: "line_context.is_eligible == True"
        next_state: VerifyTradeIn
      - condition: "line_context.is_eligible == False"
        next_state: EndNotEligible

  VerifyTradeIn:
    objective: "Ask the user if they want to trade in their current device."
    transitions:
      - condition: "trade_in_context.wants_trade_in == True"
        next_state: DevicetradeInChecks
      - condition: "trade_in_context.wants_trade_in == False"
        next_state: NewUpgradeDeviceSelection

  DevicetradeInChecks:
    objective: "Ask the user about the physical condition of their trade-in device."
    transitions:
      - condition: "trade_in_context.final_condition != null"
        next_state: TradeInPricing
        
  TradeInPricing:
    objective: "Get the trade-in quote via the pricing tool based on the final condition."
    transitions:
      - condition: "trade_in_context.quote_value >= 0"
        next_state: NewUpgradeDeviceSelection

  NewUpgradeDeviceSelection:
    objective: "Determine which new device the user wants to purchase."
    transitions:
      - condition: "new_device_context.selection != null"
        next_state: NewUpgradeDevicePricing

  NewUpgradeDevicePricing:
    objective: "Get the pricing for the selected new device using the catalog tool."
    transitions:
      - condition: "new_device_context.price > 0"
        next_state: FinalPricing

  FinalPricing:
    objective: "Calculate final price (New Device Price - Trade In Quote) and present the summary to the user."
    transitions:
      - condition: "order_context.user_confirmed == True"
        next_state: ProcessOrder

  ProcessOrder:
    objective: "Submit the final transaction using the submit_order tool and confirm success with the user."
    transitions:
      - condition: "order_context.order_id != null"
        next_state: EndSuccess
      - condition: "order_context.error == True"
        next_state: EndOrderFailed

  EndSuccess:
    objective: "Thank the user and gracefully end the conversation."
    transitions: [] # Empty transitions denote a terminal state

  EndOrderFailed:
    objective: "Apologize to the user, explain the order failed, and offer to transfer to a human agent."
    transitions: []
  
  EndUnauthorized:
    objective: "Inform the user they are not authorized and end the conversation."
    transitions: []
    
  EndBadStanding:
    objective: "Inform the user their account standing prevents an upgrade and end the conversation."
    transitions: []
    
  EndNotEligible:
    objective: "Inform the user the selected line is not eligible for an upgrade at this time."
    transitions: []
```

## 4. Execution Flow & NLP Slot-Filling (Agent-as-a-Tool)

To ensure the Manager Agent does not hallucinate workflow transitions, its context is heavily modified during flight using callbacks.

1.  **Greeting & Intent:** Manager asks how it can help. User says, "Upgrade my phone."
2.  **Discovery & Extraction:** Manager enters the `Auth` state. The `before_model` hook intercepts the request and strips away all tools *except* `verify_auth`. 
3.  **Slot Filling:** Because the LLM only sees the `verify_auth` schema, it knows exactly what fields to ask for and will not try to submit an order. Manager asks the user for the required fields contextually.
4.  **Silent Execution & Ledger Update:** User provides the data. The LLM executes the tool. The MCP tool returns a JSON string via `TextContent`. The `after_tool` hook intercepts this response.
5.  **FSM Routing:** Inside the `after_tool` hook, Python updates the memory Ledger with the new data. It then immediately evaluates the FSM. The Python FSM evaluates the rules against the Ledger, moves the state pointer, and updates the session logic so the next `before_model` call will fetch the new tools.
6.  **The Fast-Forward Effect:** If the user provides upfront data (e.g., *"Upgrade line 3456, account 1234, pin 1234"*), the ADK handles this gracefully within the bounds of what the `before_model` hook allowed for that turn.

## 5. Flexibility & Navigation (Cascading Clears)

The architecture natively supports non-linear conversation ("Change of Mind") without corrupting math or bloating the LLM context.

1.  **Trigger:** User says, "Actually, let's trade in my iPad instead."
2.  **Intent Recognition:** The ADK Agent recognizes the intent and determines it falls outside the bounds of the current node's slot filling, utilizing global knowledge.
3.  **Deterministic Wipe:** A designated tool or fallback mechanism triggers a state evaluation that matches a `global_transition` in the YAML. It physically deletes `trade_in_context` and `order_context` from the JSON Ledger, ensuring old quotes are wiped, but preserving the `new_device_context`.
4.  **Handoff:** The FSM rewinds the state pointer to `DevicetradeInChecks`. The `before_model` hook immediately re-injects the trade-in tools. The Manager asks the user about the iPad with a clean context window.

## 6. Technical Debt & Phase 2 Features (Post-POC)

* **Auto-Generating `clear_memory` (Graph Traversal):** Business users should not manually map memory wipes in YAML. In V2, the Python backend will parse the YAML into a Directed Acyclic Graph (DAG) and automatically calculate the downstream data variables that must be wiped when a state regresses.
* **Human-in-the-Loop (HITL):** A mechanism to serialize the JSON Ledger and current FSM state to pass to a live human dashboard upon unresolvable workflow errors (e.g., hard failure on account standing).
* **Dynamic Agent Topologies:** Expanding the YAML to support `execution_mode: parallel` so the Manager can spin up ADK `ParallelAgents` for concurrent execution of multi-line audits.