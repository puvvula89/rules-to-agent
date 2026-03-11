# High-Level Design: Agentic Rules Engine

## 1. The Problem

Enterprise workflows in telco (and similar industries) are governed by complex business rules: eligibility checks, conditional routing, data dependencies between steps, and strict sequencing requirements. Today these live in legacy rules engines like Pega — hard to change, expensive to maintain, and disconnected from modern conversational interfaces.

LLMs offer a compelling alternative for the conversational layer, but they introduce a fundamental tension:

> **LLMs are flexible and probabilistic. Business rules must be deterministic.**

An LLM asked to guide a user through a phone upgrade might hallucinate an eligibility outcome, skip a required check, or route the user incorrectly based on conversational context rather than verified data. This is unacceptable in production.

The goal of this system is to **get the best of both worlds**: use the LLM for what it's good at (conversation, intent understanding, data extraction, empathy), and use a deterministic FSM for what it's good at (routing, condition evaluation, state management).

---

## 2. The Core Design Principle

**The LLM can never move the workflow forward on its own.**

Every state transition is controlled exclusively by the FSM, which only advances when verified data satisfies a predefined condition. The LLM's job is to collect that data — by talking to the user and calling backend tools — and hand it to the FSM. The FSM decides what happens next, not the LLM.

This separation is strict and intentional:

| Responsibility | Owner |
|---|---|
| Conversational tone and empathy | LLM |
| Deciding what to ask the user | LLM (guided by objective) |
| Calling backend tools | LLM |
| Translating tool responses into structured data | LLM |
| Evaluating business conditions | FSM |
| Deciding which state to move to | FSM |
| Enforcing sequence and rules | FSM |

---

## 3. Architecture Overview

The system has three layers with strict separation of concerns:

```
┌────────────────────────────────────────┐
│              CONVERSATIONAL LAYER      │
│                                        │
│  LLM (Gemini 2.5 Pro via Google ADK)  │
│                                        │
│  Talks to the user. Calls tools.       │
│  Extracts structured data.             │
│  Knows what to do next because the     │
│  Bridge tells it via its system prompt.│
└──────────────────┬─────────────────────┘
                   │ structured data
┌──────────────────▼─────────────────────┐
│              BRIDGE LAYER              │
│                                        │
│  Python (ADK callbacks)                │
│                                        │
│  Translates between the LLM world and  │
│  the FSM world. Maintains the ledger.  │
│  Injects FSM context into LLM prompts. │
│  Routes FSM events when triggered.     │
└──────────────────┬─────────────────────┘
                   │ ledger + state
┌──────────────────▼─────────────────────┐
│              ROUTING LAYER             │
│                                        │
│  Python FSM (driven by YAML rules)     │
│                                        │
│  Evaluates conditions against ledger.  │
│  Advances state deterministically.     │
│  Handles change-of-mind rewinds.       │
│  Has zero knowledge of conversation.   │
└────────────────────────────────────────┘
```

Each layer communicates with only the layer immediately adjacent to it. The LLM never touches the FSM directly, and the FSM never touches the conversation.

---

## 4. Key Design Decisions

### 4.1 YAML as the Single Source of Business Truth

All business logic lives in a YAML configuration file:
- The states of the workflow and their objectives
- The data that must be collected at each state
- The conditions that determine which state to transition to
- The change-of-mind intents and their memory-wipe rules

**Why?** This makes the system genuinely maintainable by non-engineers. Adding a new state, changing an eligibility condition, or introducing a new change-of-mind scenario requires only a YAML edit. No Python deployment, no test changes, no risk of introducing a code bug.

The Python code is fully domain-agnostic — it contains no state names, no field names, no condition logic.

### 4.2 The LLM Does Not Filter Tools or Know the State Machine

The LLM always sees all available tools. It is never told which tools are "allowed" in the current state.

**Why?** Early versions filtered tools per state — the idea being that restricting the LLM's options would prevent hallucination. This was rejected because:

1. **The FSM is the real guardrail.** Even if the LLM calls the wrong tool, the FSM will not advance unless the correct conditions are met with verified data.
2. **Tool filtering adds fragile coupling.** It requires maintaining a mapping between states and tools — which lives outside the YAML and must be kept in sync manually.
3. **The LLM infers correctly from the objective.** When told "verify the user is an authorized account holder," it naturally reaches for the auth tool without being forced.

The objective, not the tool list, guides the LLM's decisions.

### 4.3 Explicit FSM Advancement (Not Implicit)

The LLM explicitly calls a dedicated `fsm_advance` tool after every domain tool call, passing the structured data it collected. The FSM then evaluates and returns the next objective.

**Why not advance the FSM automatically when a tool responds?**

Advancing automatically inside a tool callback was considered. It was rejected because it creates hidden, hard-to-debug behaviour. When FSM advancement is an explicit tool call, the interaction is transparent: the LLM decides when it has enough data, structures it into the correct format, and consciously hands it to the FSM. The conversation trace shows exactly what data moved the workflow forward and why.

This also enables **slot-filling**: within a single user turn, the LLM can call multiple tools and advance the FSM multiple times — as long as it has the data to do so — before pausing to ask the user for something it doesn't have yet.

### 4.4 The Ledger as Accumulating Memory

Rather than passing full data payloads on every tool call, the system maintains a cumulative ledger — a structured JSON object that grows across the conversation. Each state's data is added to the appropriate section of the ledger; nothing is ever overwritten unless a change-of-mind explicitly clears it.

**Why?** The FSM evaluates conditions against the entire ledger, not just the latest response. This means a transition condition at step 8 can reference data collected at step 2 without the LLM needing to re-collect it or the system needing to pass it forward explicitly.

### 4.5 Change-of-Mind via Explicit Intent + Targeted Memory Wipes

When a user changes a previous decision (e.g., "actually I want a different phone"), the LLM calls a dedicated intent-signalling tool. The FSM then:
1. Wipes only the ledger sections that are downstream of the change
2. Jumps the state pointer back to the appropriate step

**Why targeted wipes?** If a user changes their device choice after receiving a trade-in quote, the trade-in quote is still valid — only the device pricing and order details need clearing. Wiping everything would force unnecessary re-collection of data.

The list of intents and which ledger keys to clear are defined entirely in YAML. The LLM learns what intents are available from its dynamically-generated system prompt — no intent names are hardcoded in Python.

### 4.6 Dynamic System Prompt (Not Static Instructions)

Before every LLM request, the system rewrites the system prompt to tell the LLM exactly what it must accomplish in the current state and what data it needs to collect. This is derived live from the FSM's current state and the YAML.

**Why?** A static system prompt that describes the entire workflow would be long, confusing, and would require the LLM to figure out where it is in the flow. A dynamic prompt that says "right now, your job is X, and you need these specific fields" is both shorter and more precise — leading to far more reliable behaviour.

---

## 5. What This Enables

### Zero-Code Business Rule Changes

The key demo value of this system: change a business rule in YAML and it immediately affects routing, the LLM's objectives, and the change-of-mind options — with no Python changes and no redeployment of application logic.

For example:
- Add a new eligibility check: add a state and two transitions in YAML
- Change what constitutes "good standing": edit one condition string in YAML
- Add a new change-of-mind scenario: add one transition block with `transition_type: global` in YAML

### Semantic Resilience to API Changes

If a backend API changes its response field names (e.g., `is_authorized` becomes `authorization_status: GRANTED`), the LLM will still correctly map the new field to the required ledger structure — because it understands the meaning, not just the key name. No YAML or Python changes needed.

This is a meaningful advantage over traditional rules engines, where field name changes ripple through mapping tables and configuration.

---

## 6. Trade-offs and Limitations

### LLM Non-Determinism
The system is deterministic at the routing level but not at the conversational level. The LLM might phrase a question differently on each run. For workflows where exact scripted phrasing is required, additional constraints would need to be added.

### Context Growth
In a long conversation, the LLM's context window accumulates tool responses and conversation history. For the POC this is acceptable. In production, a sub-agent architecture (see Section 7) would isolate tool call context per state.

### Condition String Syntax
YAML condition strings must follow a specific syntax compatible with the safe evaluation library used. Complex conditions (e.g., range checks, multi-field logic) are possible but require care. This is a documentation and onboarding concern, not a fundamental limitation.

---

## 7. Production Path: Sub-Agent Architecture

The POC uses a single agent. In production, context isolation across a long workflow is important. The recommended V2 architecture introduces a sub-agent:

- **Root agent:** Owns the entire conversation. Knows the current FSM state. Decides what to ask the user. Delegates tool execution to the sub-agent.
- **Sub-agent:** Receives only the current objective and executes exactly one tool call. Returns structured data. Has no conversation history and no context from previous states.

The root agent processes the sub-agent's output, advances the FSM, and continues the conversation. Tool responses never accumulate in the root agent's context — only the user conversation and a small JSON summary per step do.

This pattern scales cleanly to long workflows without context window pressure.
