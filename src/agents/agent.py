import os
import re
import json
import logging

from google.adk import Agent
from google.adk.tools.mcp_tool import McpToolset, StreamableHTTPConnectionParams
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.tool_context import ToolContext

from .orchestrator.fsm import WorkflowFSM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FSM (stateless — reads/writes via callback_context.state)
# ---------------------------------------------------------------------------

fsm = WorkflowFSM(os.path.abspath("config/phone_upgrade.yaml"))

# Loaded once at startup from YAML — no hardcoding of intent names in Python.
_GLOBAL_INTENTS = fsm.get_global_intents()
_GLOBAL_INTENTS_TEXT = '\n'.join(
    f'  - {g["trigger"]}: {g["description"]}' for g in _GLOBAL_INTENTS
)

# Build fsm_advance docstring examples dynamically from YAML extract_variables.
# Groups all fields by context key so examples reflect the actual schema.
def _build_fsm_advance_examples() -> str:
    groups: dict = {}
    for var_path in fsm.get_all_extract_variables():
        parts = var_path.split('.')
        if len(parts) == 2:
            ctx_key, field = parts
            groups.setdefault(ctx_key, []).append(field)
    lines = []
    for ctx_key, fields in groups.items():
        example = {ctx_key: {f: "<value>" for f in fields}}
        lines.append(f'              {json.dumps(example)}')
    return '\n'.join(lines)

_FSM_ADVANCE_EXAMPLES = _build_fsm_advance_examples()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json_block(text: str) -> dict:
    """Parse the first ```json ... ``` block from LLM response text."""
    match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return {}


def _deep_merge(base: dict, override: dict):
    """Recursively merge override into base in-place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _normalize_booleans(obj):
    """Recursively convert string 'true'/'false' to Python booleans.

    The LLM occasionally emits boolean values as strings ("true"/"false").
    FSM conditions use == True / == False so string values cause silent failures.
    """
    if isinstance(obj, dict):
        return {k: _normalize_booleans(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_booleans(v) for v in obj]
    if isinstance(obj, str):
        if obj.lower() == 'true':
            return True
        if obj.lower() == 'false':
            return False
    return obj


# ---------------------------------------------------------------------------
# fsm_advance — internal ADK tool (LLM calls this after each MCP tool)
# ---------------------------------------------------------------------------

def fsm_advance(data: dict, tool_context: ToolContext) -> dict:
    """Advance the workflow state with data collected from an MCP tool.

    WHEN TO CALL:
      After every MCP tool call, immediately and without exception.
      Pattern: MCP tool → `fsm_advance` → MCP tool → `fsm_advance`
      Never produce a text response when the last MCP tool called was not followed by `fsm_advance`.

    Args:
        data: Structured data from the tool response, nested by context group.
{_FSM_ADVANCE_EXAMPLES}

    Returns:
        workflow_advanced_to: the new FSM state name
        next_objective: what you must accomplish next
        fields_to_collect: fields required for the next step — scan conversation
            history first; if the value is already known, call the appropriate
            MCP tool with it immediately rather than asking the user again.
        next_action: CONTINUE means call the next tool now; ASK_USER means the
            workflow needs new input from the user.
    """
    ledger = tool_context.state.get("ledger", {})
    normalized = _normalize_booleans(data)
    _deep_merge(ledger, normalized)
    tool_context.state["ledger"] = ledger

    current_state = tool_context.state.get("fsm_state", fsm.initial_state)
    new_state = fsm.evaluate(current_state, ledger)
    tool_context.state["fsm_state"] = new_state

    logger.info(f"[FSM] {current_state} → {new_state} | data: {data}")

    fields = fsm.get_extract_variables(new_state)
    next_action = "ASK_USER" if (fsm.is_terminal(new_state) or not fields) else "CONTINUE"

    return {
        "workflow_advanced_to": new_state,
        "next_objective": fsm.get_objective(new_state),
        "fields_to_collect": fields,
        "next_action": next_action,
    }


fsm_advance.__doc__ = fsm_advance.__doc__.replace('{_FSM_ADVANCE_EXAMPLES}', _FSM_ADVANCE_EXAMPLES)


# ---------------------------------------------------------------------------
# detect_intent — ADK tool for change-of-mind global transitions
# ---------------------------------------------------------------------------

def detect_intent(intent: str, tool_context: ToolContext) -> dict:
    """Signal that the user wants to change a previous choice in the workflow.

    WHEN TO CALL: When the user expresses a change of mind about a previous selection.
    Do NOT call `fsm_advance` after this — it handles FSM advancement internally.

    Args:
        intent: Exact trigger name from the CHANGE OF MIND list in your instructions.

    Returns:
        workflow_rewound_to: the state the workflow has rewound to
        next_objective: what you must accomplish next
    """
    ledger = tool_context.state.get("ledger", {})
    current_state = tool_context.state.get("fsm_state", fsm.initial_state)
    new_state = fsm.fire_intent(current_state, intent, ledger)
    tool_context.state["fsm_state"] = new_state
    tool_context.state["ledger"] = ledger
    logger.info(f"[FSM] Intent '{intent}': {current_state} → {new_state}")
    return {
        "workflow_rewound_to": new_state,
        "next_objective": fsm.get_objective(new_state),
    }


# ---------------------------------------------------------------------------
# System prompt — static sections (built once at startup)
# ---------------------------------------------------------------------------

_BRAND = (
    'You are Alex, a friendly and knowledgeable Verizon customer service representative.\n\n'
    'BRAND VOICE\n'
    '- Greet first-time callers warmly: "Welcome to Verizon! I\'m Alex, and I\'m here to help."\n'
    '- Use "we" and "our" when referring to Verizon (e.g. "our plans", "we can offer you").\n'
    '- Be empathetic and patient. Acknowledge the customer\'s situation before moving forward.\n'
    '- Celebrate good news with genuine enthusiasm; deliver bad news kindly with next steps.\n'
    '- Always thank the customer for choosing Verizon at the end of a completed interaction.\n'
    '- Keep responses concise but complete. Never sound robotic or scripted.\n\n'
)

_TOOL_CONTRACT = (
    'TOOL CONTRACT\n'
    '- After every MCP tool call, immediately call `fsm_advance`. No exceptions.\n'
    '  Pattern: MCP tool → `fsm_advance` → MCP tool → `fsm_advance`\n'
    '- Ending a turn with an MCP tool as the last call is an error.\n'
    '- `detect_intent` handles FSM advancement internally — do NOT call `fsm_advance` after it.\n\n'
)

_HISTORY_CONTRACT = (
    'HISTORY CONTRACT\n'
    '- Before asking the user for anything, scan the full conversation history.\n'
    '- If the value is already there, call the appropriate MCP tool with it immediately.\n'
    '  Never ask for information the user has already provided.\n\n'
)

_CONTINUATION_CONTRACT = (
    'CONTINUATION CONTRACT\n'
    '- After `fsm_advance` returns next_action=CONTINUE: call the next tool immediately.\n'
    '  Do not produce any text. Do not narrate. Do not pause.\n'
    '- After `fsm_advance` returns next_action=ASK_USER: produce one response —\n'
    '  single, cohesive, warm — summarising all outcomes of this turn, then stop.\n'
    '- Never produce text mid-turn ("let me check", "I\'ll get the pricing now", etc.).\n'
    '  If you can call a tool, call it. Narrating instead of acting is an error.\n\n'
)

_CHANGE_OF_MIND = (
    'CHANGE OF MIND\n'
    'If the user changes a previous choice, call `detect_intent` with the exact trigger name.\n'
    '`detect_intent` handles FSM advancement internally — do NOT call `fsm_advance` after it.\n'
    f'{_GLOBAL_INTENTS_TEXT}\n'
)

# Combined static instruction — injected every turn unchanged
_STATIC_INSTRUCTION = _BRAND + _TOOL_CONTRACT + _HISTORY_CONTRACT + _CONTINUATION_CONTRACT + _CHANGE_OF_MIND


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def before_model(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
):
    current_state = callback_context.state.get("fsm_state", fsm.initial_state)
    extract_vars = fsm.get_extract_variables(current_state)

    # Build a per-state fsm_advance example from YAML extract_variables
    example_data: dict = {}
    for var_path in extract_vars:
        parts = var_path.split('.')
        if len(parts) == 2:
            ctx_key, field = parts
            example_data.setdefault(ctx_key, {})[field] = "<value>"

    where_you_are = (
        'WHERE YOU ARE\n'
        f'State:     {current_state}\n'
        f'Objective: {fsm.get_objective(current_state)}\n'
        f'`fsm_advance` example: `fsm_advance(data={json.dumps(example_data)})`\n'
        'Reminder: call `fsm_advance` immediately after every tool call.\n'
    )

    if llm_request.config:
        llm_request.config.system_instruction = _STATIC_INSTRUCTION + where_you_are

    return None


def after_model(
    callback_context: CallbackContext,
    llm_response,
):
    """
    Fallback: parses ```json``` block from LLM text and advances FSM.

    Primary flow uses fsm_advance tool. This fires for terminal states or if
    the LLM produces a JSON block instead of calling fsm_advance.
    """
    parts = getattr(getattr(llm_response, 'content', None), 'parts', []) or []
    has_function_calls = any(getattr(p, 'function_call', None) for p in parts)
    if has_function_calls:
        return None

    text = "".join(p.text for p in parts if getattr(p, 'text', None))
    json_data = _normalize_booleans(_extract_json_block(text))

    if not json_data:
        return None

    ledger = callback_context.state.get("ledger", {})
    _deep_merge(ledger, json_data)
    callback_context.state["ledger"] = ledger

    current_state = callback_context.state.get("fsm_state", fsm.initial_state)
    while True:
        new_state = fsm.evaluate(current_state, ledger)
        if new_state == current_state:
            break
        logger.info(f"[FSM] {current_state} → {new_state}")
        current_state = new_state
    callback_context.state["fsm_state"] = current_state
    logger.info(f"[FSM] final state: {current_state} | extracted: {json_data}")

    return None


# ---------------------------------------------------------------------------
# MCP Toolset (HTTP transport — Phase 2)
# ---------------------------------------------------------------------------

MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://127.0.0.1:8080/mcp")

mcp_toolset = McpToolset(
    connection_params=StreamableHTTPConnectionParams(url=MCP_SERVER_URL, timeout=10.0)
)

# ---------------------------------------------------------------------------
# ADK-discoverable agent (adk web looks for `root_agent` at module level)
# ---------------------------------------------------------------------------

root_agent = Agent(
    name="TelcoManager",
    model="gemini-2.5-pro",
    instruction="You are Alex, a friendly Verizon customer service representative.",
    tools=[mcp_toolset, fsm_advance, detect_intent],
    before_model_callback=before_model,
    after_model_callback=after_model,
)
