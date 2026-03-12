import os
import re
import json
import logging

from google.adk import Agent
from google.adk.tools.mcp_tool import McpToolset, StreamableHTTPConnectionParams
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.base_tool import BaseTool
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

def _parse_mcp_response(tool_response) -> dict:
    """Unwrap MCP envelope and return parsed dict."""
    try:
        if isinstance(tool_response, dict):
            content_list = tool_response.get("content")
            if isinstance(content_list, list) and content_list:
                first = content_list[0]
                if isinstance(first, dict) and "text" in first:
                    return json.loads(first["text"])
            return tool_response
        elif isinstance(tool_response, list):
            return json.loads(tool_response[0].text)
        elif isinstance(tool_response, str):
            return json.loads(tool_response)
    except Exception as e:
        logger.warning(f"[Hook Warning] Could not parse tool response: {e}")
    return {}


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
# fsm_advance — internal ADK tool (LLM calls this after each domain tool)
# ---------------------------------------------------------------------------

def fsm_advance(data: dict, tool_context: ToolContext) -> dict:
    """Advance the workflow after collecting data from a domain tool OR from the conversation.

    Call this after EVERY domain tool call, passing the structured data you collected.
    You may also call this directly with data you extracted from the user's messages
    (without calling a domain tool first) when the user has already provided the needed values.

    Args:
        data: Structured data collected from a tool response or the user's message,
              nested by context group.
{_FSM_ADVANCE_EXAMPLES}

    Returns:
        workflow_advanced_to: the new FSM state name
        next_objective: what you must accomplish next
        fields_to_collect: fields required for the next step — IMPORTANT: scan the
            ENTIRE conversation history for these values before asking the user.
            If found in history, call the relevant domain tool or fsm_advance immediately
            without asking. Only ask the user if the value is genuinely absent from
            the conversation.
    """
    ledger = tool_context.state.get("ledger", {})
    normalized = _normalize_booleans(data)
    _deep_merge(ledger, normalized)
    tool_context.state["ledger"] = ledger

    current_state = tool_context.state.get("fsm_state", fsm.initial_state)
    new_state = fsm.evaluate(current_state, ledger)
    tool_context.state["fsm_state"] = new_state

    logger.info(f"[FSM] {current_state} → {new_state} | data: {data}")

    return {
        "workflow_advanced_to": new_state,
        "next_objective": fsm.get_objective(new_state),
        "fields_to_collect": fsm.get_extract_variables(new_state),
    }


fsm_advance.__doc__ = fsm_advance.__doc__.replace('{_FSM_ADVANCE_EXAMPLES}', _FSM_ADVANCE_EXAMPLES)


# ---------------------------------------------------------------------------
# Global brand persona — constant across all states
# ---------------------------------------------------------------------------

BRAND_INSTRUCTION = (
    'You are Alex, a friendly and knowledgeable Verizon customer service representative.\n\n'

    'BRAND VOICE:\n'
    '- Greet first-time callers warmly: "Welcome to Verizon! I\'m Alex, and I\'m here to help."\n'
    '- Use "we" and "our" when referring to Verizon (e.g. "our plans", "we can offer you").\n'
    '- Be empathetic and patient. Acknowledge the customer\'s situation before moving forward.\n'
    '- Celebrate good news (eligible line, successful order) with genuine enthusiasm.\n'
    '- When delivering bad news (not authorized, not eligible), be kind, apologetic, and offer next steps.\n'
    '- Always end a completed interaction by thanking the customer for choosing Verizon.\n'
    '- Keep responses concise but complete — never leave the customer wondering what happens next.\n'
    '- Never sound robotic, scripted, or mechanical. Sound like a real, caring person.\n\n'
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def before_model(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
):
    current_state = callback_context.state.get("fsm_state", fsm.initial_state)

    objective = fsm.get_objective(current_state)
    extract_vars = fsm.get_extract_variables(current_state)

    # Build a concrete example of what to pass to fsm_advance for the current step
    example_data: dict = {}
    for var_path in extract_vars:
        parts = var_path.split('.')
        if len(parts) == 2:
            ctx_key, field = parts
            example_data.setdefault(ctx_key, {})[field] = "<value>"
    example_json = json.dumps(example_data)

    dynamic_instruction = (
        BRAND_INSTRUCTION +

        f'CURRENT STATE: {current_state}\n'
        f'CURRENT OBJECTIVE: {objective}\n\n'

        'WORKFLOW RULES:\n'
        '1. Before calling any tool, SCAN THE ENTIRE CONVERSATION HISTORY for values '
        'the current objective needs. Users often provide information early — a device '
        'name, a line number, a trade-in preference — before you have reached that step.\n'
        '2. For each field listed in fields_to_collect:\n'
        '   a. If the value exists ANYWHERE in conversation history → use it immediately '
        '(call the domain tool with that value, or call fsm_advance directly with it).\n'
        '   b. If the value is genuinely absent from the conversation → ask the user.\n'
        '   NEVER ask for information the user has already provided.\n'
        f'3. After each domain tool call, call fsm_advance with the data you collected.\n'
        f'   Example for this step: fsm_advance(data={example_json})\n'
        '4. You may also call fsm_advance directly (without a domain tool) when the user '
        'has already stated the needed value in conversation — just pass the extracted data.\n'
        '5. After fsm_advance returns, immediately check:\n'
        '   - Do I have EVERYTHING needed for the next_objective from history or tools?\n'
        '   - If YES → call the next tool immediately. Do NOT pause. Do NOT narrate.\n'
        '   - If NO  → ask the user only for what is missing. Be specific and concise.\n'
        '6. Continue this loop (tool → fsm_advance → tool → fsm_advance) until you '
        'genuinely cannot proceed without new information from the user.\n\n'

        'RESPONSE RULES (CRITICAL — read carefully):\n'
        '- SILENCE during tool execution: Do NOT produce ANY text before, between, or '
        'around tool calls. No greetings, no "let me check that", no "I\'ll now get the '
        'pricing", no status updates, no narration of any kind while tools are running.\n'
        '- NEVER produce a response like "Let me get the pricing for that" and then stop. '
        'If you can call a tool, CALL IT. A response like that with no follow-up tool call '
        'is a failure — you are making the user wait for no reason.\n'
        '- Speak ONCE, at the end: Only produce your single response AFTER every tool '
        'call for this turn is fully complete.\n'
        '- Your response must be a SINGLE, COHESIVE message that summarises ALL outcomes '
        'of this turn and tells the user exactly what happens next or what you need from them.\n'
        '- Example of a good response after auth+standing+eligibility all in one turn:\n'
        '  "Welcome to Verizon! I\'m Alex. Your account is verified and in great standing, '
        'and the line 555-111-2222 is eligible for an upgrade. Would you like to trade in '
        'your current device?"\n'
        '- Never produce step-by-step narration. Synthesise all results into one '
        'natural, warm sentence or two.\n\n'

        f'CHANGE OF MIND — if the user changes a previous choice, call detect_intent(intent) '
        f'passing the exact trigger name from this list:\n{_GLOBAL_INTENTS_TEXT}'
    )

    if llm_request.config:
        llm_request.config.system_instruction = dynamic_instruction

    return None


def after_tool(
    tool: BaseTool,
    args: dict,
    tool_context: ToolContext,
    tool_response: dict,
):
    """Handles detect_intent for change-of-mind flows. fsm_advance is handled by the tool itself."""
    if tool.name != "detect_intent":
        return tool_response

    result_data = _parse_mcp_response(tool_response)
    intent = result_data.get("detected_intent", "")
    if intent:
        ledger = tool_context.state.get("ledger", {})
        current_state = tool_context.state.get("fsm_state", fsm.initial_state)
        new_state = fsm.fire_intent(current_state, intent, ledger)
        tool_context.state["fsm_state"] = new_state
        tool_context.state["ledger"] = ledger
        logger.info(f"[FSM] Intent '{intent}': {current_state} → {new_state}")

    return tool_response


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
    tools=[mcp_toolset, fsm_advance],
    before_model_callback=before_model,
    after_tool_callback=after_tool,
    after_model_callback=after_model,
)
