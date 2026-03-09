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

from orchestrator.fsm import WorkflowFSM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FSM (stateless — reads/writes via callback_context.state)
# ---------------------------------------------------------------------------

fsm = WorkflowFSM(os.path.abspath("config/phone_upgrade.yaml"))


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
    """Advance the workflow after collecting data from a domain tool.

    Call this after EVERY domain tool call, passing the structured data you collected.
    It updates the workflow state and tells you what to do next.

    Args:
        data: Structured data from the tool response, nested by context group. Examples:
              {"account_context": {"is_authorized": true}}
              {"account_context": {"standing": "GOOD"}}
              {"line_context": {"selected_number": "555-1234"}}
              {"line_context": {"is_eligible": true}}
              {"trade_in_context": {"wants_trade_in": false}}
              {"trade_in_context": {"final_condition": "Good", "quote_value": 200}}
              {"new_device_context": {"selection": "iPhone 16", "price": 1000}}
              {"order_context": {"order_id": "ORD-123", "error": false}}

    Returns:
        workflow_advanced_to: the new FSM state name
        next_objective: what you must accomplish next
        data_still_needed: list of data fields still required for the next step
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
        "data_still_needed": fsm.get_extract_variables(new_state),
    }


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
        'You are a warm, helpful Telco Customer Service AI assisting with a phone upgrade.\n\n'

        f'CURRENT STATE: {current_state}\n'
        f'CURRENT OBJECTIVE: {objective}\n\n'

        'WORKFLOW RULES:\n'
        '1. Call the appropriate domain tool(s) to fulfill the current objective.\n'
        f'2. After each domain tool call, call fsm_advance with the data you collected.\n'
        f'   Example for this step: fsm_advance(data={example_json})\n'
        '3. fsm_advance returns the next objective and what data is still needed.\n'
        '   - If you already have all needed information from the conversation, '
        'call the next tool immediately — do NOT ask the user for info you already have.\n'
        '   - If you need information the user has not yet provided, ask naturally.\n'
        '4. Continue this loop (tool → fsm_advance → tool → fsm_advance) until you '
        'reach a point where you must ask the user something.\n'
        '5. Keep all responses warm, concise, and conversational. Never sound robotic.\n\n'

        'CHANGE OF MIND — if the user changes a previous choice, call detect_intent(intent) '
        'with one of: change_line, change_trade_in_device, change_new_device.'
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
    instruction="You are a helpful Telco Customer Service AI.",
    tools=[mcp_toolset, fsm_advance],
    before_model_callback=before_model,
    after_tool_callback=after_tool,
    after_model_callback=after_model,
)
