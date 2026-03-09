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
# Module-level constants
# ---------------------------------------------------------------------------

STATE_TOOLS = {
    "Auth":                        ["verify_auth"],
    "AccountStandingCheck":        ["check_standing"],
    "LineToUpgrade":               ["set_line"],
    "CheckLineUpgradeEligibility": ["check_eligibility"],
    "VerifyTradeIn":               ["set_trade_in_preference"],
    "DeviceTradeInChecks":         ["record_condition"],
    "TradeInPricing":              ["pricing"],
    "NewUpgradeDeviceSelection":   ["select_device"],
    "NewUpgradeDevicePricing":     ["pricing"],
    "FinalPricing":                ["confirm_order", "decline_order"],
    "ProcessOrder":                ["submit_order"],
}

GLOBAL_INTENT_STATES = set(STATE_TOOLS.keys()) - {"Auth", "AccountStandingCheck"}


def _empty_ledger() -> dict:
    return {
        "account_context": {},
        "line_context": {},
        "trade_in_context": {},
        "new_device_context": {},
        "order_context": {},
    }


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


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def before_model(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
):
    current_state = callback_context.state.get("fsm_state", fsm.initial_state)

    # Tool filtering (safety guardrail — unchanged)
    allowed_tools = list(STATE_TOOLS.get(current_state, []))
    if current_state in GLOBAL_INTENT_STATES:
        allowed_tools = allowed_tools + ["detect_intent"]

    if llm_request.config and llm_request.config.tools:
        for t in llm_request.config.tools:
            if getattr(t, "function_declarations", None):
                before = [f.name for f in t.function_declarations]
                t.function_declarations = [
                    f for f in t.function_declarations if f.name in allowed_tools
                ]
                after = [f.name for f in t.function_declarations]
                logger.debug(f"[Filter] state={current_state} tools: {before} → {after}")

    # Dynamic instruction: objective + structured JSON output requirement
    objective = fsm.get_objective(current_state)
    extract_vars = fsm.get_extract_variables(current_state)

    if extract_vars:
        # Build a minimal example showing the expected nested JSON structure
        example_parts = {}
        for var_path in extract_vars:
            parts = var_path.split('.')
            if len(parts) == 2:
                ctx_key, field = parts
                example_parts.setdefault(ctx_key, {})[field] = "<value>"
        example_json = json.dumps(example_parts)

        json_instruction = (
            f'2. After calling the tool, YOU MUST include in your response a JSON block '
            f'(wrapped in ```json ... ```) containing EXACTLY these nested fields:\n'
            f'   {extract_vars}\n'
            f'   Example: {example_json}\n'
        )
    else:
        json_instruction = '2. No JSON block required for this state.\n'

    dynamic_instruction = (
        f'OBJECTIVE: {objective}\n\n'
        f'RULES:\n'
        f'1. Use the provided tool(s) to collect the required data.\n'
        f'{json_instruction}'
        f'3. If the user changes their mind, call detect_intent(intent) with one of: '
        f'change_line, change_trade_in_device, change_new_device.'
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
    """Only handles detect_intent. All other tool results are handled by after_model."""
    if tool.name != "detect_intent":
        return tool_response

    result_data = _parse_mcp_response(tool_response)
    intent = result_data.get("detected_intent", "")
    if intent:
        ledger = tool_context.state.get("ledger", _empty_ledger())
        current_state = tool_context.state.get("fsm_state", fsm.initial_state)
        new_state = fsm.fire_intent(current_state, intent, ledger)
        tool_context.state["fsm_state"] = new_state
        tool_context.state["ledger"] = ledger  # ledger keys cleared in-place by fire_intent
        logger.info(f"[FSM] Intent '{intent}': {current_state} → {new_state}")

    return tool_response


def after_model(
    callback_context: CallbackContext,
    llm_response,
):
    """
    Parses structured JSON from LLM final text responses, merges into ledger,
    and advances the FSM. Skips intermediate function-call responses.
    """
    # Skip if this response contains function calls (tool is being called — not final yet)
    parts = getattr(getattr(llm_response, 'content', None), 'parts', []) or []
    has_function_calls = any(getattr(p, 'function_call', None) for p in parts)
    if has_function_calls:
        return None

    text = "".join(p.text for p in parts if getattr(p, 'text', None))
    json_data = _extract_json_block(text)

    if not json_data:
        return None  # No JSON block — conversational reply or terminal state

    ledger = callback_context.state.get("ledger", _empty_ledger())
    _deep_merge(ledger, json_data)
    callback_context.state["ledger"] = ledger

    current_state = callback_context.state.get("fsm_state", fsm.initial_state)
    new_state = fsm.evaluate(current_state, ledger)
    callback_context.state["fsm_state"] = new_state
    logger.info(f"[FSM] {current_state} → {new_state} | extracted: {json_data}")

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
    tools=[mcp_toolset],
    before_model_callback=before_model,
    after_tool_callback=after_tool,
    after_model_callback=after_model,
)
