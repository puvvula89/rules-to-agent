import os
import json

from google.adk import Agent
from google.adk.tools.mcp_tool import McpToolset, StreamableHTTPConnectionParams
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from orchestrator.fsm import WorkflowFSM

# ---------------------------------------------------------------------------
# Module-level constants (not re-created per LLM call)
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

TOOL_LEDGER_MAP = {
    "verify_auth":             "account_context",
    "check_standing":          "account_context",
    "set_line":                "line_context",
    "check_eligibility":       "line_context",
    "set_trade_in_preference": "trade_in_context",
    "record_condition":        "trade_in_context",
    "pricing":                 None,   # determined by current FSM state
    "select_device":           "new_device_context",
    "confirm_order":           "order_context",
    "decline_order":           "order_context",
    "submit_order":            "order_context",
    "detect_intent":           None,   # triggers global FSM transition
}


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
# Callbacks
# ---------------------------------------------------------------------------

def before_model(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
):
    current_state = callback_context.state.get("fsm_state", fsm.initial_state)

    allowed_tools = list(STATE_TOOLS.get(current_state, []))
    if current_state in GLOBAL_INTENT_STATES:
        allowed_tools = allowed_tools + ["detect_intent"]

    if llm_request.config and llm_request.config.tools:
        for t in llm_request.config.tools:
            if getattr(t, "function_declarations", None):
                t.function_declarations = [
                    f for f in t.function_declarations if f.name in allowed_tools
                ]

    objective = fsm.get_objective(current_state)
    dynamic_instruction = (
        f'CRITICAL RULES (FSM Guardrail):\n'
        f'Your CURRENT OBJECTIVE is strictly defined by the backend state machine:\n'
        f'"{objective}"\n\n'
        f'1. Do NOT decide what to do next. Only collect the data required by the tool provided to you.\n'
        f'2. If you have the data, call the explicit tool provided.\n'
        f'3. If the user changes their mind about a previous choice, call '
        f'detect_intent(intent) with one of: change_line, change_trade_in_device, change_new_device.'
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
    # Parse tool_response safely — ADK may deliver dict, list[TextContent], or str
    try:
        if isinstance(tool_response, dict):
            result_data = tool_response
        elif isinstance(tool_response, list):
            result_data = json.loads(tool_response[0].text)
        elif isinstance(tool_response, str):
            result_data = json.loads(tool_response)
        else:
            result_data = {}
    except Exception as e:
        print(f"[Hook Warning] Could not parse tool result: {e}, type: {type(tool_response)}")
        result_data = {}

    ledger = tool_context.state.get("ledger", _empty_ledger())
    current_state = tool_context.state.get("fsm_state", fsm.initial_state)

    # detect_intent: global FSM transition (wipes ledger keys in-place), early return
    if tool.name == "detect_intent":
        intent = result_data.get("detected_intent", "")
        if intent:
            new_state = fsm.evaluate(current_state, ledger, intent_override=intent)
            tool_context.state["fsm_state"] = new_state
            tool_context.state["ledger"] = ledger  # ledger may have been wiped
        return tool_response

    # Determine ledger context key
    if tool.name == "pricing":
        context_key = "trade_in_context" if current_state == "TradeInPricing" else "new_device_context"
    else:
        context_key = TOOL_LEDGER_MAP.get(tool.name)

    if context_key and isinstance(result_data, dict):
        ledger.setdefault(context_key, {}).update(result_data)
        print(f"[Ledger] Updated {context_key}: {ledger[context_key]}")

    tool_context.state["ledger"] = ledger
    new_state = fsm.evaluate(current_state, ledger)
    tool_context.state["fsm_state"] = new_state

    return tool_response


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
)
