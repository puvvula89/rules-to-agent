"""
Integration tests for the full agent flow.

Two layers:
  1. MCP server HTTP smoke tests — verifies all 12 tools respond correctly
  2. Callback pipeline simulation — exercises before_model + after_tool + FSM
     with mock ADK objects (no real LLM call needed)
"""

import sys
import os
import json
import pytest
import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

MCP_URL = os.environ.get("MCP_SERVER_URL", "http://127.0.0.1:8080/mcp")


# ---------------------------------------------------------------------------
# Helpers — minimal mocks for ADK callback objects
# ---------------------------------------------------------------------------

class MockState(dict):
    """Behaves like callback_context.state (dict-like)."""
    pass


class MockCallbackContext:
    def __init__(self, state=None):
        self.state = MockState(state or {})


class MockFunctionDeclaration:
    def __init__(self, name):
        self.name = name


class MockTool:
    def __init__(self, function_declarations):
        self.function_declarations = function_declarations


class MockLlmRequestConfig:
    def __init__(self, tool_names):
        self.tools = [MockTool([MockFunctionDeclaration(n) for n in tool_names])]
        self.system_instruction = None


class MockLlmRequest:
    def __init__(self, tool_names):
        self.config = MockLlmRequestConfig(tool_names)


class MockBaseTool:
    def __init__(self, name):
        self.name = name


# ---------------------------------------------------------------------------
# MCP HTTP smoke tests
# ---------------------------------------------------------------------------

def mcp_call(tool_name: str, arguments: dict) -> dict:
    """Send a JSON-RPC tools/call request via MCP Streamable HTTP (SSE response)."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }
    resp = httpx.post(
        MCP_URL, json=payload, timeout=10,
        headers={"Accept": "application/json, text/event-stream"},
    )
    resp.raise_for_status()
    # Response is SSE: find the `data:` line and parse it
    for line in resp.text.splitlines():
        if line.startswith("data:"):
            data = json.loads(line[len("data:"):].strip())
            assert "result" in data, f"Error response: {data}"
            return json.loads(data["result"]["content"][0]["text"])
    raise AssertionError(f"No data line in SSE response: {resp.text}")


@pytest.mark.integration
class TestMCPServer:
    def test_verify_auth_valid(self):
        result = mcp_call("verify_auth", {"account_number": "1234", "pin": "0000"})
        assert result == {"is_authorized": True}

    def test_verify_auth_invalid(self):
        result = mcp_call("verify_auth", {"account_number": "9999", "pin": "0000"})
        assert result == {"is_authorized": False}

    def test_check_standing_good(self):
        result = mcp_call("check_standing", {"account_number": "1234"})
        assert result == {"standing": "GOOD"}

    def test_check_standing_delinquent(self):
        result = mcp_call("check_standing", {"account_number": "9999"})
        assert result == {"standing": "DELINQUENT"}

    def test_check_eligibility(self):
        result = mcp_call("check_eligibility", {"phone_number": "555-1234"})
        assert result == {"is_eligible": True}

    def test_set_line(self):
        result = mcp_call("set_line", {"phone_number": "555-9876"})
        assert result == {"selected_number": "555-9876"}

    def test_set_trade_in_preference_yes(self):
        result = mcp_call("set_trade_in_preference", {"wants_trade_in": True})
        assert result == {"wants_trade_in": True}

    def test_set_trade_in_preference_no(self):
        result = mcp_call("set_trade_in_preference", {"wants_trade_in": False})
        assert result == {"wants_trade_in": False}

    def test_record_condition(self):
        result = mcp_call("record_condition", {"device_model": "iPhone 13", "condition": "Good"})
        assert result == {"trade_in_device": "iPhone 13", "final_condition": "Good"}

    def test_pricing_trade_in_iphone_excellent(self):
        result = mcp_call("pricing", {"device_model": "iPhone 13", "condition": "Excellent"})
        assert result == {"final_condition": "Excellent", "quote_value": 400}

    def test_pricing_trade_in_iphone_good(self):
        result = mcp_call("pricing", {"device_model": "iPhone 13", "condition": "Good"})
        assert result == {"final_condition": "Good", "quote_value": 200}

    def test_pricing_trade_in_pixel(self):
        result = mcp_call("pricing", {"device_model": "Pixel 8", "condition": "Excellent"})
        assert result == {"final_condition": "Excellent", "quote_value": 300}

    def test_pricing_new_device_premium(self):
        result = mcp_call("pricing", {"device_model": "iPhone 16"})
        assert result == {"selection": "iPhone 16", "price": 1000}

    def test_pricing_new_device_standard(self):
        result = mcp_call("pricing", {"device_model": "Moto G"})
        assert result == {"selection": "Moto G", "price": 800}

    def test_select_device(self):
        result = mcp_call("select_device", {"device_model": "Pixel 9"})
        assert result == {"selection": "Pixel 9"}

    def test_confirm_order(self):
        result = mcp_call("confirm_order", {})
        assert result == {"user_confirmed": True}

    def test_decline_order(self):
        result = mcp_call("decline_order", {})
        assert result == {"user_confirmed": False}

    def test_submit_order(self):
        result = mcp_call("submit_order", {
            "account_number": "1234",
            "phone_number": "555-1234",
            "new_device": "iPhone 16",
        })
        assert result["order_id"] == "ORD-999888777"
        assert result["error"] is False

    def test_detect_intent(self):
        result = mcp_call("detect_intent", {"intent": "change_new_device"})
        assert result == {"detected_intent": "change_new_device"}


# ---------------------------------------------------------------------------
# Callback pipeline simulation (no LLM)
# ---------------------------------------------------------------------------

from agents.manager import (
    before_model, after_tool, fsm, STATE_TOOLS, GLOBAL_INTENT_STATES
)

ALL_TOOL_NAMES = [
    "verify_auth", "check_standing", "set_line", "check_eligibility",
    "set_trade_in_preference", "record_condition", "pricing",
    "select_device", "confirm_order", "decline_order", "submit_order", "detect_intent",
]


def run_before_model(state: str, state_dict: dict = None) -> MockLlmRequest:
    """Run before_model callback and return the mutated request."""
    ctx = MockCallbackContext({"fsm_state": state, **(state_dict or {})})
    req = MockLlmRequest(ALL_TOOL_NAMES)
    before_model(ctx, req)
    return req


def visible_tools(req: MockLlmRequest) -> list[str]:
    return [f.name for f in req.config.tools[0].function_declarations]


def run_after_tool(tool_name: str, result: dict, state: str, ledger: dict = None) -> MockCallbackContext:
    """Run after_tool callback and return the updated context."""
    ledger = ledger or {
        "account_context": {}, "line_context": {}, "trade_in_context": {},
        "new_device_context": {}, "order_context": {},
    }
    ctx = MockCallbackContext({"fsm_state": state, "ledger": ledger})
    after_tool(MockBaseTool(tool_name), {}, ctx, result)
    return ctx


class TestBeforeModelFiltering:
    def test_auth_state_only_sees_verify_auth(self):
        assert visible_tools(run_before_model("Auth")) == ["verify_auth"]

    def test_account_standing_only_sees_check_standing(self):
        assert visible_tools(run_before_model("AccountStandingCheck")) == ["check_standing"]

    def test_verify_trade_in_sees_tool_and_detect_intent(self):
        tools = visible_tools(run_before_model("VerifyTradeIn"))
        assert "set_trade_in_preference" in tools
        assert "detect_intent" in tools
        assert "verify_auth" not in tools

    def test_final_pricing_sees_confirm_decline_and_detect_intent(self):
        tools = visible_tools(run_before_model("FinalPricing"))
        assert set(tools) == {"confirm_order", "decline_order", "detect_intent"}

    def test_process_order_sees_submit_and_detect_intent(self):
        tools = visible_tools(run_before_model("ProcessOrder"))
        assert "submit_order" in tools
        assert "detect_intent" in tools

    def test_auth_does_not_see_detect_intent(self):
        tools = visible_tools(run_before_model("Auth"))
        assert "detect_intent" not in tools

    def test_system_instruction_contains_objective(self):
        req = run_before_model("Auth")
        assert "Verify the user is an authorized" in req.config.system_instruction

    def test_system_instruction_references_detect_intent(self):
        req = run_before_model("LineToUpgrade")
        assert "detect_intent" in req.config.system_instruction


class TestAfterToolLedgerAndFSM:
    def test_verify_auth_success_updates_ledger_and_advances(self):
        ctx = run_after_tool("verify_auth", {"is_authorized": True}, "Auth")
        assert ctx.state["ledger"]["account_context"]["is_authorized"] is True
        assert ctx.state["fsm_state"] == "AccountStandingCheck"

    def test_verify_auth_failure_goes_to_end_unauthorized(self):
        ctx = run_after_tool("verify_auth", {"is_authorized": False}, "Auth")
        assert ctx.state["fsm_state"] == "EndUnauthorized"

    def test_check_standing_good_advances(self):
        ctx = run_after_tool("check_standing", {"standing": "GOOD"}, "AccountStandingCheck")
        assert ctx.state["fsm_state"] == "LineToUpgrade"

    def test_check_standing_bad_goes_to_end(self):
        ctx = run_after_tool("check_standing", {"standing": "DELINQUENT"}, "AccountStandingCheck")
        assert ctx.state["fsm_state"] == "EndBadStanding"

    def test_set_line_advances(self):
        ctx = run_after_tool("set_line", {"selected_number": "555-0000"}, "LineToUpgrade")
        assert ctx.state["ledger"]["line_context"]["selected_number"] == "555-0000"
        assert ctx.state["fsm_state"] == "CheckLineUpgradeEligibility"

    def test_eligibility_true_advances(self):
        ctx = run_after_tool("check_eligibility", {"is_eligible": True}, "CheckLineUpgradeEligibility")
        assert ctx.state["fsm_state"] == "VerifyTradeIn"

    def test_set_trade_in_yes_advances(self):
        ctx = run_after_tool("set_trade_in_preference", {"wants_trade_in": True}, "VerifyTradeIn")
        assert ctx.state["fsm_state"] == "DeviceTradeInChecks"

    def test_set_trade_in_no_skips_to_device_selection(self):
        ctx = run_after_tool("set_trade_in_preference", {"wants_trade_in": False}, "VerifyTradeIn")
        assert ctx.state["fsm_state"] == "NewUpgradeDeviceSelection"

    def test_record_condition_advances(self):
        ctx = run_after_tool("record_condition", {"trade_in_device": "iPhone 13", "final_condition": "Good"}, "DeviceTradeInChecks")
        assert ctx.state["fsm_state"] == "TradeInPricing"

    def test_pricing_in_trade_in_state_updates_trade_in_context(self):
        ledger = {"account_context": {}, "line_context": {}, "trade_in_context": {},
                  "new_device_context": {}, "order_context": {}}
        ctx = run_after_tool("pricing", {"final_condition": "Good", "quote_value": 200}, "TradeInPricing", ledger)
        assert ctx.state["ledger"]["trade_in_context"]["quote_value"] == 200
        assert ctx.state["fsm_state"] == "NewUpgradeDeviceSelection"

    def test_select_device_advances(self):
        ctx = run_after_tool("select_device", {"selection": "Pixel 9"}, "NewUpgradeDeviceSelection")
        assert ctx.state["ledger"]["new_device_context"]["selection"] == "Pixel 9"
        assert ctx.state["fsm_state"] == "NewUpgradeDevicePricing"

    def test_pricing_in_new_device_state_updates_new_device_context(self):
        ledger = {"account_context": {}, "line_context": {}, "trade_in_context": {},
                  "new_device_context": {}, "order_context": {}}
        ctx = run_after_tool("pricing", {"selection": "Pixel 9", "price": 1000}, "NewUpgradeDevicePricing", ledger)
        assert ctx.state["ledger"]["new_device_context"]["price"] == 1000
        assert ctx.state["fsm_state"] == "FinalPricing"

    def test_confirm_order_advances_to_process(self):
        ctx = run_after_tool("confirm_order", {"user_confirmed": True}, "FinalPricing")
        assert ctx.state["fsm_state"] == "ProcessOrder"

    def test_decline_order_stays_in_final_pricing(self):
        ctx = run_after_tool("decline_order", {"user_confirmed": False}, "FinalPricing")
        assert ctx.state["fsm_state"] == "FinalPricing"

    def test_submit_order_goes_to_end_success(self):
        ctx = run_after_tool("submit_order", {"order_id": "ORD-999888777", "error": False}, "ProcessOrder")
        assert ctx.state["fsm_state"] == "EndSuccess"


class TestDetectIntentCallback:
    def test_change_new_device_rewinds_and_wipes(self):
        ledger = {
            "account_context": {"is_authorized": True},
            "line_context": {"selected_number": "555-0000"},
            "trade_in_context": {"quote_value": 200},
            "new_device_context": {"selection": "Moto G", "price": 800},
            "order_context": {"user_confirmed": True},
        }
        ctx = run_after_tool("detect_intent", {"detected_intent": "change_new_device"}, "FinalPricing", ledger)
        assert ctx.state["fsm_state"] == "NewUpgradeDeviceSelection"
        assert ctx.state["ledger"]["new_device_context"] == {}
        assert ctx.state["ledger"]["order_context"] == {}
        assert ctx.state["ledger"]["trade_in_context"]["quote_value"] == 200  # preserved

    def test_change_line_rewinds_and_wipes_everything(self):
        ledger = {
            "account_context": {"is_authorized": True},
            "line_context": {"selected_number": "555-0000"},
            "trade_in_context": {"quote_value": 200},
            "new_device_context": {"selection": "Moto G"},
            "order_context": {},
        }
        ctx = run_after_tool("detect_intent", {"detected_intent": "change_line"}, "FinalPricing", ledger)
        assert ctx.state["fsm_state"] == "LineToUpgrade"
        assert ctx.state["ledger"]["line_context"] == {}
        assert ctx.state["ledger"]["trade_in_context"] == {}
        assert ctx.state["ledger"]["new_device_context"] == {}


class TestHappyPathEndToEnd:
    """Simulate a complete happy-path run through all states without LLM."""

    def test_full_flow_with_trade_in(self):
        state = fsm.initial_state
        ledger = {
            "account_context": {}, "line_context": {}, "trade_in_context": {},
            "new_device_context": {}, "order_context": {},
        }

        steps = [
            ("verify_auth",             {"is_authorized": True},                      "Auth"),
            ("check_standing",          {"standing": "GOOD"},                          "AccountStandingCheck"),
            ("set_line",                {"selected_number": "555-1234"},               "LineToUpgrade"),
            ("check_eligibility",       {"is_eligible": True},                         "CheckLineUpgradeEligibility"),
            ("set_trade_in_preference", {"wants_trade_in": True},                      "VerifyTradeIn"),
            ("record_condition",        {"trade_in_device": "iPhone 13", "final_condition": "Good"}, "DeviceTradeInChecks"),
            ("pricing",                 {"final_condition": "Good", "quote_value": 200}, "TradeInPricing"),
            ("select_device",           {"selection": "Pixel 9"},                      "NewUpgradeDeviceSelection"),
            ("pricing",                 {"selection": "Pixel 9", "price": 1000},       "NewUpgradeDevicePricing"),
            ("confirm_order",           {"user_confirmed": True},                      "FinalPricing"),
            ("submit_order",            {"order_id": "ORD-999888777", "error": False}, "ProcessOrder"),
        ]

        for tool_name, result, expected_before_state in steps:
            assert state == expected_before_state, f"Expected {expected_before_state}, got {state}"
            ctx = MockCallbackContext({"fsm_state": state, "ledger": ledger})
            after_tool(MockBaseTool(tool_name), {}, ctx, result)
            state = ctx.state["fsm_state"]
            ledger = ctx.state["ledger"]

        assert state == "EndSuccess"

    def test_full_flow_no_trade_in(self):
        state = fsm.initial_state
        ledger = {
            "account_context": {}, "line_context": {}, "trade_in_context": {},
            "new_device_context": {}, "order_context": {},
        }

        steps = [
            ("verify_auth",             {"is_authorized": True},          "Auth"),
            ("check_standing",          {"standing": "GOOD"},              "AccountStandingCheck"),
            ("set_line",                {"selected_number": "555-1234"},   "LineToUpgrade"),
            ("check_eligibility",       {"is_eligible": True},             "CheckLineUpgradeEligibility"),
            ("set_trade_in_preference", {"wants_trade_in": False},         "VerifyTradeIn"),
            # Jumps directly to NewUpgradeDeviceSelection (no trade-in path)
            ("select_device",           {"selection": "iPhone 16"},        "NewUpgradeDeviceSelection"),
            ("pricing",                 {"selection": "iPhone 16", "price": 1000}, "NewUpgradeDevicePricing"),
            ("confirm_order",           {"user_confirmed": True},          "FinalPricing"),
            ("submit_order",            {"order_id": "ORD-999888777", "error": False}, "ProcessOrder"),
        ]

        for tool_name, result, expected_before_state in steps:
            assert state == expected_before_state, f"Expected {expected_before_state}, got {state}"
            ctx = MockCallbackContext({"fsm_state": state, "ledger": ledger})
            after_tool(MockBaseTool(tool_name), {}, ctx, result)
            state = ctx.state["fsm_state"]
            ledger = ctx.state["ledger"]

        assert state == "EndSuccess"

    def test_error_path_unauthorized(self):
        ctx = run_after_tool("verify_auth", {"is_authorized": False}, "Auth")
        assert ctx.state["fsm_state"] == "EndUnauthorized"

    def test_error_path_bad_standing(self):
        state = "AccountStandingCheck"
        ledger = {"account_context": {}, "line_context": {}, "trade_in_context": {},
                  "new_device_context": {}, "order_context": {}}
        ctx = MockCallbackContext({"fsm_state": state, "ledger": ledger})
        after_tool(MockBaseTool("check_standing"), {}, ctx, {"standing": "DELINQUENT"})
        assert ctx.state["fsm_state"] == "EndBadStanding"
