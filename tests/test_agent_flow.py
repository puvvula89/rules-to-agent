"""
Integration tests for the full agent flow.

Two layers:
  1. MCP server HTTP smoke tests — verifies all 11 MCP tools respond correctly
  2. Callback pipeline simulation — exercises before_model + after_model + FSM
     with mock ADK objects (no real LLM call needed)

Architecture:
  - detect_intent: ADK FunctionTool in agent.py — handles FSM rewind directly
  - after_model: parses ```json...``` block from LLM text, merges into ledger, advances FSM
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


class MockPart:
    def __init__(self, text=None, function_call=None):
        self.text = text
        self.function_call = function_call


class MockContent:
    def __init__(self, parts):
        self.parts = parts


class MockLlmResponse:
    def __init__(self, text=None, has_function_call=False):
        parts = []
        if text:
            parts.append(MockPart(text=text))
        if has_function_call:
            parts.append(MockPart(function_call=object()))
        self.content = MockContent(parts)


def _empty_ledger():
    return {
        "account_context": {},
        "line_context": {},
        "trade_in_context": {},
        "new_device_context": {},
        "order_context": {},
    }


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

# ---------------------------------------------------------------------------
# Callback pipeline simulation (no LLM)
# ---------------------------------------------------------------------------

from agents.agent import before_model, after_model, fsm_advance, detect_intent, fsm

ALL_TOOL_NAMES = [
    "verify_auth", "check_standing", "set_line", "check_eligibility",
    "set_trade_in_preference", "record_condition", "pricing",
    "select_device", "confirm_order", "decline_order", "submit_order",
    "fsm_advance", "detect_intent",
]


def run_before_model(state: str, state_dict: dict = None) -> MockLlmRequest:
    """Run before_model callback and return the mutated request."""
    ctx = MockCallbackContext({"fsm_state": state, **(state_dict or {})})
    req = MockLlmRequest(ALL_TOOL_NAMES)
    before_model(ctx, req)
    return req


def visible_tools(req: MockLlmRequest) -> list[str]:
    return [f.name for f in req.config.tools[0].function_declarations]


def run_detect_intent(intent: str, state: str, ledger: dict = None) -> MockCallbackContext:
    """Run detect_intent ADK tool and return the updated context."""
    ctx = MockCallbackContext({"fsm_state": state, "ledger": ledger or _empty_ledger()})
    detect_intent(intent, ctx)
    return ctx


def run_after_model(json_data: dict, state: str, ledger: dict = None) -> MockCallbackContext:
    """Run after_model callback with a response containing a JSON block."""
    json_str = json.dumps(json_data)
    text = f"Processing complete.\n```json\n{json_str}\n```"
    ctx = MockCallbackContext({"fsm_state": state, "ledger": ledger or _empty_ledger()})
    response = MockLlmResponse(text=text)
    after_model(ctx, response)
    return ctx


class TestBeforeModelInstruction:
    """LLM sees all tools; before_model only injects the dynamic system instruction."""

    def test_all_tools_remain_visible(self):
        # No filtering — all tools passed through unchanged
        req = run_before_model("Auth")
        assert visible_tools(req) == ALL_TOOL_NAMES

    def test_system_instruction_contains_objective(self):
        req = run_before_model("Auth")
        assert "Verify the user is an authorized" in req.config.system_instruction

    def test_system_instruction_references_detect_intent(self):
        req = run_before_model("LineToUpgrade")
        assert "detect_intent" in req.config.system_instruction

    def test_system_instruction_contains_extract_variables(self):
        # New prompt embeds extract_variables as a fsm_advance example (nested JSON format)
        req = run_before_model("Auth")
        assert "is_authorized" in req.config.system_instruction

    def test_system_instruction_contains_fsm_advance_instruction(self):
        # New prompt tells LLM to call fsm_advance after each MCP tool
        req = run_before_model("AccountStandingCheck")
        assert "fsm_advance" in req.config.system_instruction


class TestAfterModelLedgerAndFSM:
    """after_model parses JSON block, merges into ledger, and advances FSM."""

    def test_auth_authorized_advances(self):
        ctx = run_after_model({"account_context": {"is_authorized": True}}, "Auth")
        assert ctx.state["ledger"]["account_context"]["is_authorized"] is True
        assert ctx.state["fsm_state"] == "AccountStandingCheck"

    def test_auth_unauthorized_goes_to_end(self):
        ctx = run_after_model({"account_context": {"is_authorized": False}}, "Auth")
        assert ctx.state["fsm_state"] == "EndUnauthorized"

    def test_auth_no_data_stays(self):
        # No JSON block → no advancement
        ctx = MockCallbackContext({"fsm_state": "Auth", "ledger": _empty_ledger()})
        after_model(ctx, MockLlmResponse(text="Please provide your credentials."))
        assert ctx.state["fsm_state"] == "Auth"

    def test_good_standing_advances(self):
        ctx = run_after_model({"account_context": {"standing": "GOOD"}}, "AccountStandingCheck")
        assert ctx.state["fsm_state"] == "LineToUpgrade"

    def test_bad_standing_goes_to_end(self):
        ctx = run_after_model({"account_context": {"standing": "DELINQUENT"}}, "AccountStandingCheck")
        assert ctx.state["fsm_state"] == "EndBadStanding"

    def test_line_set_advances(self):
        ctx = run_after_model({"line_context": {"selected_number": "555-1234"}}, "LineToUpgrade")
        assert ctx.state["ledger"]["line_context"]["selected_number"] == "555-1234"
        assert ctx.state["fsm_state"] == "CheckLineUpgradeEligibility"

    def test_eligible_advances(self):
        ctx = run_after_model({"line_context": {"is_eligible": True}}, "CheckLineUpgradeEligibility")
        assert ctx.state["fsm_state"] == "VerifyTradeIn"

    def test_not_eligible_goes_to_end(self):
        ctx = run_after_model({"line_context": {"is_eligible": False}}, "CheckLineUpgradeEligibility")
        assert ctx.state["fsm_state"] == "EndNotEligible"

    def test_wants_trade_in_advances(self):
        ctx = run_after_model({"trade_in_context": {"wants_trade_in": True}}, "VerifyTradeIn")
        assert ctx.state["fsm_state"] == "DeviceTradeInChecks"

    def test_no_trade_in_skips_to_device_selection(self):
        ctx = run_after_model({"trade_in_context": {"wants_trade_in": False}}, "VerifyTradeIn")
        assert ctx.state["fsm_state"] == "NewUpgradeDeviceSelection"

    def test_final_condition_advances(self):
        ctx = run_after_model({"trade_in_context": {"final_condition": "Good"}}, "DeviceTradeInChecks")
        assert ctx.state["fsm_state"] == "TradeInPricing"

    def test_trade_in_quote_advances(self):
        ctx = run_after_model({"trade_in_context": {"quote_value": 200}}, "TradeInPricing")
        assert ctx.state["fsm_state"] == "NewUpgradeDeviceSelection"

    def test_device_selected_advances(self):
        ctx = run_after_model({"new_device_context": {"selection": "iPhone 16"}}, "NewUpgradeDeviceSelection")
        assert ctx.state["ledger"]["new_device_context"]["selection"] == "iPhone 16"
        assert ctx.state["fsm_state"] == "NewUpgradeDevicePricing"

    def test_device_priced_advances(self):
        ctx = run_after_model({"new_device_context": {"price": 1000}}, "NewUpgradeDevicePricing")
        assert ctx.state["fsm_state"] == "FinalPricing"

    def test_confirmed_advances(self):
        ctx = run_after_model({"order_context": {"user_confirmed": True}}, "FinalPricing")
        assert ctx.state["fsm_state"] == "ProcessOrder"

    def test_declined_stays_in_final_pricing(self):
        ctx = run_after_model({"order_context": {"user_confirmed": False}}, "FinalPricing")
        assert ctx.state["fsm_state"] == "FinalPricing"

    def test_order_submitted_advances(self):
        ctx = run_after_model({"order_context": {"order_id": "ORD-999888777"}}, "ProcessOrder")
        assert ctx.state["fsm_state"] == "EndSuccess"

    def test_function_call_response_skipped(self):
        """after_model should not advance FSM when response contains a function call."""
        ctx = MockCallbackContext({"fsm_state": "Auth", "ledger": _empty_ledger()})
        response = MockLlmResponse(
            text='```json\n{"account_context": {"is_authorized": true}}\n```',
            has_function_call=True,
        )
        after_model(ctx, response)
        assert ctx.state["fsm_state"] == "Auth"  # not advanced

    def test_deep_merge_preserves_existing_ledger(self):
        """after_model merges JSON into existing ledger without overwriting unrelated keys."""
        ledger = _empty_ledger()
        ledger["account_context"] = {"is_authorized": True, "standing": "GOOD"}
        ctx = run_after_model({"line_context": {"selected_number": "555-0000"}}, "LineToUpgrade", ledger)
        # Previous account_context preserved
        assert ctx.state["ledger"]["account_context"]["is_authorized"] is True
        assert ctx.state["ledger"]["account_context"]["standing"] == "GOOD"
        # New field added
        assert ctx.state["ledger"]["line_context"]["selected_number"] == "555-0000"


class TestDetectIntentTool:
    """detect_intent is now an ADK FunctionTool — fires FSM rewind directly."""

    def test_change_new_device_rewinds_and_wipes(self):
        ledger = {
            "account_context": {"is_authorized": True},
            "line_context": {"selected_number": "555-0000"},
            "trade_in_context": {"quote_value": 200},
            "new_device_context": {"selection": "Moto G", "price": 800},
            "order_context": {"user_confirmed": True},
        }
        ctx = run_detect_intent("intent_change_new_device", "FinalPricing", ledger)
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
        ctx = run_detect_intent("intent_change_line", "FinalPricing", ledger)
        assert ctx.state["fsm_state"] == "LineToUpgrade"
        assert ctx.state["ledger"]["line_context"] == {}
        assert ctx.state["ledger"]["trade_in_context"] == {}
        assert ctx.state["ledger"]["new_device_context"] == {}


def run_fsm_advance(data: dict, state: str, ledger: dict = None) -> MockCallbackContext:
    """Run fsm_advance tool and return the updated context."""
    ctx = MockCallbackContext({"fsm_state": state, "ledger": ledger or _empty_ledger()})
    fsm_advance(data, ctx)
    return ctx


class TestFsmAdvanceTool:
    """fsm_advance tool: updates ledger, normalises booleans, advances FSM, returns next objective."""

    def test_auth_authorized_advances(self):
        ctx = run_fsm_advance({"account_context": {"is_authorized": True}}, "Auth")
        assert ctx.state["fsm_state"] == "AccountStandingCheck"
        assert ctx.state["ledger"]["account_context"]["is_authorized"] is True

    def test_auth_string_true_normalized_and_advances(self):
        # LLM sometimes returns "true" as string — must be normalized to bool
        ctx = run_fsm_advance({"account_context": {"is_authorized": "true"}}, "Auth")
        assert ctx.state["ledger"]["account_context"]["is_authorized"] is True
        assert ctx.state["fsm_state"] == "AccountStandingCheck"

    def test_auth_string_false_normalized(self):
        ctx = run_fsm_advance({"account_context": {"is_authorized": "false"}}, "Auth")
        assert ctx.state["ledger"]["account_context"]["is_authorized"] is False
        assert ctx.state["fsm_state"] == "EndUnauthorized"

    def test_auth_unauthorized_goes_to_end(self):
        ctx = run_fsm_advance({"account_context": {"is_authorized": False}}, "Auth")
        assert ctx.state["fsm_state"] == "EndUnauthorized"

    def test_good_standing_advances(self):
        ctx = run_fsm_advance({"account_context": {"standing": "GOOD"}}, "AccountStandingCheck")
        assert ctx.state["fsm_state"] == "LineToUpgrade"

    def test_bad_standing_goes_to_end(self):
        ctx = run_fsm_advance({"account_context": {"standing": "DELINQUENT"}}, "AccountStandingCheck")
        assert ctx.state["fsm_state"] == "EndBadStanding"

    def test_line_set_advances(self):
        ctx = run_fsm_advance({"line_context": {"selected_number": "555-1234"}}, "LineToUpgrade")
        assert ctx.state["fsm_state"] == "CheckLineUpgradeEligibility"

    def test_eligible_advances(self):
        ctx = run_fsm_advance({"line_context": {"is_eligible": True}}, "CheckLineUpgradeEligibility")
        assert ctx.state["fsm_state"] == "VerifyTradeIn"

    def test_not_eligible_goes_to_end(self):
        ctx = run_fsm_advance({"line_context": {"is_eligible": False}}, "CheckLineUpgradeEligibility")
        assert ctx.state["fsm_state"] == "EndNotEligible"

    def test_wants_trade_in_advances(self):
        ctx = run_fsm_advance({"trade_in_context": {"wants_trade_in": True}}, "VerifyTradeIn")
        assert ctx.state["fsm_state"] == "DeviceTradeInChecks"

    def test_no_trade_in_skips_to_device_selection(self):
        ctx = run_fsm_advance({"trade_in_context": {"wants_trade_in": False}}, "VerifyTradeIn")
        assert ctx.state["fsm_state"] == "NewUpgradeDeviceSelection"

    def test_returns_next_objective(self):
        ctx = MockCallbackContext({"fsm_state": "Auth", "ledger": _empty_ledger()})
        result = fsm_advance({"account_context": {"is_authorized": True}}, ctx)
        assert result["workflow_advanced_to"] == "AccountStandingCheck"
        assert "standing" in result["next_objective"].lower()
        assert "account_context.standing" in result["fields_to_collect"]

    def test_deep_merge_preserves_existing_ledger(self):
        ledger = _empty_ledger()
        ledger["account_context"] = {"is_authorized": True, "standing": "GOOD"}
        ctx = run_fsm_advance({"line_context": {"selected_number": "555-0000"}}, "LineToUpgrade", ledger)
        assert ctx.state["ledger"]["account_context"]["is_authorized"] is True
        assert ctx.state["ledger"]["line_context"]["selected_number"] == "555-0000"

    def test_slot_fill_two_steps_in_one_call(self):
        """If user provides data for two states at once, fsm_advance advances one step.
        Caller (LLM) must call fsm_advance again for each subsequent step."""
        ledger = _empty_ledger()
        ledger["account_context"]["is_authorized"] = True  # Auth already done

        ctx = run_fsm_advance({"account_context": {"standing": "GOOD"}}, "AccountStandingCheck", ledger)
        assert ctx.state["fsm_state"] == "LineToUpgrade"

    def test_full_happy_path_via_fsm_advance(self):
        state = fsm.initial_state
        ledger = _empty_ledger()

        steps = [
            ("Auth",                        {"account_context": {"is_authorized": True}},         "AccountStandingCheck"),
            ("AccountStandingCheck",        {"account_context": {"standing": "GOOD"}},            "LineToUpgrade"),
            ("LineToUpgrade",               {"line_context": {"selected_number": "555-1234"}},    "CheckLineUpgradeEligibility"),
            ("CheckLineUpgradeEligibility", {"line_context": {"is_eligible": True}},              "VerifyTradeIn"),
            ("VerifyTradeIn",               {"trade_in_context": {"wants_trade_in": True}},       "DeviceTradeInChecks"),
            ("DeviceTradeInChecks",         {"trade_in_context": {"final_condition": "Good"}},    "TradeInPricing"),
            ("TradeInPricing",              {"trade_in_context": {"quote_value": 200}},           "NewUpgradeDeviceSelection"),
            ("NewUpgradeDeviceSelection",   {"new_device_context": {"selection": "iPhone 16"}},   "NewUpgradeDevicePricing"),
            ("NewUpgradeDevicePricing",     {"new_device_context": {"price": 1000}},              "FinalPricing"),
            ("FinalPricing",                {"order_context": {"user_confirmed": True}},           "ProcessOrder"),
            ("ProcessOrder",                {"order_context": {"order_id": "ORD-999888777"}},     "EndSuccess"),
        ]

        for expected_before, data, expected_after in steps:
            assert state == expected_before, f"Expected {expected_before}, got {state}"
            ctx = run_fsm_advance(data, state, ledger)
            state = ctx.state["fsm_state"]
            ledger = ctx.state["ledger"]
            assert state == expected_after, f"Expected {expected_after}, got {state}"

        assert state == "EndSuccess"


class TestHappyPathEndToEnd:
    """Simulate a complete happy-path run through all states using after_model."""

    def test_full_flow_with_trade_in(self):
        state = fsm.initial_state
        ledger = _empty_ledger()

        steps = [
            ("Auth",                       {"account_context": {"is_authorized": True}},          "AccountStandingCheck"),
            ("AccountStandingCheck",       {"account_context": {"standing": "GOOD"}},             "LineToUpgrade"),
            ("LineToUpgrade",              {"line_context": {"selected_number": "555-1234"}},     "CheckLineUpgradeEligibility"),
            ("CheckLineUpgradeEligibility",{"line_context": {"is_eligible": True}},               "VerifyTradeIn"),
            ("VerifyTradeIn",              {"trade_in_context": {"wants_trade_in": True}},        "DeviceTradeInChecks"),
            ("DeviceTradeInChecks",        {"trade_in_context": {"final_condition": "Good"}},     "TradeInPricing"),
            ("TradeInPricing",             {"trade_in_context": {"quote_value": 200}},            "NewUpgradeDeviceSelection"),
            ("NewUpgradeDeviceSelection",  {"new_device_context": {"selection": "iPhone 16"}},   "NewUpgradeDevicePricing"),
            ("NewUpgradeDevicePricing",    {"new_device_context": {"price": 1000}},               "FinalPricing"),
            ("FinalPricing",               {"order_context": {"user_confirmed": True}},           "ProcessOrder"),
            ("ProcessOrder",               {"order_context": {"order_id": "ORD-999888777"}},      "EndSuccess"),
        ]

        for expected_before_state, json_data, expected_after_state in steps:
            assert state == expected_before_state, f"Expected {expected_before_state}, got {state}"
            ctx = run_after_model(json_data, state, ledger)
            state = ctx.state["fsm_state"]
            ledger = ctx.state["ledger"]

        assert state == "EndSuccess"

    def test_full_flow_no_trade_in(self):
        state = fsm.initial_state
        ledger = _empty_ledger()

        steps = [
            ("Auth",                       {"account_context": {"is_authorized": True}},         "AccountStandingCheck"),
            ("AccountStandingCheck",       {"account_context": {"standing": "GOOD"}},            "LineToUpgrade"),
            ("LineToUpgrade",              {"line_context": {"selected_number": "555-1234"}},    "CheckLineUpgradeEligibility"),
            ("CheckLineUpgradeEligibility",{"line_context": {"is_eligible": True}},              "VerifyTradeIn"),
            ("VerifyTradeIn",              {"trade_in_context": {"wants_trade_in": False}},      "NewUpgradeDeviceSelection"),
            ("NewUpgradeDeviceSelection",  {"new_device_context": {"selection": "iPhone 16"}},  "NewUpgradeDevicePricing"),
            ("NewUpgradeDevicePricing",    {"new_device_context": {"price": 1000}},              "FinalPricing"),
            ("FinalPricing",               {"order_context": {"user_confirmed": True}},          "ProcessOrder"),
            ("ProcessOrder",               {"order_context": {"order_id": "ORD-999888777"}},     "EndSuccess"),
        ]

        for expected_before_state, json_data, expected_after_state in steps:
            assert state == expected_before_state, f"Expected {expected_before_state}, got {state}"
            ctx = run_after_model(json_data, state, ledger)
            state = ctx.state["fsm_state"]
            ledger = ctx.state["ledger"]

        assert state == "EndSuccess"

    def test_error_path_unauthorized(self):
        ctx = run_after_model({"account_context": {"is_authorized": False}}, "Auth")
        assert ctx.state["fsm_state"] == "EndUnauthorized"

    def test_error_path_bad_standing(self):
        ctx = run_after_model({"account_context": {"standing": "DELINQUENT"}}, "AccountStandingCheck")
        assert ctx.state["fsm_state"] == "EndBadStanding"
