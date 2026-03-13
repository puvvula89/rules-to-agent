import os
import pytest
from agents.orchestrator.fsm import WorkflowFSM

YAML_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "phone_upgrade.yaml"))


@pytest.fixture
def fsm():
    return WorkflowFSM(YAML_PATH)


def _ledger(**kwargs):
    base = {
        "account_context": {},
        "line_context": {},
        "trade_in_context": {},
        "new_device_context": {},
        "order_context": {},
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_auth_authorized_advances(fsm):
    ledger = _ledger(account_context={"is_authorized": True})
    assert fsm.evaluate("Auth", ledger) == "AccountStandingCheck"


def test_auth_unauthorized_goes_to_end(fsm):
    ledger = _ledger(account_context={"is_authorized": False})
    assert fsm.evaluate("Auth", ledger) == "EndUnauthorized"


def test_auth_no_data_stays(fsm):
    ledger = _ledger()
    assert fsm.evaluate("Auth", ledger) == "Auth"


def test_good_standing_advances(fsm):
    ledger = _ledger(account_context={"standing": "GOOD"})
    assert fsm.evaluate("AccountStandingCheck", ledger) == "LineToUpgrade"


def test_bad_standing_goes_to_end(fsm):
    ledger = _ledger(account_context={"standing": "DELINQUENT"})
    assert fsm.evaluate("AccountStandingCheck", ledger) == "EndBadStanding"


def test_standing_not_set_stays(fsm):
    ledger = _ledger()
    assert fsm.evaluate("AccountStandingCheck", ledger) == "AccountStandingCheck"


def test_line_set_advances(fsm):
    ledger = _ledger(line_context={"selected_number": "555-1234"})
    assert fsm.evaluate("LineToUpgrade", ledger) == "CheckLineUpgradeEligibility"


def test_eligible_advances(fsm):
    ledger = _ledger(line_context={"is_eligible": True})
    assert fsm.evaluate("CheckLineUpgradeEligibility", ledger) == "VerifyTradeIn"


def test_not_eligible_goes_to_end(fsm):
    ledger = _ledger(line_context={"is_eligible": False})
    assert fsm.evaluate("CheckLineUpgradeEligibility", ledger) == "EndNotEligible"


def test_wants_trade_in_advances(fsm):
    ledger = _ledger(trade_in_context={"wants_trade_in": True})
    assert fsm.evaluate("VerifyTradeIn", ledger) == "DeviceTradeInChecks"


def test_no_trade_in_skips_to_device_selection(fsm):
    ledger = _ledger(trade_in_context={"wants_trade_in": False})
    assert fsm.evaluate("VerifyTradeIn", ledger) == "NewUpgradeDeviceSelection"


def test_final_condition_advances(fsm):
    ledger = _ledger(trade_in_context={"final_condition": "Good"})
    assert fsm.evaluate("DeviceTradeInChecks", ledger) == "TradeInPricing"


def test_trade_in_quote_advances(fsm):
    ledger = _ledger(trade_in_context={"quote_value": 200})
    assert fsm.evaluate("TradeInPricing", ledger) == "NewUpgradeDeviceSelection"


def test_device_selected_advances(fsm):
    ledger = _ledger(new_device_context={"selection": "iPhone 16"})
    assert fsm.evaluate("NewUpgradeDeviceSelection", ledger) == "NewUpgradeDevicePricing"


def test_device_priced_advances(fsm):
    ledger = _ledger(new_device_context={"price": 1000})
    assert fsm.evaluate("NewUpgradeDevicePricing", ledger) == "CalculateFinalPrice"


def test_final_price_calculated_advances(fsm):
    ledger = _ledger(order_context={"final_price": 800})
    assert fsm.evaluate("CalculateFinalPrice", ledger) == "FinalPricing"


def test_confirmed_advances(fsm):
    ledger = _ledger(order_context={"user_confirmed": True})
    assert fsm.evaluate("FinalPricing", ledger) == "ProcessOrder"


def test_order_submitted_advances(fsm):
    ledger = _ledger(order_context={"order_id": "ORD-999"})
    assert fsm.evaluate("ProcessOrder", ledger) == "EndSuccess"


def test_terminal_state_stays(fsm):
    ledger = _ledger()
    for terminal in ["EndSuccess", "EndUnauthorized", "EndBadStanding", "EndNotEligible", "EndOrderFailed"]:
        assert fsm.evaluate(terminal, ledger) == terminal


# ---------------------------------------------------------------------------
# Global intent transitions
# ---------------------------------------------------------------------------

def test_get_global_intents_loaded_from_yaml(fsm):
    intents = fsm.get_global_intents()
    assert len(intents) == 3
    triggers = [g['trigger'] for g in intents]
    assert 'intent_change_line' in triggers
    assert 'intent_change_trade_in_device' in triggers
    assert 'intent_change_new_device' in triggers
    for g in intents:
        assert g.get('description'), "Each global intent must have a non-empty description"


def test_change_line_intent_resets_ledger(fsm):
    ledger = _ledger(
        line_context={"selected_number": "555-0000"},
        trade_in_context={"wants_trade_in": True},
        new_device_context={"selection": "Pixel 9"},
        order_context={"user_confirmed": True},
    )
    new_state = fsm.fire_intent("FinalPricing", "intent_change_line", ledger)
    assert new_state == "LineToUpgrade"
    assert ledger["line_context"] == {}
    assert ledger["trade_in_context"] == {}
    assert ledger["new_device_context"] == {}
    assert ledger["order_context"] == {}


def test_change_new_device_intent_resets_device(fsm):
    ledger = _ledger(
        trade_in_context={"quote_value": 200},
        new_device_context={"selection": "Pixel 9", "price": 800},
        order_context={"user_confirmed": True},
    )
    new_state = fsm.fire_intent("FinalPricing", "intent_change_new_device", ledger)
    assert new_state == "NewUpgradeDeviceSelection"
    assert ledger["new_device_context"] == {}
    assert ledger["order_context"] == {}
    assert ledger["trade_in_context"] == {"quote_value": 200}  # preserved
