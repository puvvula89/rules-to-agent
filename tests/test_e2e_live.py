"""
End-to-end live test: real Gemini LLM + real MCP server + real FSM.

Drives the agent through a full conversation programmatically,
exactly like the adk web UI would, and asserts FSM state at each turn.

Prerequisites:
  - MCP server running:  uvicorn mock_mcp_server.server:app --port 8080
  - Auth configured:     GOOGLE_API_KEY  OR  GOOGLE_GENAI_USE_VERTEXAI=true

Run:
  poetry run pytest tests/test_e2e_live.py -v -s
  poetry run python  tests/test_e2e_live.py        # standalone with full trace

Logs are written to: logs/flow-test.log (overwritten on each run)
"""

import asyncio
import sys
import os
import logging
import traceback
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ---------------------------------------------------------------------------
# Logging setup — overwrites logs/flow-test.log on every run
# ---------------------------------------------------------------------------

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "flow-test.log"

_fmt = logging.Formatter("%(message)s")

_file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
_file_handler.setFormatter(_fmt)

_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.setFormatter(_fmt)

# Use a dedicated logger with propagate=False so pytest's log-capture
# plugin doesn't swallow the records before they reach our file handler.
log = logging.getLogger("e2e_live")
log.setLevel(logging.INFO)
log.propagate = False
log.addHandler(_file_handler)
log.addHandler(_stdout_handler)

# Also route internal module loggers (manager.py, fsm.py) to the same file.
for _mod in ("agents.manager", "orchestrator.fsm"):
    _ml = logging.getLogger(_mod)
    _ml.setLevel(logging.DEBUG)
    _ml.propagate = False
    _ml.addHandler(_file_handler)
    _ml.addHandler(_stdout_handler)

from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


# ---------------------------------------------------------------------------
# Session driver
# ---------------------------------------------------------------------------

class AgentSession:
    """Wraps ADK Runner for multi-turn conversation testing."""

    def __init__(self, agent, app_name="telco_e2e_test"):
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            app_name=app_name,
            agent=agent,
            session_service=self.session_service,
        )
        self.app_name = app_name
        self.user_id = "test_user"
        self.session_id = None

    async def start(self):
        session = await self.session_service.create_session(
            app_name=self.app_name,
            user_id=self.user_id,
        )
        self.session_id = session.id
        log.info(f"\n{'='*60}")
        log.info(f"Session started: {self.session_id}")
        log.info(f"{'='*60}")

    async def send(self, user_message: str) -> str:
        """Send a user message and return the agent's text response."""
        log.info(f"\n[USER] {user_message}")

        message = types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)]
        )

        final_text = ""
        tool_calls = []

        try:
            async for event in self.runner.run_async(
                user_id=self.user_id,
                session_id=self.session_id,
                new_message=message,
            ):
                if getattr(event, "content", None) and getattr(event.content, "parts", None):
                    for part in event.content.parts:
                        if hasattr(part, "function_call") and part.function_call:
                            tool_calls.append(part.function_call.name)
                        if hasattr(part, "text") and part.text:
                            final_text = part.text  # keep only the last text (summarized response)
        except Exception:
            log.error(f"[ERROR] Exception during runner.run_async on turn: '{user_message}'")
            log.error(traceback.format_exc())
            raise

        if tool_calls:
            log.info(f"  [TOOLS] {' → '.join(tool_calls)}")

        fsm_state = self.fsm_state()
        log.info(f"  [FSM]   {fsm_state}")
        log.info(f"[AGENT] {final_text.strip()}")
        return final_text.strip()

    def _get_session(self):
        """Read the current session from InMemorySessionService."""
        return (
            self.session_service.sessions
            .get(self.app_name, {})
            .get(self.user_id, {})
            .get(self.session_id)
        )

    def fsm_state(self) -> str:
        session = self._get_session()
        return session.state.get("fsm_state", "Auth") if session else "unknown"

    def ledger(self) -> dict:
        session = self._get_session()
        return session.state.get("ledger", {}) if session else {}


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

async def run_happy_path_with_trade_in():
    """Full happy path: account 1234, trade in iPhone 13 Good, buy iPhone 16."""
    from agents.agent import root_agent

    session = AgentSession(root_agent)
    await session.start()

    # Turn 1: Auth + AccountStandingCheck auto-advance in one turn (slot-filling)
    # Agent calls verify_auth → fsm_advance → check_standing → fsm_advance without user input
    response = await session.send("Hi, I'd like to upgrade my phone. My account number is 1234 and my PIN is 5678.")
    assert session.fsm_state() in ("AccountStandingCheck", "LineToUpgrade"), \
        f"Expected AccountStandingCheck or LineToUpgrade, got {session.fsm_state()}"

    # Turn 2: Line selection
    response = await session.send("I want to upgrade the line 555-123-4567.")
    assert session.fsm_state() in ("LineToUpgrade", "CheckLineUpgradeEligibility", "VerifyTradeIn"), \
        f"Unexpected FSM state: {session.fsm_state()}"

    # Turn 3: Push through to VerifyTradeIn if needed
    if session.fsm_state() != "VerifyTradeIn":
        response = await session.send("Please check eligibility for that line.")

    # Turn 4: Trade-in yes
    response = await session.send("Yes, I want to trade in my current device.")
    assert session.fsm_state() in ("DeviceTradeInChecks", "VerifyTradeIn"), \
        f"Unexpected FSM state: {session.fsm_state()}"

    # Turn 5: Device condition
    response = await session.send("I'm trading in an iPhone 13 in Good condition.")
    assert session.fsm_state() in ("DeviceTradeInChecks", "TradeInPricing", "NewUpgradeDeviceSelection"), \
        f"Unexpected FSM state: {session.fsm_state()}"

    # Turn 6: New device selection
    response = await session.send("I'd like to get the iPhone 16.")
    assert session.fsm_state() in ("NewUpgradeDeviceSelection", "NewUpgradeDevicePricing", "FinalPricing"), \
        f"Unexpected FSM state: {session.fsm_state()}"

    # Turn 7: Confirm order (may need to push if still in pricing)
    response = await session.send("Yes, I confirm the order. Please proceed.")
    assert session.fsm_state() in ("FinalPricing", "ProcessOrder", "EndSuccess"), \
        f"Unexpected FSM state: {session.fsm_state()}"

    # Turn 8: Wrap up if needed
    if session.fsm_state() != "EndSuccess":
        response = await session.send("Yes please submit the order.")

    log.info(f"\n{'='*60}")
    log.info(f"FINAL FSM STATE: {session.fsm_state()}")
    log.info(f"LEDGER: {session.ledger()}")
    log.info(f"{'='*60}")

    assert session.fsm_state() == "EndSuccess", f"Expected EndSuccess, got {session.fsm_state()}"
    log.info("\n✓ Happy path with trade-in PASSED")


async def run_no_trade_in_path():
    """No trade-in path: skip straight to device selection."""
    from agents.agent import root_agent

    session = AgentSession(root_agent)
    await session.start()

    await session.send("Account 1234, PIN 0000.")
    await session.send("Upgrade line 555-999-0000.")
    await session.send("No, I don't want to trade in my old device.")

    assert session.fsm_state() in ("NewUpgradeDeviceSelection", "VerifyTradeIn"), \
        f"Expected to skip trade-in, got {session.fsm_state()}"

    await session.send("I want the Pixel 9.")
    await session.send("Yes, confirm the order.")

    if session.fsm_state() != "EndSuccess":
        await session.send("Submit it please.")

    log.info(f"\nFINAL STATE: {session.fsm_state()}")
    assert session.fsm_state() == "EndSuccess", f"Expected EndSuccess, got {session.fsm_state()}"
    log.info("✓ No trade-in path PASSED")


async def run_unauthorized_path():
    """Account 9999 should end at EndUnauthorized."""
    from agents.agent import root_agent

    session = AgentSession(root_agent)
    await session.start()

    await session.send("My account number is 9999 and PIN is 1234.")

    log.info(f"\nFINAL STATE: {session.fsm_state()}")
    assert session.fsm_state() in ("EndUnauthorized", "AccountStandingCheck", "Auth"), \
        f"Unexpected state: {session.fsm_state()}"
    log.info(f"✓ Unauthorized path reached: {session.fsm_state()}")


async def run_change_of_mind():
    """User changes device after reaching FinalPricing."""
    from agents.agent import root_agent

    session = AgentSession(root_agent)
    await session.start()

    await session.send("Account 1234, PIN 5678.")
    await session.send("Line 555-111-2222.")
    await session.send("No trade-in.")
    await session.send("I want the Pixel 9.")
    await session.send("Actually wait, I changed my mind. I want a different device.")

    state = session.fsm_state()
    log.info(f"\nAfter change of mind: {state}")
    assert state in ("NewUpgradeDeviceSelection", "NewUpgradeDevicePricing", "FinalPricing"), \
        f"Expected rewind, got {state}"
    log.info(f"✓ Change of mind handled: {state}")


async def run_line_provided_in_first_message():
    """
    Issue 1: User provides line number in the SAME message as account/PIN.

    After auth + standing are verified in that single turn, the agent MUST check
    eligibility for the provided line WITHOUT asking 'which line do you want to upgrade?'

    Failure mode being tested: agent advances Auth→AccountStanding→LineToUpgrade
    but then asks 'which line?' even though the user already said '555-111-2222'.
    """
    from agents.agent import root_agent

    session = AgentSession(root_agent)
    await session.start()

    # Turn 1: account, PIN, AND line all in one message
    response = await session.send(
        "Hi, I want to upgrade line 555-111-2222. My account number is 1234 and PIN is 5678."
    )

    final_state = session.fsm_state()
    ledger = session.ledger()
    log.info(f"  [CHECK] FSM after turn 1: {final_state}")
    log.info(f"  [CHECK] Ledger: {ledger}")
    log.info(f"  [CHECK] Agent response: {response[:200]}")

    # Primary assertion: FSM must have advanced past LineToUpgrade.
    # The agent had account+PIN+line — it should have run verify_auth → check_standing
    # → set_line → check_eligibility in one slot-filling turn.
    assert final_state not in ("Auth", "AccountStandingCheck", "LineToUpgrade"), (
        f"Agent should have advanced past LineToUpgrade using the provided line, "
        f"but got {final_state}. This means it stopped and asked for the line."
    )

    # Ledger assertion: line must be captured
    line_ctx = ledger.get("line_context", {})
    assert line_ctx.get("selected_number") is not None, (
        f"line_context.selected_number missing from ledger — agent never recorded the line. "
        f"Ledger: {ledger}"
    )
    assert "2222" in str(line_ctx.get("selected_number", "")), (
        f"Expected 555-111-2222 in ledger, got: {line_ctx.get('selected_number')}"
    )

    # Soft assertion: response should NOT be asking for the line
    resp_lower = response.lower()
    assert "which line" not in resp_lower and "what line" not in resp_lower, (
        f"Agent re-asked 'which line' even though 555-111-2222 was provided upfront: {response[:300]}"
    )

    log.info("\n✓ Line-provided-upfront scenario PASSED")


async def run_front_loaded_device_and_tradein_intent():
    """
    Issue 3: User states the new device AND trade-in intent BEFORE providing auth.

    Expected flow:
      Turn 1: 'I want to upgrade to iPhone 16. I also want to trade in my iPhone 13.'
              → FSM stays at Auth; agent greets and asks for account/PIN.
      Turn 2: 'Account 1234, PIN 5678.'
              → Auth+Standing done → FSM at LineToUpgrade.
              → Agent asks ONLY for line number (does NOT re-ask about device or trade-in).
      Turn 3: 'Line 555-777-8888.'
              → Agent checks eligibility → recognises trade-in intent from history
              → calls set_trade_in_preference(True) → advances to DeviceTradeInChecks
              → asks for condition ONLY (does NOT ask 'do you want to trade in?').

    Failure modes being tested:
      - Agent re-asks 'do you want to trade in?' (already answered in turn 1)
      - Agent re-asks 'which device are you upgrading to?' (already answered in turn 1)
      - Ledger never populated with trade_in_context.wants_trade_in = True
    """
    from agents.agent import root_agent

    session = AgentSession(root_agent)
    await session.start()

    # Turn 1: device + trade-in intent BEFORE auth
    response = await session.send(
        "Hi! I want to upgrade to an iPhone 16. I also want to trade in my current iPhone 13."
    )
    assert session.fsm_state() == "Auth", (
        f"Expected FSM=Auth (account/PIN not yet provided), got {session.fsm_state()}"
    )
    resp_lower = response.lower()
    assert "account" in resp_lower or "pin" in resp_lower, (
        f"Expected agent to ask for account/PIN in turn 1, got: {response[:200]}"
    )
    log.info(f"  [T1] FSM: {session.fsm_state()} | Response asks for account/PIN: OK")

    # Turn 2: account + PIN
    response = await session.send("Account 1234, PIN 5678.")
    assert session.fsm_state() in ("LineToUpgrade", "CheckLineUpgradeEligibility"), (
        f"Expected LineToUpgrade after auth+standing, got {session.fsm_state()}"
    )
    # Agent should ask for line — nothing else. It already knows the device and trade-in intent.
    resp_lower = response.lower()
    assert "line" in resp_lower, (
        f"Expected agent to ask for line number in turn 2, got: {response[:200]}"
    )
    # Soft check: it should NOT be asking about trade-in at this point
    if "trade" in resp_lower and "do you want" in resp_lower:
        log.info(f"  [T2 WARN] Agent re-asked about trade-in even though user already stated intent")
    log.info(f"  [T2] FSM: {session.fsm_state()} | Asks for line: OK")

    # Turn 3: provide line
    response = await session.send("Line 555-777-8888.")

    final_state = session.fsm_state()
    ledger = session.ledger()
    log.info(f"  [T3] FSM: {final_state}")
    log.info(f"  [T3] Ledger: {ledger}")
    log.info(f"  [T3] Agent response: {response[:300]}")

    # Primary: agent should have advanced past eligibility check into the trade-in path
    assert final_state not in ("Auth", "AccountStandingCheck", "LineToUpgrade", "CheckLineUpgradeEligibility"), (
        f"Expected FSM to advance into trade-in path after providing line, got {final_state}"
    )

    # Ledger: trade-in intent must have been recorded (from conversation, not from asking again)
    trade_ctx = ledger.get("trade_in_context", {})
    assert trade_ctx.get("wants_trade_in") == True, (
        f"Expected trade_in_context.wants_trade_in=True from conversation history, "
        f"got trade_ctx={trade_ctx}. Agent should have used the user's earlier statement."
    )

    # Ledger: new device should be recorded
    new_dev_ctx = ledger.get("new_device_context", {})
    if new_dev_ctx.get("selection"):
        assert "iphone 16" in str(new_dev_ctx["selection"]).lower(), (
            f"Expected iPhone 16 in new_device_context.selection, got: {new_dev_ctx['selection']}"
        )

    # Soft check: agent should NOT be re-asking 'do you want to trade in'
    resp_lower = response.lower()
    if "do you want to trade" in resp_lower or "would you like to trade" in resp_lower:
        log.info(
            f"  [T3 WARN] Agent re-asked trade-in intent even though user stated it upfront. "
            f"Response: {response[:200]}"
        )

    # If FSM is at DeviceTradeInChecks, agent should be asking about condition (not trade-in intent)
    if final_state == "DeviceTradeInChecks":
        assert "condition" in resp_lower or "shape" in resp_lower or "excellent" in resp_lower \
               or "good" in resp_lower or "poor" in resp_lower, (
            f"Expected agent to ask about device condition (not re-ask trade-in intent), "
            f"got: {response[:300]}"
        )

    log.info("\n✓ Front-loaded device + trade-in intent scenario PASSED")


async def run_auto_progression_no_nudge():
    """
    Issue 2: Agent must NOT pause and wait for a nudge mid-flow.

    Scenario: user provides BOTH trade-in device+condition AND new device name in a
    single message. The agent should:
      - call record_condition → fsm_advance (→ TradeInPricing)
      - call pricing (trade-in) → fsm_advance (→ NewUpgradeDeviceSelection)
      - recognise new device from the message → call select_device → fsm_advance (→ NewUpgradeDevicePricing)
      - call pricing (new device) → fsm_advance (→ FinalPricing)
    All in ONE turn, without producing 'Let me now get the pricing...' and stopping.

    Failure mode being tested: agent says 'Let me get the pricing for your new device'
    as a response and then waits for the user to say 'ok' before fetching the price.
    """
    from agents.agent import root_agent

    session = AgentSession(root_agent)
    await session.start()

    # Setup: get auth+standing+line out of the way
    await session.send("Account 1234, PIN 5678.")
    await session.send("I want to upgrade line 555-555-0001.")

    # Confirm at VerifyTradeIn before the critical turn
    state_before = session.fsm_state()
    assert state_before in ("VerifyTradeIn", "DeviceTradeInChecks", "CheckLineUpgradeEligibility"), (
        f"Expected to be at or near VerifyTradeIn before critical turn, got {state_before}"
    )

    # Critical turn: give ALL remaining device info in one message.
    # Agent should auto-advance all the way to FinalPricing without waiting for nudges.
    response = await session.send(
        "Yes, I want to trade in my iPhone 13 — it's in Good condition. "
        "I'd like to get the iPhone 16 as my new device."
    )

    final_state = session.fsm_state()
    ledger = session.ledger()
    log.info(f"  [CHECK] FSM after all-device-info turn: {final_state}")
    log.info(f"  [CHECK] Ledger: {ledger}")
    log.info(f"  [CHECK] Agent response: {response[:300]}")

    # Primary assertion: must have reached FinalPricing (or beyond) in a SINGLE turn.
    # If agent is still at TradeInPricing or NewUpgradeDeviceSelection, it paused mid-flow.
    assert final_state in ("FinalPricing", "ProcessOrder", "EndSuccess"), (
        f"Expected FinalPricing after providing full device info (no nudge), got {final_state}. "
        f"Agent stopped mid-flow and waited for a nudge."
    )

    # Ledger: trade-in quote must have been fetched (not just recorded)
    trade_ctx = ledger.get("trade_in_context", {})
    assert trade_ctx.get("quote_value") is not None, (
        f"Expected trade-in quote_value in ledger (auto-fetched via pricing tool), "
        f"got trade_ctx={trade_ctx}"
    )

    # Ledger: new device price must have been fetched
    new_dev_ctx = ledger.get("new_device_context", {})
    assert new_dev_ctx.get("price") is not None, (
        f"Expected new_device_context.price in ledger (auto-fetched via pricing tool), "
        f"got new_dev_ctx={new_dev_ctx}"
    )

    log.info("\n✓ Auto-progression (no nudge) scenario PASSED")


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------

import pytest

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Attach the call-phase report to the item so fixtures can read it."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


@pytest.fixture(autouse=True)
def log_test_boundaries(request):
    """Log test name + PASSED/FAILED + full error around every test."""
    log.info(f"\n{'='*60}")
    log.info(f"TEST: {request.node.name}")
    log.info(f"{'='*60}")
    yield
    rep = getattr(request.node, "rep_call", None)
    if rep is None:
        return
    if rep.passed:
        log.info(f"RESULT: {request.node.name} — PASSED")
    else:
        log.error(f"RESULT: {request.node.name} — FAILED")
        log.error(f"\n[FAILURE DETAIL]\n{rep.longrepr}")
    log.info(f"Log file: {LOG_FILE}")


@pytest.mark.asyncio
@pytest.mark.live
async def test_happy_path_with_trade_in():
    await run_happy_path_with_trade_in()


@pytest.mark.asyncio
@pytest.mark.live
async def test_no_trade_in_path():
    await run_no_trade_in_path()


@pytest.mark.asyncio
@pytest.mark.live
async def test_unauthorized_path():
    await run_unauthorized_path()


@pytest.mark.asyncio
@pytest.mark.live
async def test_change_of_mind():
    await run_change_of_mind()


@pytest.mark.asyncio
@pytest.mark.live
async def test_line_provided_in_first_message():
    """Issue 1: Line number given with account/PIN — agent must not re-ask for it."""
    await run_line_provided_in_first_message()


@pytest.mark.asyncio
@pytest.mark.live
async def test_front_loaded_device_and_tradein_intent():
    """Issue 3: Device + trade-in intent given before auth — agent must use from history."""
    await run_front_loaded_device_and_tradein_intent()


@pytest.mark.asyncio
@pytest.mark.live
async def test_auto_progression_no_nudge():
    """Issue 2: All device info in one turn — agent must auto-advance to FinalPricing."""
    await run_auto_progression_no_nudge()


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scenarios = {
        "1": ("Happy path (trade-in)",             run_happy_path_with_trade_in),
        "2": ("No trade-in path",                  run_no_trade_in_path),
        "3": ("Unauthorized account",              run_unauthorized_path),
        "4": ("Change of mind",                    run_change_of_mind),
        "5": ("Line provided in first message",    run_line_provided_in_first_message),
        "6": ("Front-loaded device + trade-in",    run_front_loaded_device_and_tradein_intent),
        "7": ("Auto-progression (no nudge)",       run_auto_progression_no_nudge),
    }

    log.info("Select scenario:")
    for key, (name, _) in scenarios.items():
        log.info(f"  {key}. {name}")
    choice = input("Choice [1]: ").strip() or "1"

    _, fn = scenarios.get(choice, scenarios["1"])
    asyncio.run(fn())
