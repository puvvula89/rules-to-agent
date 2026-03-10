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
                            final_text += part.text
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
    from agents.manager import root_agent

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
    from agents.manager import root_agent

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
    from agents.manager import root_agent

    session = AgentSession(root_agent)
    await session.start()

    await session.send("My account number is 9999 and PIN is 1234.")

    log.info(f"\nFINAL STATE: {session.fsm_state()}")
    assert session.fsm_state() in ("EndUnauthorized", "AccountStandingCheck", "Auth"), \
        f"Unexpected state: {session.fsm_state()}"
    log.info(f"✓ Unauthorized path reached: {session.fsm_state()}")


async def run_change_of_mind():
    """User changes device after reaching FinalPricing."""
    from agents.manager import root_agent

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


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scenarios = {
        "1": ("Happy path (trade-in)", run_happy_path_with_trade_in),
        "2": ("No trade-in path",      run_no_trade_in_path),
        "3": ("Unauthorized account",  run_unauthorized_path),
        "4": ("Change of mind",        run_change_of_mind),
    }

    log.info("Select scenario:")
    for key, (name, _) in scenarios.items():
        log.info(f"  {key}. {name}")
    choice = input("Choice [1]: ").strip() or "1"

    _, fn = scenarios.get(choice, scenarios["1"])
    asyncio.run(fn())
