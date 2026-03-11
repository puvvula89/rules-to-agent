from mcp.server.fastmcp import FastMCP

mcp = FastMCP("telco-mock-api", stateless_http=True)


@mcp.tool()
def verify_auth(account_number: str, pin: str) -> dict:
    """Verify a user's account with their account number and PIN. Require both to proceed."""
    return {"is_authorized": account_number == "1234"}


@mcp.tool()
def check_standing(account_number: str) -> dict:
    """Check the standing of the authenticated account."""
    standing = "GOOD" if account_number == "1234" else "DELINQUENT"
    return {"standing": standing}


@mcp.tool()
def check_eligibility(phone_number: str) -> dict:
    """Check if a specific phone line on the account is eligible for an upgrade."""
    return {"is_eligible": True}


@mcp.tool()
def set_line(phone_number: str) -> dict:
    """Set the phone number that the user wants to upgrade."""
    return {"selected_number": phone_number}


@mcp.tool()
def set_trade_in_preference(wants_trade_in: bool) -> dict:
    """Record whether the user wants to trade in their current device."""
    return {"wants_trade_in": wants_trade_in}


@mcp.tool()
def record_condition(device_model: str, condition: str) -> dict:
    """Record the physical condition of the trade-in device. condition must be Excellent, Good, or Poor."""
    return {"trade_in_device": device_model, "final_condition": condition}


@mcp.tool()
def pricing(device_model: str, condition: str = "N/A") -> dict:
    """Get trade-in value or new device price. Supply condition (Excellent/Good/Poor) for trade-ins."""
    model = device_model.lower()
    cond = condition.lower()

    if cond not in ("n/a", ""):
        # Trade-in quote
        if "iphone" in model:
            val = 400 if cond == "excellent" else 200
        elif "samsung" in model or "pixel" in model:
            val = 300 if cond == "excellent" else 150
        else:
            val = 50
        return {"final_condition": condition, "quote_value": val}
    else:
        # New device price
        if "iphone 16" in model or "pixel 9" in model or "s24" in model:
            price = 1000
        else:
            price = 800
        return {"selection": device_model, "price": price}


@mcp.tool()
def select_device(device_model: str) -> dict:
    """Select the new device the user wants to purchase."""
    return {"selection": device_model}


@mcp.tool()
def confirm_order() -> dict:
    """Confirm the user wants to proceed with the order."""
    return {"user_confirmed": True}


@mcp.tool()
def decline_order() -> dict:
    """Decline the order — user does not want to proceed."""
    return {"user_confirmed": False}


@mcp.tool()
def submit_order(account_number: str, phone_number: str, new_device: str, trade_in_device: str = "") -> dict:
    """Submit the final upgrade transaction."""
    return {"order_id": "ORD-999888777", "error": False}


@mcp.tool()
def detect_intent(intent: str) -> dict:
    """Signal a user change-of-mind. Pass the exact trigger name as returned by the workflow configuration."""
    return {"detected_intent": intent}


app = mcp.streamable_http_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
