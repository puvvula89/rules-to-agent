import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, CallToolResult
import json

app = Server("telco-mock-api")

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available mock telco tools."""
    return [
        Tool(
            name="verify_auth",
            description="Verify a user's account with their account number and PIN. Require both to proceed.",
            inputSchema={
                "type": "object",
                "properties": {
                    "account_number": {"type": "string"},
                    "pin": {"type": "string"}
                },
                "required": ["account_number", "pin"]
            }
        ),
        Tool(
            name="check_standing",
            description="Check the standing of the authenticated account.",
            inputSchema={
                "type": "object",
                "properties": {
                    "account_number": {"type": "string"}
                },
                "required": ["account_number"]
            }
        ),
        Tool(
            name="check_eligibility",
            description="Check if a specific phone line on the account is eligible for an upgrade. You MUST provide the phone_number parameter.",
            inputSchema={
                "type": "object",
                "properties": {
                    "phone_number": {"type": "string"}
                },
                "required": ["phone_number"]
            }
        ),
        Tool(
            name="set_line",
            description="Set the phone number that the user wants to upgrade.",
            inputSchema={
                "type": "object",
                "properties": {
                    "phone_number": {"type": "string"}
                },
                "required": ["phone_number"]
            }
        ),
        Tool(
            name="pricing",
            description="Get the price of a device, either for trade-in value or purchasing a new device. Supply 'condition' for trade-ins.",
            inputSchema={
                "type": "object",
                "properties": {
                    "device_model": {"type": "string", "description": "e.g., iPhone 13, Pixel 9"},
                    "condition": {"type": "string", "description": "Excellent, Good, or Poor. Only required for trade-ins.", "enum": ["Excellent", "Good", "Poor", "N/A"]}
                },
                "required": ["device_model"]
            }
        ),
        Tool(
            name="submit_order",
            description="Submit the final transaction.",
            inputSchema={
                "type": "object",
                "properties": {
                    "account_number": {"type": "string"},
                    "phone_number": {"type": "string"},
                    "new_device": {"type": "string"},
                    "trade_in_device": {"type": "string"},
                },
                "required": ["account_number", "phone_number", "new_device"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool execution requests."""
    
    if name == "verify_auth":
        # Mock logic: any pin is valid for '1234', else False
        is_valid = arguments.get("account_number") == "1234"
        return [TextContent(type="text", text=json.dumps({"is_authorized": is_valid}))]
        
    elif name == "check_standing":
        # Mock logic: Account 1234 is GOOD, all others DELINQUENT
        standing = "GOOD" if arguments.get("account_number") == "1234" else "DELINQUENT"
        return [TextContent(type="text", text=json.dumps({"standing": standing}))]
        
    elif name == "check_eligibility":
        # Let's just make everything eligible for POC ease
        return [TextContent(type="text", text=json.dumps({"is_eligible": True}))]
        
    elif name == "set_line":
        phone = arguments.get("phone_number", "")
        return [TextContent(type="text", text=json.dumps({"selected_number": phone}))]
        
    elif name == "pricing":
        model = arguments.get("device_model", "").lower()
        condition = arguments.get("condition", "N/A").lower()
        
        # Determine if this is a quote or a new price based on if condition was provided
        if condition != "n/a" and condition != "":
            # Trade in quote
            if "iphone" in model:
                val = 400 if condition == "excellent" else 200
            elif "samsung" in model or "pixel" in model:
                val = 300 if condition == "excellent" else 150
            else:
                val = 50
            return [TextContent(type="text", text=json.dumps({"final_condition": condition, "quote_value": val}))]
        else:
            # New device price
            if "iphone 16" in model or "pixel 9" in model or "s24" in model:
                price = 1000
            else:
                price = 800
            return [TextContent(type="text", text=json.dumps({"selection": model, "price": price}))]
            
    elif name == "submit_order":
        return [TextContent(type="text", text=json.dumps({"order_id": "ORD-999888777", "error": False}))]
        
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
