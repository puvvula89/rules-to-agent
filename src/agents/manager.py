from google.adk import Agent
from google.adk.tools import McpToolset
from mcp import StdioServerParameters
import os
from typing import Optional, Dict, Any, List


from orchestrator.session import SessionManager
from orchestrator.fsm import WorkflowFSM
from tools.mcp_client import MCPClient
import json
from google.genai import Client
from google.adk.models import Gemini

class ManagerAgent:
    """
    Tier 1: The Orchestrator (Single-Agent ADK Architecture)
    Handles user interaction and slot filling, heavily guarded by ADK hooks
    to filter tools strictly based on the FSM current state.
    """
    def __init__(self, session_manager: SessionManager, fsm: WorkflowFSM):
        self.session_manager = session_manager
        self.fsm = fsm
        
        # Instantiate McpToolset using connection_params. 
        # ADK natively handles the AnyIO TaskGroup/ClientSession lifecycle internally.
        server_path = os.path.abspath("mock_mcp_server/server.py")
        self.mcp_toolset = McpToolset(
            connection_params=StdioServerParameters(
                command="python",
                args=[server_path]
            )
        )
        
        def before_model(*args, **kwargs):
            callback_context = kwargs.get("callback_context") or (args[0] if len(args) > 0 else None)
            llm_request = kwargs.get("llm_request") or (args[1] if len(args) > 1 else None)
            
            session_id = getattr(callback_context.session, "session_id", "default") if getattr(callback_context, "session", None) else "default"
            session = self.session_manager.get_session(session_id)
            current_state = session.current_state
            
            allowed_tools = []
            if current_state == "Auth": allowed_tools = ["verify_auth"]
            elif current_state == "AccountStandingCheck": allowed_tools = ["check_standing"]
            elif current_state == "LineToUpgrade": allowed_tools = ["set_line"]
            elif current_state == "CheckLineUpgradeEligibility": allowed_tools = ["check_eligibility"]
            elif current_state in ["TradeInPricing", "NewUpgradeDevicePricing"]: allowed_tools = ["pricing"]
            elif current_state == "ProcessOrder": allowed_tools = ["submit_order"]
            
            if llm_request.config and llm_request.config.tools:
                for t in llm_request.config.tools:
                    if getattr(t, "function_declarations", None):
                        t.function_declarations = [f for f in t.function_declarations if f.name in allowed_tools]

            objective = self.fsm.get_current_objective(session_id)
            dynamic_instruction = f'''CRITICAL RULES (FSM Guardrail):
Your CURRENT OBJECTIVE is strictly defined by the backend state machine:
"{objective}"

1. Do NOT decide what to do next. Only collect the data required by the tool provided to you.
2. If you have the data, call the explicit tool provided.
3. If the user changes their mind, call `global_trigger(intent)`.
'''         
            if llm_request.config:
                llm_request.config.system_instruction = dynamic_instruction
                
            # Returning None tells ADK to proceed with the Vertex call
            return None

        def after_tool(*args, **kwargs):
            tool = kwargs.get("tool") or (args[0] if len(args) > 0 else None)
            context = kwargs.get("callback_context") or kwargs.get("context") or (args[2] if len(args) > 2 else None)
            tool_result = kwargs.get("tool_result") or (args[3] if len(args) > 3 else None)
            
            session_id = getattr(context.session, "session_id", "default") if getattr(context, "session", None) else "default"
            session = self.session_manager.get_session(session_id)
            
            try:
                # The MCP tool result might be a list of TextContent objects
                if isinstance(tool_result, list) and len(tool_result) > 0:
                    first_item = tool_result[0]
                    if hasattr(first_item, "text"):
                        # Extract the text and parse that as JSON
                        result_data = json.loads(first_item.text)
                    else:
                        result_data = tool_result
                elif isinstance(tool_result, str):
                    result_data = json.loads(tool_result)
                else:
                    result_data = tool_result
            except Exception as e:
                print(f"  [Hook Warning] Could not parse tool result as JSON: {e}, type: {type(tool_result)}")
                return tool_result
                
            mapping = {
                "verify_auth": "account_context",
                "check_standing": "account_context",
                "set_line": "line_context",
                "check_eligibility": "line_context",
                "pricing": "trade_in_context" if isinstance(result_data, dict) and "quote_value" in result_data else "new_device_context",
                "submit_order": "order_context"
            }
            
            context_key = mapping.get(tool.name)
            if context_key and isinstance(result_data, dict):
                getattr(session.ledger, context_key).update(result_data)
                print(f"[Ledger] Updated {context_key}: {getattr(session.ledger, context_key)}")
                
            self.session_manager.save_session(session)
            old_state = session.current_state
            new_state, new_objective = self.fsm.evaluate(session_id)
            if old_state != new_state:
                print(f"[FSM] Advancing {old_state} -> {new_state}")
                
            return tool_result

        self.agent = Agent(
            name="TelcoManager",
            model=Gemini(
                name="gemini-2.5-pro",
                api_client=Client(
                    vertexai=True, 
                    project="tmeg-working-demos", 
                    location="us-central1"
                )
            ),
            instruction="You are a helpful Telco Customer Service AI.",
            tools=[self.mcp_toolset],
            before_model_callback=[before_model],
            after_tool_callback=[after_tool]
        )
                
    async def handle_turn(self, session_id: str, user_input: str) -> str:
        """Entrypoint for the CLI/API wrapper to talk to the ADK agent."""
        from google.adk import Runner
        from google.genai import types
        from google.adk.sessions import InMemorySessionService
        
        runner = Runner(
            app_name="telco_poc",
            agent=self.agent, 
            session_service=InMemorySessionService(),
            auto_create_session=True
        )
        message = types.Content(role="user", parts=[types.Part.from_text(text=user_input)])
        
        final_text = ""
        # Runner manages the MCP connection cleanly across turns
        async for event in runner.run_async(user_id="user_1", session_id=session_id, new_message=message):
             if getattr(event, "content", None) and getattr(event.content, "parts", None):
                 for part in event.content.parts:
                     if hasattr(part, "text") and part.text:
                          final_text += part.text
                          
        return final_text
