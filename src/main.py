import asyncio
import os

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
os.environ["GOOGLE_CLOUD_PROJECT"] = "tmeg-working-demos"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"
import sys

from orchestrator.session import SessionManager
from orchestrator.fsm import WorkflowFSM
from agents.manager import ManagerAgent

async def main():
    print("=========================================")
    print("  🚀 Agentic Rules Engine (Telco POC) 🚀  ")
    print("=========================================\n")
    
    # 1. Initialize Core Components
    session_manager = SessionManager()
    
    # Ensure config path is absolute for safety
    yaml_path = os.path.abspath("config/phone_upgrade.yaml")
    if not os.path.exists(yaml_path):
        print(f"Error: Could not find YAML config at {yaml_path}")
        sys.exit(1)
        
    fsm = WorkflowFSM(yaml_path, session_manager)
    
    # 2. Initialize Agent Hierarchy
    manager = ManagerAgent(session_manager, fsm)
    
    session_id = "test-session-123"
    
    print("\n[System] Type 'quit' or 'exit' to end the simulation.\n")
    
    try:
        while True:
            # Display current internal state header
            session = session_manager.get_session(session_id)
            print(f"\n--- FSM State: [{session.current_state}] ---")
            
            user_input = input("\nUser: ")
            if user_input.lower() in ['quit', 'exit']:
                break
                
            print("\nManager: ", end="", flush=True)
            response = await manager.handle_turn(session_id, user_input)
            print(response)
            
            # Print Ledger Diagnostics underneath for visibility
            session = session_manager.get_session(session_id)
            print("\n  [Diagnostic] Ledger Snapshot:")
            print(f"  Account:  {session.ledger.account_context}")
            print(f"  Line:     {session.ledger.line_context}")
            print(f"  Trade-in: {session.ledger.trade_in_context}")
            print(f"  New Dev:  {session.ledger.new_device_context}")
            print(f"  Order:    {session.ledger.order_context}")

    finally:
        print("\nSession Terminated.")

if __name__ == "__main__":
    asyncio.run(main())
