# Dynamic FSM-LLM Orchestration: Implementation Plan

This document outlines a "Zero-Code-Change" architecture where a Large Language Model (LLM) acts as the intelligent agent gathering data, and the `transitions` Finite State Machine (FSM) acts as the strict, mathematical router.

By utilizing dynamic string evaluation, you can update your business logic, add new states, and request new variables entirely within the YAML file without ever touching the Python code.

## 1. Architecture Overview

This system separates the "thinking" from the "routing":

* **The Orchestrator (LLM):** Talks to the user, executes backend tools, and translates messy API responses into clean, structured JSON.
* **The Router (FSM):** Uses the `transitions` library to hold the map (YAML). It takes the clean JSON from the LLM, evaluates it against plain-text business rules, and strictly dictates the next state.
* **The Bridge (Python):** A lightweight, static script. It reads the YAML's instructions, prompts the LLM, catches the LLM's JSON, and blindly feeds it to the FSM's universal evaluator.

---

## 2. The Configuration (YAML)

This YAML is written in the native format expected by the `transitions` library. It uses custom keys (`objective`, `extract_variables`, `condition_string`) which the library safely stores as metadata on the State objects for Python to read dynamically.

```yaml
# phone_upgrade_flow.yaml
name: phone_upgrade_flow
initial: Auth

states:
  - name: Auth
    objective: "Verify the user is an authorized account holder using the verify_auth tool."
    extract_variables: ["account_context.is_authorized"]

  - name: AccountStandingCheck
    objective: "Check if the account standing is good using the check_standing tool."
    extract_variables: ["account_context.standing"]

  - name: LineToUpgrade
    objective: "Determine which phone line the user wants to upgrade."
    extract_variables: ["line_context.selected_number"]

  - name: CheckLineUpgradeEligibility
    objective: "Check if the selected line is eligible for an upgrade using the backend tool."
    extract_variables: ["line_context.is_eligible"]

  - name: VerifyTradeIn
    objective: "Ask the user if they want to trade in their current device."
    extract_variables: ["trade_in_context.wants_trade_in"]

  - name: DevicetradeInChecks
    objective: "Ask the user about the physical condition of their trade-in device."
    extract_variables: ["trade_in_context.final_condition"]

  - name: TradeInPricing
    objective: "Get the trade-in quote via the pricing tool based on the final condition."
    extract_variables: ["trade_in_context.quote_value"]

  - name: NewUpgradeDeviceSelection
    objective: "Determine which new device the user wants to purchase."
    extract_variables: ["new_device_context.selection"]

  - name: NewUpgradeDevicePricing
    objective: "Get the pricing for the selected new device using the catalog tool."
    extract_variables: ["new_device_context.price"]

  - name: FinalPricing
    objective: "Calculate final price (New Device Price - Trade In Quote) and present the summary to the user."
    extract_variables: ["order_context.user_confirmed"]

  # --- Terminal States ---
  # transitions requires destinations to be explicitly defined as states
  - name: EndUnauthorized
    objective: "End flow: User not authorized."
  - name: EndBadStanding
    objective: "End flow: Account in bad standing."
  - name: EndNotEligible
    objective: "End flow: Line not eligible."
  - name: ProcessOrder
    objective: "Proceed to backend order processing."

transitions:
  # ==========================================
  # GLOBAL TRANSITIONS (Interruption Intents)
  # Using source: "*" means they trigger from anywhere
  # ==========================================
  - trigger: intent_change_line
    source: "*"
    dest: LineToUpgrade
    before: clear_line_memory 

  - trigger: intent_change_trade_in_device
    source: "*"
    dest: DevicetradeInChecks
    before: clear_trade_in_memory 

  - trigger: intent_change_new_device
    source: "*"
    dest: NewUpgradeDeviceSelection
    before: clear_new_device_memory 

  # ==========================================
  # LOCAL TRANSITIONS (Standard Flow)
  # All use the universal trigger: 'advance'
  # ==========================================
  
  # Auth -> ...
  - trigger: advance
    source: Auth
    dest: AccountStandingCheck
    conditions: generic_evaluator
    condition_string: "context.get('account_context', {}).get('is_authorized') == True"

  - trigger: advance
    source: Auth
    dest: EndUnauthorized
    conditions: generic_evaluator
    condition_string: "context.get('account_context', {}).get('is_authorized') == False"

  # AccountStandingCheck -> ...
  - trigger: advance
    source: AccountStandingCheck
    dest: LineToUpgrade
    conditions: generic_evaluator
    condition_string: "context.get('account_context', {}).get('standing') == 'GOOD'"

  - trigger: advance
    source: AccountStandingCheck
    dest: EndBadStanding
    conditions: generic_evaluator
    condition_string: "context.get('account_context', {}).get('standing') != 'GOOD'"

  # LineToUpgrade -> ...
  - trigger: advance
    source: LineToUpgrade
    dest: CheckLineUpgradeEligibility
    conditions: generic_evaluator
    condition_string: "context.get('line_context', {}).get('selected_number') is not None"

  # CheckLineUpgradeEligibility -> ...
  - trigger: advance
    source: CheckLineUpgradeEligibility
    dest: VerifyTradeIn
    conditions: generic_evaluator
    condition_string: "context.get('line_context', {}).get('is_eligible') == True"

  - trigger: advance
    source: CheckLineUpgradeEligibility
    dest: EndNotEligible
    conditions: generic_evaluator
    condition_string: "context.get('line_context', {}).get('is_eligible') == False"

  # VerifyTradeIn -> ...
  - trigger: advance
    source: VerifyTradeIn
    dest: DevicetradeInChecks
    conditions: generic_evaluator
    condition_string: "context.get('trade_in_context', {}).get('wants_trade_in') == True"

  - trigger: advance
    source: VerifyTradeIn
    dest: NewUpgradeDeviceSelection
    conditions: generic_evaluator
    condition_string: "context.get('trade_in_context', {}).get('wants_trade_in') == False"

  # DevicetradeInChecks -> ...
  - trigger: advance
    source: DevicetradeInChecks
    dest: TradeInPricing
    conditions: generic_evaluator
    condition_string: "context.get('trade_in_context', {}).get('final_condition') is not None"

  # TradeInPricing -> ...
  - trigger: advance
    source: TradeInPricing
    dest: NewUpgradeDeviceSelection
    conditions: generic_evaluator
    condition_string: "context.get('trade_in_context', {}).get('quote_value', -1) >= 0"

  # NewUpgradeDeviceSelection -> ...
  - trigger: advance
    source: NewUpgradeDeviceSelection
    dest: NewUpgradeDevicePricing
    conditions: generic_evaluator
    condition_string: "context.get('new_device_context', {}).get('selection') is not None"

  # NewUpgradeDevicePricing -> ...
  - trigger: advance
    source: NewUpgradeDevicePricing
    dest: FinalPricing
    conditions: generic_evaluator
    condition_string: "context.get('new_device_context', {}).get('price', 0) > 0"

  # FinalPricing -> ...
  - trigger: advance
    source: FinalPricing
    dest: ProcessOrder
    conditions: generic_evaluator
    condition_string: "context.get('order_context', {}).get('user_confirmed') == True"
```

---

## 3. The Routing Engine (Python FSM Setup) - Example code

This Python code sets up the FSM and defines the `generic_evaluator`, which is the single function responsible for calculating every routing decision in the YAML.

```python
import yaml
from transitions import Machine

class FlowController:
    def generic_evaluator(self, event):
        """
        The Universal Brain: Evaluates the YAML condition_string 
        against the JSON context provided by the LLM.
        """
        condition_str = event.kwargs.get('condition_string')
        llm_context = event.kwargs.get('context', {}) 
        
        try:
            # Safely evaluate the plain-text YAML logic using the LLM's data
            return bool(eval(condition_str, {}, {"context": llm_context}))
        except Exception as e:
            print(f"Evaluation Error on '{condition_str}': {e}")
            return False

    def clear_line_memory(self, event):
        """Called automatically when the intent_change_line global transition fires."""
        print("System: Wiping line_context memory for a fresh start...")
        # Add logic here to clear session variables or database state

# Load YAML
with open("phone_upgrade_flow.yaml", "r") as f:
    yaml_config = yaml.safe_load(f)

# Initialize Controller & FSM
controller = FlowController()
machine = Machine(
    model=controller,
    states=yaml_config['states'],
    transitions=yaml_config['transitions'],
    initial=yaml_config['initial'],
    send_event=True  # CRITICAL: Allows passing 'context' dict to generic_evaluator
)
```

---

## 4. The Orchestration Loop (LLM Interaction) - Example code

This function represents a single "turn" in your application. It reads the current FSM state, builds a dynamic prompt, mocks the LLM response, and feeds the resulting JSON back into the FSM.

```python
import json

def run_agent_turn(user_input):
    # 1. Dynamically read instructions from the current FSM state
    current_state_obj = machine.get_state(controller.state)
    objective = getattr(current_state_obj, 'objective', 'No objective defined.')
    required_vars = getattr(current_state_obj, 'extract_variables', [])

    # 2. Build the System Prompt
    system_prompt = f"""
    You are an AI assistant. You are currently in the state: '{controller.state}'.
    Your objective is: {objective}
    
    User said: "{user_input}"
    
    1. Use your backend tools to achieve this objective.
    2. Once you have the final answer, you MUST return a structured JSON response containing EXACTLY these nested variables:
    {required_vars}
    """
    print(f"\n--- 1. Prompting LLM ---\n{{system_prompt}}")
    
    # ---------------------------------------------------------
    # 3. [MOCK LLM EXECUTION]
    # In production, you pass the prompt to OpenAI/Gemini, the LLM 
    # uses its tools, and returns a JSON string via Function Calling.
    # ---------------------------------------------------------
    mock_llm_json_output = """
    {
        "account_context": {
            "is_authorized": true
        }
    }
    """
    print(f"--- 2. LLM JSON Response ---\n{{mock_llm_json_output.strip()}}")
    
    # 4. Feed the LLM's JSON back to the FSM Engine
    llm_context_dict = json.loads(mock_llm_json_output)
    
    print(f"\n--- 3. FSM Routing ---")
    print(f"Old State: {{controller.state}}")
    
    # Fire the universal trigger, passing the LLM's data as 'context'
    controller.advance(context=llm_context_dict)
    
    print(f"New State: {{controller.state}}")

# Run a test iteration
if __name__ == "__main__":
    run_agent_turn("My pin is 1234")
```

---

## 5. Execution Walkthrough

When you execute `run_agent_turn()`, the following sequence dynamically occurs:

1. **FSM Initialized:** `controller.state` starts at the initial YAML state: `Auth`.
2. **Prompt Generation:** Python dynamically grabs `Auth`'s objective ("Verify the user...") and `extract_variables` (`["account_context.is_authorized"]`) and instructs the LLM.
3. **LLM Work:** The LLM processes the user's input, executes the necessary backend tool (e.g., `auth_check`), interprets the messy response, and formats it into the exact requested JSON: `{"account_context": {"is_authorized": true}}`.
4. **Trigger Fired:** Python parses the JSON and blindly calls the FSM trigger: `controller.advance(context=...)`.
5. **FSM Evaluates:**
    * `transitions` checks the first transition rule out of the `Auth` state.
    * It passes the JSON data to the `generic_evaluator` hook.
    * The evaluator grabs the YAML rule: `"context.get('account_context', {}).get('is_authorized') == True"` and runs the math against the LLM's JSON.
    * Because the LLM passed `true`, the mathematical evaluation returns `True`.
6. **State Change:** The FSM instantly shifts to `AccountStandingCheck`. The next time the loop runs, the prompt will automatically instruct the LLM to verify the account standing.