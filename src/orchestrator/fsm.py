import yaml
from typing import Optional
from simpleeval import simple_eval
from transitions import Machine


class WorkflowFSM:
    def __init__(self, yaml_path: str):
        with open(yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.states = list(self.config['states'].keys())
        self.initial_state = self.config['initial_state']

        # transitions.Machine validates all next_state values are registered states,
        # catching YAML authoring errors at startup.
        self.machine = Machine(model=self, states=self.states, initial=self.initial_state)

    def evaluate(self, current_state: str, ledger: dict, intent_override: Optional[str] = None) -> str:
        """
        Evaluates current_state against ledger and returns next_state.
        If intent_override is provided, checks global transitions first and
        wipes the specified ledger keys in-place before returning.
        """
        # 1. Global Transitions (change-of-mind)
        if intent_override:
            for global_tx in self.config.get('global_transitions', []):
                if global_tx['intent'] == intent_override:
                    print(f"[FSM] Global intent '{intent_override}'. Wiping: {global_tx.get('clear_memory')}")
                    for key in global_tx.get('clear_memory', []):
                        ledger[key] = {}
                    return global_tx['next_state']

        # 2. Local State Evaluation
        current_state_config = self.config['states'].get(current_state, {})
        transitions = current_state_config.get('transitions', [])

        if not transitions:
            # Terminal state — stay
            return current_state

        eval_context = {k: ledger.get(k, {}) for k in [
            "account_context", "line_context", "trade_in_context",
            "new_device_context", "order_context"
        ]}

        for tx in transitions:
            condition = tx['condition']
            try:
                if simple_eval(condition, names=eval_context):
                    next_state = tx['next_state']
                    print(f"[FSM] '{condition}' → {next_state}")
                    return next_state
            except Exception as e:
                print(f"[FSM Error] Could not evaluate '{condition}': {e}")

        print(f"[FSM] No conditions met. Staying in {current_state}.")
        return current_state

    def get_objective(self, state_name: str) -> str:
        return self.config['states'].get(state_name, {}).get('objective', "No objective found.")
