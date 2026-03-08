import yaml
from typing import Dict, Any, Tuple, Optional
from simpleeval import simple_eval

from transitions import Machine
from orchestrator.session import SessionManager, SessionData

class WorkflowFSM:
    def __init__(self, yaml_path: str, session_manager: SessionManager):
        self.session_manager = session_manager
        with open(yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.states = list(self.config['states'].keys())
        self.initial_state = self.config['initial_state']
        
        # We don't use 'transitions' library triggers for progression, 
        # we manually move the state pointer after evaluating the rules.
        self.machine = Machine(model=self, states=self.states, initial=self.initial_state)

    def evaluate(self, session_id: str, intent_override: Optional[str] = None) -> Tuple[str, str]:
        """
        Evaluates the current state against the Session Ledger.
        Returns the (new_state, objective).
        """
        session = self.session_manager.get_session(session_id)
        
        # 1. Check Global Transitions First (e.g. Change of Mind)
        if intent_override:
            for global_tx in self.config.get('global_transitions', []):
                if global_tx['intent'] == intent_override:
                    print(f"[FSM] Global intent '{intent_override}' detected. Wiping memory: {global_tx.get('clear_memory')}")
                    # Wipe memory
                    if 'clear_memory' in global_tx:
                        self.session_manager.clear_memory_keys(session_id, global_tx['clear_memory'])
                    
                    # Update State
                    self.state = global_tx['next_state']
                    session.current_state = self.state
                    session.current_objective = self._get_objective(self.state)
                    self.session_manager.save_session(session)
                    return (self.state, session.current_objective)

        # 2. Local State Evaluation
        current_state_config = self.config['states'][session.current_state]
        transitions = current_state_config.get('transitions', [])
        
        # If terminal state, return current
        if not transitions:
            return (session.current_state, self._get_objective(session.current_state))
            
        # Context payload for simpleeval
        # Flatten the ledger into a dict for easy dot notation evaluation
        eval_context = {
            "account_context": session.ledger.account_context,
            "line_context": session.ledger.line_context,
            "trade_in_context": session.ledger.trade_in_context,
            "new_device_context": session.ledger.new_device_context,
            "order_context": session.ledger.order_context
        }

        for tx in transitions:
            condition = tx['condition']
            try:
                # Evaluate the string against our context ("account_context.is_authorized == True")
                # simpleeval needs variables passed in 'names' mapping
                is_true = simple_eval(condition, names=eval_context)
                
                if is_true:
                    next_state = tx['next_state']
                    print(f"[FSM] Condition '{condition}' met. Transitioning to {next_state}.")
                    self.state = next_state
                    session.current_state = next_state
                    session.current_objective = self._get_objective(next_state)
                    self.session_manager.save_session(session)
                    return (next_state, session.current_objective)
            except Exception as e:
                print(f"[FSM Error] Could not evaluate '{condition}': {e}")
                
        # If no conditions met, stay in current state
        print(f"[FSM] No conditions met. Staying in {session.current_state}.")
        session.current_objective = self._get_objective(session.current_state)
        self.session_manager.save_session(session)
        return (session.current_state, session.current_objective)

    def _get_objective(self, state_name: str) -> str:
        return self.config['states'].get(state_name, {}).get('objective', "No objective found.")
    
    def get_current_objective(self, session_id: str) -> str:
        session = self.session_manager.get_session(session_id)
        if session.current_state is None:
            session.current_state = self.initial_state
        return self._get_objective(session.current_state)
