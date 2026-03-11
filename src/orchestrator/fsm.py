import yaml
import logging
from simpleeval import simple_eval
from transitions import Machine

logger = logging.getLogger(__name__)


def _make_condition(condition_str: str, ledger_keys: list):
    """Return a condition function that evaluates condition_str against context kwarg.

    Normalizes context so all ledger keys are present (safe subscript access in conditions).
    ledger_keys is derived from the YAML's extract_variables — no hardcoding in Python.
    simpleeval blocks {} dict literals, so conditions use context['key'].get(...) syntax.
    """
    def condition(event_data):
        raw = event_data.kwargs.get('context', {})
        # Ensure all ledger keys exist so condition strings can use subscript access safely
        context = {k: raw.get(k, {}) for k in ledger_keys}
        try:
            return bool(simple_eval(condition_str, names={'context': context}))
        except Exception as e:
            logger.warning(f"[FSM Error] '{condition_str}': {e}")
            return False
    condition.__name__ = f"cond_{condition_str[:30]}"
    return condition


class FlowController:
    """Model object for the transitions Machine. Memory-wipe callbacks are generated dynamically."""
    pass


class WorkflowFSM:
    """Stateless FSM: set state externally, fire advance, read new state."""

    def __init__(self, yaml_path: str):
        with open(yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.initial_state = self.config['initial']

        # Keep state metadata (objective, extract_variables) in a fast lookup dict.
        self._state_meta = {s['name']: s for s in self.config['states']}

        # Derive ledger keys from YAML extract_variables — no hardcoding in Python.
        ledger_keys = list({
            var_path.split('.')[0]
            for state in self.config['states']
            for var_path in state.get('extract_variables', [])
            if '.' in var_path
        })

        self.controller = FlowController()

        # Pre-process transitions:
        # - Strip custom YAML keys (condition_string, transition_type, clear_keys, description)
        #   that are not recognised by the transitions library.
        # - For condition_string: generate a condition closure and attach to controller.
        # - For global transitions (transition_type==global): generate a memory-wipe closure
        #   from clear_keys and wire it as the before callback.
        processed_transitions = []
        for i, tx in enumerate(self.config['transitions']):
            tx_copy = dict(tx)

            # Strip all custom keys up front
            cond_str = tx_copy.pop('condition_string', None)
            tx_copy.pop('transition_type', None)
            tx_copy.pop('description', None)
            clear_keys = tx_copy.pop('clear_keys', None)

            if cond_str:
                method_name = f'_cond_{i}'
                setattr(self.controller, method_name, _make_condition(cond_str, ledger_keys))
                tx_copy['conditions'] = method_name

            if clear_keys:
                method_name = f'_clear_memory_{i}'
                def _make_clearer(keys):
                    def clearer(event_data):
                        ledger = event_data.kwargs.get('ledger', {})
                        for k in keys:
                            ledger[k] = {}
                    return clearer
                setattr(self.controller, method_name, _make_clearer(clear_keys))
                tx_copy['before'] = method_name

            processed_transitions.append(tx_copy)

        # Pass only state names to Machine (strip objective/extract_variables).
        state_names = [s['name'] for s in self.config['states']]

        self.machine = Machine(
            model=self.controller,
            states=state_names,
            transitions=processed_transitions,
            initial=self.initial_state,
            send_event=True,           # EventData passed to all callbacks
            ignore_invalid_triggers=True,  # return False instead of raising on terminal states
        )

    def evaluate(self, current_state: str, context: dict) -> str:
        """Set state, fire advance, return new state (or current if no transition fires)."""
        self.machine.set_state(current_state, model=self.controller)
        try:
            self.controller.advance(context=context)
        except Exception:
            pass  # safety net
        return self.controller.state

    def fire_intent(self, current_state: str, trigger_name: str, ledger: dict) -> str:
        """Fire a global intent trigger by its full trigger name; clears relevant ledger keys in-place."""
        self.machine.set_state(current_state, model=self.controller)
        trigger = getattr(self.controller, trigger_name, None)
        if trigger:
            try:
                trigger(ledger=ledger)
            except Exception as e:
                logger.warning(f"[FSM] Intent trigger failed: {e}")
        return self.controller.state

    def get_global_intents(self) -> list:
        """Return trigger name and description for all global transitions (transition_type==global)."""
        return [
            {'trigger': tx['trigger'], 'description': tx.get('description', '')}
            for tx in self.config['transitions']
            if tx.get('transition_type') == 'global'
        ]

    def get_objective(self, state_name: str) -> str:
        return self._state_meta.get(state_name, {}).get('objective', 'No objective found.')

    def get_extract_variables(self, state_name: str) -> list:
        return self._state_meta.get(state_name, {}).get('extract_variables', [])

    def get_all_extract_variables(self) -> list:
        """Return all extract_variable paths across all states (deduplicated, ordered)."""
        seen = set()
        result = []
        for state in self.config['states']:
            for var in state.get('extract_variables', []):
                if var not in seen:
                    seen.add(var)
                    result.append(var)
        return result
