"""Spec Interpreter Agent - converts natural language to structured TokenSpec using LangChain."""

import json
import re

from src.agents.base import LLMOnlyAgent
from src.schemas.models import TokenSpec

SYSTEM_PROMPT = """You are a Solana smart contract specification interpreter.
Your job is to convert ANY natural language specification into a structured TokenSpec.

You support ALL types of Solana programs:
- Token contracts (mintable, burnable, transferable, etc.)
- Counter programs (increment, decrement, reset)
- Escrow contracts (deposit, withdraw, cancel)
- Staking programs (stake, unstake, claim rewards)
- Vault programs (deposit, withdraw, split)
- Custom programs with any instructions

For the specification, determine:
1. What is the program NAME?
2. What INSTRUCTIONS are needed (e.g., initialize, mint, transfer, increment, deposit)?
3. What ACCOUNTS are required (e.g., counter, mint, token_account, user)?
4. What DATA STRUCTURES are needed (custom account state)?
5. What FEATURES apply (mintable, burnable, counter, escrow, etc.)?

Output format: JSON only, no markdown, matching this schema:
{
    "name": "Program Name",
    "symbol": "SYM or null (only for tokens)",
    "description": "Brief description of what this program does",
    "decimals": 9 or null (only for tokens)",
    "features": ["mintable", "counter", "escrow", ...],
    "initial_supply": null or number (only for tokens)",
    "instructions": ["initialize", "increment", "decrement", ...],
    "accounts": ["counter", "authority", "user", ...],
    "data_structs": [{"name": "Counter", "fields": [{"name": "count", "type": "u64"}, {"name": "authority", "type": "pubkey"}]}]
}

Examples:
- "create a counter program" -> instructions: ["initialize", "increment"], accounts: ["counter", "authority"], data_structs: Counter with count field
- "create a mintable token" -> instructions: ["initialize", "mint", "transfer"], features: ["mintable", "transferable"], accounts: ["mint", "token_account"]
- "create an escrow contract" -> instructions: ["initialize", "deposit", "withdraw", "cancel"], features: ["escrow"], accounts: ["escrow", " initializer", "temp_token_account"]

Infer missing information from the specification. Default to simple, secure implementations.
"""


class SpecInterpreter(LLMOnlyAgent):
    """Agent that interprets natural language specifications using LangChain."""

    def __init__(self, test_mode: bool = False):
        """Initialize SpecInterpreter agent.

        Args:
            test_mode: If True, use mock LLM for testing
        """
        super().__init__()
        self.test_mode = test_mode
        if test_mode:
            from src.utils.llm_utils import MockLLM

            self.llm = MockLLM(model="mock-spec-interpreter")

    @property
    def agent_name(self) -> str:
        return "spec_interpreter"

    def _get_system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def _format_state_for_agent(self, state: dict) -> str:
        """Extract user spec from state for the agent."""
        user_spec = state.get("user_spec", "")
        if not user_spec:
            return "Error: No user specification provided"
        return f"Interpret this specification:\n\n{user_spec}"

    def _extract_state_from_response(self, state: dict, response: str) -> dict:
        """Parse the LLM response and extract TokenSpec."""
        try:
            # Clean up response if it has markdown formatting
            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()

            data = json.loads(clean_response)
            token_spec = TokenSpec(**data)

            # Generate project_name from token name
            name = re.sub(r"[^a-z0-9]+", "_", token_spec.name.lower()).strip("_")[:32]
            if not name:
                name = "solana_contract"

            return {
                **state,
                "interpreted_spec": token_spec.model_dump(),
                "project_name": name,
                "current_step": "project_planner",
            }
        except (json.JSONDecodeError, ValueError) as e:
            return {
                **state,
                "error_message": f"Failed to parse spec interpretation: {e}\nRaw: {response}",
                "current_step": "spec_interpreter",
            }
