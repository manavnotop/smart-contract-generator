"""Code Generator Agent - writes Rust instruction handlers using LangChain."""

from src.agents.base import LLMOnlyAgent
from src.schemas.models import ProjectFiles

SYSTEM_PROMPT = """You are an expert Solana smart contract Rust developer specializing in
Anchor 0.30.x. The Anchor project is already initialized. Your task is to write
complete, production-ready Rust code for ANY Solana program based on the specification.

Output a JSON object with a "files" array. Each file has:
- path: relative file path (e.g., "programs/project/src/lib.rs")
- content: complete file contents

KEY: Use anchor_spl ONLY if the contract needs token operations (mint, transfer, burn).
For non-token programs (counter, escrow without tokens, etc.), use plain Anchor without anchor_spl.

Requirements:
1. Follow Anchor 0.30.x idioms and best practices
2. Use modern Rust (2021 edition)
3. Implement proper error handling
4. Include all necessary imports and use statements
5. Add inline documentation for complex logic
6. Use proper error types with human-readable messages

For each instruction handler:
- Create context struct with required accounts (#[derive(Accounts)])
- Define instruction data struct with #[derive(AnchorSerialize, AnchorDeserialize)]
- Implement the handler function with proper access control
- Include precondition checks with appropriate errors

For custom data structures (accounts):
- Define them with #[account]
- Use appropriate #[init] if needing PDA initialization
- Include space calculation

File structure:
- programs/{project}/src/lib.rs - Main module with #[program], declare_id!, instruction modules (DO NOT add "pub mod accounts;" here!)
- programs/{project}/src/instructions/*.rs - Each instruction handler in separate files
- programs/{project}/src/accounts.rs - Account struct definitions (import with: use crate::accounts::StructName;)
- programs/{project}/src/errors.rs - Custom error types

CRITICAL: Do NOT declare "pub mod accounts;" in lib.rs - the #[program] macro generates its own accounts namespace and this will cause E0428 conflicts!

Examples:
- Counter: lib.rs should NOT have "pub mod accounts;" - instead use: use crate::accounts::Counter;
- accounts.rs has Counter { count: u64, authority: Pubkey }, increment instruction bumps count
- Token: uses anchor_spl::token, Mint, TokenAccount, MintTo, Transfer
- Escrow: has Escrow account with seed, state, amounts

Only include files that are needed. Skip events.rs if no events. Skip errors.rs if using generic errors.
"""


class CodeGenerator(LLMOnlyAgent):
    """Agent that generates Rust instruction implementations using LangChain."""

    @property
    def agent_name(self):
        return "code_generator"

    def _create_executor(self):
        """Create structured LLM for JSON output."""
        return self.llm.with_structured_output(ProjectFiles)

    def _get_system_prompt(self):
        return SYSTEM_PROMPT

    def _format_state_for_agent(self, state: dict) -> str:
        """Format the token spec and existing files for the agent."""
        spec = state.get("interpreted_spec", {})
        existing_files = state.get("files", {})

        files_summary = "\n".join(f"- {path}" for path in existing_files)

        return f"""Generate Rust instruction implementations for:

Contract: {spec.get("name", "Unknown")}
Description: {spec.get("description", "N/A")}
Features: {spec.get("features", [])}

Instructions to implement: {spec.get("instructions", [])}
Accounts needed: {spec.get("accounts", [])}
Data structures: {spec.get("data_structs", [])}

Existing files:
{files_summary}

Write complete Rust code for all instruction handlers. Split into proper files.
Use anchor_spl only if mintable, burnable, or transferable features are present."""

    def _format_agent_result(self, state: dict, result: ProjectFiles) -> dict:
        """Format the structured response."""
        if result and result.files:
            # Convert list of ProjectFile to dict[str, str]
            files_dict = {f.path: f.content for f in result.files}
            updated_files = {**state.get("files", {}), **files_dict}
            return {
                **state,
                "files": updated_files,
                "current_step": "static_validator",
            }

        return {
            **state,
            "error_message": "Failed to generate code",
            "current_step": "code_generator",
        }
