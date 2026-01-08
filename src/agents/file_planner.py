"""File Planner Agent - creates generation plans with parallel batches."""

from src.agents.base import LLMOnlyAgent
from src.schemas.models import GenerationPlan

SYSTEM_PROMPT = """You are a file generation planner for Anchor 0.30.x Solana contracts.
Given a token specification, plan which files to generate and which can be done in parallel.

Output a JSON object with:
- batches: array of batch objects with:
  - batch_id: unique identifier (e.g., "batch_1", "accounts", "instructions")
  - file_paths: array of relative file paths
  - description: what these files are (e.g., "Account struct definitions")
  - dependencies: array of batch_ids this depends on (empty if first batch)
  - priority: lower numbers = generate first
- total_files: count of all files to generate
- generation_order: array of arrays, where each inner array contains
  batch_ids that can run in PARALLEL at that step

Rules:
1. Files with NO dependencies on each other can be parallelized
2. accounts.rs must come BEFORE instruction handlers that use those accounts
3. errors.rs can be generated in parallel with accounts.rs (independent)
4. Instruction handlers (in instructions/) depend on accounts.rs and errors.rs
5. Test files can be generated last (they depend on everything else)
6. Keep batches balanced (2-5 files per batch ideal)
7. Always include: accounts.rs, errors.rs (if needed), instruction handlers
8. NEVER include lib.rs - it already exists from anchor init with correct declare_id!

File structure for Anchor projects:
- programs/{project}/src/lib.rs - Main module with #[program], ALREADY HAS correct declare_id! from anchor init (DO NOT include in batches - code_generator only adds handlers inside #[program] module, never rewrites the file)
- programs/{project}/src/accounts.rs - Account struct definitions
- programs/{project}/src/errors.rs - Custom error types
- programs/{project}/src/instructions/mod.rs - Module declarations
- programs/{project}/src/instructions/*.rs - Individual instruction handlers

CRITICAL: lib.rs ALREADY EXISTS with correct declare_id! from anchor init.
- DO NOT include lib.rs in a batch that regenerates it - it would overwrite the correct program ID
- Only code_generator adds instruction handlers INSIDE the #[program] module
- Batches should focus on: accounts.rs, errors.rs, instructions/*.rs

Example output:
{
  "batches": [
    {"batch_id": "accounts", "file_paths": ["programs/{project}/src/accounts.rs"],
     "description": "Account struct definitions", "dependencies": [], "priority": 1},
    {"batch_id": "errors", "file_paths": ["programs/{project}/src/errors.rs"],
     "description": "Custom error types", "dependencies": [], "priority": 1},
    {"batch_id": "instructions", "file_paths": ["programs/{project}/src/instructions/mod.rs",
     "programs/{project}/src/instructions/initialize.rs", "programs/{project}/src/instructions/transfer.rs"],
     "description": "Instruction handlers", "dependencies": ["accounts", "errors"], "priority": 2}
  ],
  "total_files": 5,
  "generation_order": [["accounts", "errors"], ["instructions"]]
}

Return ONLY valid JSON. No markdown formatting, no explanations.
"""


class FilePlanner(LLMOnlyAgent):
    """Agent that creates generation plans with parallel batches."""

    @property
    def agent_name(self):
        return "file_planner"

    def _create_executor(self):
        """Create structured LLM for JSON output."""
        return self.llm.with_structured_output(GenerationPlan)

    def _get_system_prompt(self):
        return SYSTEM_PROMPT

    def _format_state_for_agent(self, state: dict) -> str:
        """Format the token spec for the planner."""
        spec = state.get("interpreted_spec", {})
        project_name = state.get("project_name", "my_contract")

        # Extract key information from spec
        name = spec.get("name", project_name) if spec else project_name
        instructions = spec.get("instructions", []) if spec else []
        accounts = spec.get("accounts", []) if spec else []
        features = spec.get("features", []) if spec else []

        # Determine which files are needed based on the spec
        needed_files = []
        if accounts:
            needed_files.append("programs/{project}/src/accounts.rs")
        needed_files.append("programs/{project}/src/errors.rs")
        # lib.rs is NOT included - it already exists from anchor init with correct declare_id!
        # Only code_generator adds instruction handlers INSIDE the #[program] module
        needed_files.append("programs/{project}/src/instructions/mod.rs")

        for instr in instructions:
            # Convert instruction name to file path
            safe_name = instr.lower().replace("_", "")
            needed_files.append(f"programs/{project_name}/src/instructions/{safe_name}.rs")

        return f"""Plan file generation for this Anchor 0.30.x contract:

Project name (crate): {name}
Features: {features}

Instructions to implement: {instructions}
Accounts needed: {accounts}

Files to generate ({len(needed_files)} total):
{chr(10).join(f"- {f}" for f in needed_files)}

Generate a GenerationPlan JSON object that:
1. Groups independent files into parallel batches
2. Respects dependencies (accounts before instructions, etc.)
3. Includes all necessary files

Return valid JSON only."""

    def _format_agent_result(self, state: dict, response) -> dict:
        """Format the structured GenerationPlan response."""
        # DEBUG: Log response type and content
        import sys

        print(f"[FilePlanner DEBUG] Response type: {type(response)}", file=sys.stderr)
        if hasattr(response, "content"):
            content = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
            print(f"[FilePlanner DEBUG] Raw content: {content[:500]}", file=sys.stderr)
        print(
            f"[FilePlanner DEBUG] Has model_dump: {hasattr(response, 'model_dump')}",
            file=sys.stderr,
        )

        # Handle Pydantic model response from with_structured_output
        if hasattr(response, "model_dump"):
            # Structured output - response is a Pydantic model (GenerationPlan)
            generation_plan = response
            print(
                f"[FilePlanner DEBUG] Parsed generation_plan: {generation_plan.model_dump()}",
                file=sys.stderr,
            )
            return {
                **state,
                "generation_plan": generation_plan,
                "current_step": "file_planner",
            }
        # Fallback for regular text response
        print(f"[FilePlanner DEBUG] Non-structured response: {response}", file=sys.stderr)
        return super()._format_agent_result(state, response)
