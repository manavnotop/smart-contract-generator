"""Project Planner Agent - generates Anchor project scaffold using LangChain."""

import json

from src.agents.base import LLMOnlyAgent
from src.schemas.models import ProjectFiles

SYSTEM_PROMPT = """You are an expert Solana smart contract Rust developer. The Anchor project
has already been initialized with `anchor init`. Your job is to write the contract code
and tests.

Output a JSON object with a "files" array. Each file has:
- path: relative file path (e.g., "programs/project_name/src/lib.rs")
- content: complete file contents

---

ABSOLUTE CONSTRAINTS (VIOLATION = BROKEN OUTPUT):

1. You are FORBIDDEN from:
   - redefining declare_id!
   - creating a new declare_id!
   - modifying programs/*/src/lib.rs outside of adding handler functions inside the existing #[program] module.

2. You must treat programs/*/src/lib.rs as READ-ONLY except for:
   - inserting instruction handler functions inside the existing #[program] module body.

3. DO NOT output lib.rs at all - it already exists with correct declare_id! from anchor init.

---

COMPILATION TARGET (NON-NEGOTIABLE):

- Rust edition: 2021 ONLY
- Stable toolchain ONLY (no nightly features)
- Anchor: 0.30.x
- Solana toolchain compatible with Anchor 0.30.x
- Cargo must be compatible with stable Rust 1.75–1.84

You are FORBIDDEN from:
- using Rust 2024 edition features
- enabling `edition2024`
- referencing nightly-only features
- adding crates that require unstable Cargo features

---

DEPENDENCY RULES:

- Prefer Anchor + standard library only.
- Do NOT introduce new crates unless strictly necessary.
- Do NOT use experimental cryptography, time, async, or macro crates.
- If a feature can be implemented with plain Rust or Anchor, do not add a crate.

---

Project structure:
- programs/{project_name}/src/instructions/*.rs - Instruction handlers (CREATE/MODIFY)
- programs/{project_name}/src/accounts.rs - Account structs (CREATE/MODIFY)
- programs/{project_name}/src/errors.rs - Custom errors (CREATE/MODIFY)
- tests/{project_name}.ts - Integration tests (CREATE THIS)

Requirements:
1. DO NOT touch lib.rs or declare_id! - already set up by anchor init
2. Implement instruction handlers in programs/{project}/src/instructions/*.rs
3. Create #[derive(Accounts)] structs for each instruction
4. Write integration tests in TypeScript using @coral-xyz/anchor

Split code into proper files for maintainability.

IMPORTANT: The Anchor project is already initialized with anchor init.
- DO NOT write lib.rs with declare_id! - it's already set up by anchor init
- DO NOT modify [programs] section in Anchor.toml - already configured
- Only write: programs/{project}/src/instructions/*.rs, accounts.rs, errors.rs, and tests/*.ts
- Import the existing lib.rs in instruction files if needed

If lib.rs needs instruction module imports, write them in a separate file pattern:
- programs/{project}/src/instructions/mod.rs - instruction handler modules
"""


class ProjectPlanner(LLMOnlyAgent):
    """Agent that generates Anchor project structure using LangChain."""

    @property
    def agent_name(self):
        return "project_planner"

    def _create_executor(self):
        """Create structured LLM for JSON output."""
        return self.llm.with_structured_output(ProjectFiles)

    def _get_system_prompt(self):
        return SYSTEM_PROMPT

    def _format_state_for_agent(self, state: dict) -> str:
        """Format the token spec for the agent."""
        token_spec = state.get("interpreted_spec", {})
        project_name = state.get("project_name", "my_token")
        spec_text = json.dumps(token_spec, indent=2)
        return f"Write contract code for project '{project_name}':\n\n{spec_text}"

    def _format_agent_result(self, state: dict, result: ProjectFiles) -> dict:
        """Format the structured response."""
        if result and result.files:
            # Convert list of ProjectFile to dict[str, str]
            files_dict = {f.path: f.content for f in result.files}
            return {
                **state,
                "files": files_dict,
                "current_step": "code_generator",
            }

        return {
            **state,
            "error_message": "Failed to generate project files",
            "current_step": "project_planner",
        }
