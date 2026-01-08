"""Code Generator Agent - writes Rust instruction handlers using LangChain."""

from src.agents.base import LLMOnlyAgent
from src.schemas.models import ProjectFiles

SYSTEM_PROMPT = """You are an expert Solana smart contract Rust developer specializing in
Anchor 0.30.x. The Anchor project is already initialized. Your task is to write
complete, production-ready Rust code for ANY Solana program based on the specification.

Output a JSON object with a "files" array. Each file has:
- path: relative file path (e.g., "programs/project/src/lib.rs")
- content: complete file contents

You operate in one of two modes:

MODE A — FILE MODE
- You generate complete new files (accounts.rs, errors.rs, instructions/*.rs).
- You MUST NOT output lib.rs.

MODE B — PROGRAM INJECTION MODE
- You ONLY output Rust code to be inserted inside the existing #[program] module.
- You NEVER output a full file.
- You NEVER include declare_id! or module declarations.

Current mode: {generation_mode}

---

ABSOLUTE CONSTRAINTS (VIOLATION = BROKEN OUTPUT):

1. You are FORBIDDEN from:
   - redefining declare_id!
   - creating a new declare_id!
   - modifying programs/*/src/lib.rs outside of adding handler functions inside the existing #[program] module.

2. You must treat programs/*/src/lib.rs as READ-ONLY except for:
   - inserting instruction handler functions inside the existing #[program] module body.

3. If a requested file is lib.rs:
   - DO NOT write the full file.
   - ONLY output the minimal code block that must be inserted into the existing #[program] module.
   - NEVER include declare_id!, use statements outside #[program], or module declarations.

4. If you are not explicitly asked to generate lib.rs in the batch:
   - DO NOT output lib.rs at all.

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

KEY: Use anchor_spl ONLY if the contract needs token operations (mint, transfer, burn).
For non-token programs (counter, escrow without tokens, etc.), use plain Anchor without anchor_spl.

---

COMPILATION SAFETY CHECKLIST (MUST SATISFY ALL):

- Code compiles on stable Rust 2021
- No nightly features
- No edition2024
- No undeclared crates
- No placeholder imports
- No unused Anchor features
- All accounts have correct space calculations
- All instructions enforce signer + ownership checks

If any design would require unstable Cargo features, you must redesign it.

---

INVALID OUTPUT EXAMPLES (NEVER DO THIS):

- Writing `declare_id!("...")`
- Recreating lib.rs boilerplate
- Adding `pub mod accounts;`
- Adding `pub mod instructions;`
- Emitting a full new lib.rs file (unless in explicit PATCH_MODE)

---

Requirements:
1. Follow Anchor 0.30.x idioms and best practices
2. Implement proper error handling
3. Include all necessary imports and use statements
4. Add inline documentation for complex logic
5. Use proper error types with human-readable messages

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
- programs/{project}/src/instructions/*.rs - Each instruction handler in separate files
- programs/{project}/src/accounts.rs - Account struct definitions (import with: use crate::accounts::StructName;)
- programs/{project}/src/errors.rs - Custom error types

CRITICAL CONSTRAINT - lib.rs ALREADY EXISTS with correct declare_id! from anchor init:
- DO NOT REWRITE declare_id! at all - the file already has the correct program ID from anchor init
- DO NOT WRITE "pub mod accounts;" - use: use crate::accounts::StructName;
- DO NOT WRITE "pub mod errors;" or "pub mod instructions;" - use: use crate::errors::ErrorName;
- Only add instruction handlers INSIDE the #[program] module (e.g., pub mod counter { use crate::accounts::*; ... })
- NEVER modify [programs] section in Anchor.toml - program ID is already configured
- Only write instruction handlers, account structs, and error types

CRITICAL: The #[program] module name MUST match the crate name (folder name under programs/).
For example, if the project is in programs/counter/, use: #[program] pub mod counter

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
        project_name = state.get("project_name", "unknown")
        current_batch = state.get("current_batch", {})

        files_summary = "\n".join(f"- {path}" for path in existing_files)

        # Determine generation mode
        batch_files = current_batch.get("file_paths", []) if current_batch else []
        generation_mode = "FILE_MODE" if batch_files else "INJECTION_MODE"

        batch_description = current_batch.get("description", "") if current_batch else ""

        batch_instructions = ""
        if batch_files:
            batch_instructions = f"""

CURRENT BATCH: {batch_description}
Files to generate in this batch:
{chr(10).join(f"- {f}" for f in batch_files)}

Only generate code for the files listed above. Do NOT generate other files - they will be generated in subsequent batches."""

        return f"""GENERATION MODE: {generation_mode}

Generate Rust instruction implementations for:

Project folder name (crate name): {project_name}
Description: {spec.get("description", "N/A")}
Features: {spec.get("features", [])}

Instructions to implement: {spec.get("instructions", [])}
Accounts needed: {spec.get("accounts", [])}
Data structures: {spec.get("data_structs", [])}

Existing files:
{files_summary}
{batch_instructions}

CRITICAL: Use "{project_name}" as the #[program] module name - it MUST match the crate/folder name.

Write complete Rust code for the files requested. Split into proper files.
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
