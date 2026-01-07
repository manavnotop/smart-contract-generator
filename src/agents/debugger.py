"""Debugger Agent - analyzes errors and generates fixes using LangChain."""

from src.agents.base import BaseAgent

SYSTEM_PROMPT = """You are an expert Solana smart contract debugger. Your job is to
analyze build/validation errors and generate precise fixes.

For each error:
1. Identify the root cause
2. Generate minimal, targeted fixes
3. Ensure fixes don't break other functionality

Output format - return a JSON object with patches to apply:

```json
{
    "patches": [
        {"path": "src/lib.rs", "content": "..."}
    ],
    "analysis": "Explanation of what was wrong and how it was fixed"
}
```

Guidelines:
- Use relative paths from project root (e.g., "src/lib.rs", "src/instructions/mint.rs")
- Generate complete file content for patches, not diffs
- Focus on minimal changes that fix the specific error
- Preserve existing code where possible
- Add missing imports/dependencies
- Fix syntax errors, type mismatches, etc.

The file contents are provided in the prompt below - review them carefully to understand the current code before generating fixes.
"""


class Debugger(BaseAgent):
    """Agent that debugs and fixes contract code using LangChain."""

    @property
    def agent_name(self):
        return "debugger"

    def _get_tools(self):
        """No tools needed - file contents are passed in the state."""
        return []

    def _get_system_prompt(self):
        return SYSTEM_PROMPT

    def _format_state_for_agent(self, state: dict) -> str:
        """Format the error info for the agent."""
        error_info = ""

        if state.get("validation_errors"):
            error_info += "Validation errors:\n" + "\n".join(state["validation_errors"])
        if state.get("build_logs"):
            error_info += "\n\nBuild logs:\n" + state["build_logs"]
        if state.get("error_message"):
            error_info += "\n\nError: " + state["error_message"]

        if not error_info:
            error_info = "Unknown error - no error information available"

        files = state.get("files", {})

        # Include file contents so the agent can see the code
        files_content = ""
        for path, content in files.items():
            files_content += f"\n=== {path} ===\n{content}"

        return f"""Analyze and fix these errors:

{error_info}

Current project files with contents:
{files_content}

Use the file reading tools to examine the code if needed. Return patches to fix the issues as a JSON object."""

    def _format_agent_result(self, state: dict, result: dict) -> dict:
        """Extract patches and return them in state."""
        output = self._extract_output_from_result(result)
        data = self._extract_json_from_output(output)

        # Extract analysis for logging
        analysis = data.get("analysis", "No analysis provided")

        # Try 'patches' key first, then fallback to general file map
        patches = data.get("patches", [])
        if not patches and "files" in data:
            patches = [{"path": p, "content": c} for p, c in data["files"].items()]

        if patches:
            updated_files = state.get("files", {}.copy())
            for patch in patches:
                path = patch.get("path")
                content = patch.get("content")
                if path and content:
                    updated_files[path] = content

            return {
                **state,
                "files": updated_files,
                "debugger_analysis": analysis,
                "debugger_patches_count": len(patches),
                "error_message": None,
                "current_step": "static_validator",
            }

        return {
            **state,
            "debugger_analysis": analysis,
            "current_step": "abort",
            "error_message": f"Debugger failed: {output}",
        }
