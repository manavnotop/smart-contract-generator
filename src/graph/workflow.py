"""LangGraph workflow for orchestrating contract generation pipeline."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langgraph.graph import END, StateGraph

from src.agents.code_generator import CodeGenerator
from src.agents.debugger import Debugger
from src.agents.file_planner import FilePlanner
from src.agents.project_planner import ProjectPlanner
from src.agents.spec_interpreter import SpecInterpreter
from src.schemas.models import MAX_RETRIES, GraphState
from src.utils.builder import Builder
from src.utils.file_ops import FileOps
from src.validators.static_validator import StaticValidator

# Set up logging
logger = logging.getLogger(__name__)


def _preserve_declare_id(project_root: Path, new_content: str) -> str:
    """Preserve existing declare_id! when writing lib.rs.

    If lib.rs already exists with a valid declare_id!, replace the generated
    declare_id! line with the existing one to preserve the program ID from anchor init.
    """
    lib_path = project_root / "programs" / "*" / "src" / "lib.rs"
    # Try to find the actual lib.rs path
    programs_dir = project_root / "programs"
    if programs_dir.exists():
        for child in programs_dir.iterdir():
            if child.is_dir():
                lib_path = child / "src" / "lib.rs"
                break

    if not lib_path.exists():
        return new_content

    # Read existing file to get declare_id
    try:
        existing_content = lib_path.read_text()
        # Find the declare_id line
        for line in existing_content.split("\n"):
            if line.strip().startswith("declare_id!"):
                # Replace declare_id in new content with existing one
                lines = new_content.split("\n")
                new_lines = []
                for l in lines:
                    if l.strip().startswith("declare_id!"):
                        new_lines.append(line)
                    else:
                        new_lines.append(l)
                return "\n".join(new_lines)
    except Exception:
        pass

    return new_content


def create_workflow(test_mode: bool = False) -> StateGraph:
    """Create the LangGraph workflow.

    Args:
        test_mode: If True, use mock LLM for testing

    Returns:
        Configured StateGraph ready to be compiled
    """
    workflow = StateGraph(GraphState)

    # Define nodes
    workflow.add_node("spec_interpreter", spec_interpreter_node)
    workflow.add_node("project_planner", project_planner_node)
    workflow.add_node("file_planner", file_planner_node)
    workflow.add_node("batch_processor", batch_processor_node)
    workflow.add_node("static_validator", static_validator_node)
    workflow.add_node("build_contract", build_node)
    workflow.add_node("debugger", debugger_node)
    workflow.add_node("abort", abort_node)

    # Define edges - linear flow
    workflow.set_entry_point("spec_interpreter")
    workflow.add_edge("spec_interpreter", "project_planner")
    workflow.add_edge("project_planner", "file_planner")
    workflow.add_edge("file_planner", "batch_processor")

    # Conditional edge from batch_processor - loop until all batches done
    workflow.add_conditional_edges(
        "batch_processor",
        has_more_batches,
        {
            "continue": "batch_processor",
            "done": "static_validator",
        },
    )

    # Conditional edge from static_validator
    workflow.add_conditional_edges(
        "static_validator",
        should_proceed_to_build,
        {
            "build": "build_contract",
            "debugger": "debugger",
            "abort": "abort",
        },
    )

    # Conditional edge from build_contract
    workflow.add_conditional_edges(
        "build_contract",
        should_proceed_to_end,
        {
            "end": END,
            "debugger": "debugger",
            "abort": "abort",
        },
    )

    # Edge from debugger - after fix, go directly to build (skip static validation)
    workflow.add_edge("debugger", "build_contract")

    return workflow


def should_proceed_to_build(state: GraphState) -> str:
    """Determine if we should proceed to build or go to debugger."""
    if state.validation_passed:
        return "build"
    elif state.retry_count < MAX_RETRIES:
        return "debugger"
    else:
        return "abort"


def should_proceed_to_end(state: GraphState) -> str:
    """Determine if build succeeded or we need debugging."""
    if state.build_success:
        return "end"
    elif state.retry_count < MAX_RETRIES:
        return "debugger"
    else:
        return "abort"


def has_more_batches(state: GraphState) -> str:
    """Determine if there are more batches to process.

    Returns 'continue' if we should process more batches, 'done' if finished.
    In legacy mode (generation_plan is None), always returns 'done'.
    """
    # Legacy mode - skip batch processing
    if not state.generation_plan:
        return "done"

    # Check if there are pending files to process
    if state.pending_files and len(state.pending_files) > 0:
        return "continue"
    return "done"


def _safe_merge(state: GraphState, agent_result: dict[str, Any]) -> dict[str, Any]:
    """Safely merge agent result with state, excluding control fields.

    Control fields (user_spec, retry_count, project_root, current_step, etc.)
    must NEVER come from agents to prevent LLM from corrupting workflow state.
    These fields are excluded from BOTH sources to prevent duplicate keyword errors.
    """
    control_fields = {
        "user_spec",
        "project_name",
        "retry_count",
        "project_root",
        "current_step",
        "validation_passed",
        "build_success",
        "final_artifact",
        "on_event",
        "test_mode",
    }

    # Build merged dict excluding control fields from BOTH sources
    merged = {}
    for key, value in state.model_dump().items():
        if key not in control_fields:
            merged[key] = value
    for key, value in agent_result.items():
        if key not in control_fields:
            merged[key] = value

    return merged


async def _run_agent_node(state: GraphState, agent_class: type, agent_name: str) -> GraphState:
    """Shared helper to run an agent node with logging and event handling."""
    if state.on_event:
        state.on_event(f"agent:{agent_name}:start")

    logger.info(f"[{agent_name}] Starting...")
    try:
        # Check if agent class accepts test_mode
        # (only SpecInterpreter currently does explicitly in __init__)
        if agent_name == "Spec Interpreter":
            agent = agent_class(test_mode=state.test_mode)
        else:
            agent = agent_class()

        result_state = await agent.run(state.model_dump())
        logger.info(f"[{agent_name}] Completed successfully")

        if state.on_event:
            state.on_event(f"agent:{agent_name}:end")

        # Merge non-control fields
        merged = _safe_merge(state, result_state)

        # Increment retry count if this is the debugger
        new_retry_count = state.retry_count + (1 if agent_name == "Debugger" else 0)

        return GraphState(
            **merged,
            user_spec=state.user_spec,
            retry_count=new_retry_count,
            project_root=state.project_root or result_state.get("project_root"),
            on_event=state.on_event,
            test_mode=state.test_mode,
        )
    except Exception as e:
        logger.error(f"[{agent_name}] FAILED: {e}")
        if state.on_event:
            state.on_event(f"agent:{agent_name}:failed")
        raise


async def spec_interpreter_node(state: GraphState) -> GraphState:
    return await _run_agent_node(state, SpecInterpreter, "Spec Interpreter")


async def project_planner_node(state: GraphState) -> GraphState:
    """Run project planner agent and handle file writing."""
    if state.on_event:
        state.on_event("agent:Project Planner:start")

    logger.info("[Project Planner] Starting...")
    try:
        # Get project name from state (provided by CLI)
        project_name = state.project_name
        if not project_name:
            project_name = "my_contract"

        # Create contracts directory
        contracts_dir = Path.cwd() / "contracts"
        contracts_dir.mkdir(parents=True, exist_ok=True)

        # Create project directory and run anchor init
        project_root = contracts_dir / project_name
        builder = Builder(contracts_dir)

        logger.info(f"[Project Planner] Running anchor init {project_name}...")
        success, output = builder.anchor_init(project_name)

        if not success:
            logger.warning(f"[Project Planner] anchor init failed: {output}")
            # Continue anyway - maybe the directory already exists

        # Now run the ProjectPlanner agent
        agent = ProjectPlanner()
        result_state = await agent.run(state.model_dump())

        # Write program files
        files = result_state.get("files", {})
        if files:
            file_ops = FileOps(project_root)

            for path, _content in files.items():
                if state.on_event:
                    state.on_event(f"file:write:{project_name}/{path}")

            file_ops.write_files(files)
            logger.info(f"[Project Planner] Wrote {len(files)} program files")

        logger.info(f"[Project Planner] Completed: {project_name}")

        if state.on_event:
            state.on_event("agent:Project Planner:end")

        merged = _safe_merge(state, result_state)
        return GraphState(
            **merged,
            user_spec=state.user_spec,
            retry_count=state.retry_count,
            project_name=project_name,
            project_root=str(project_root),
            on_event=state.on_event,
            test_mode=state.test_mode,
        )
    except Exception as e:
        logger.error(f"[Project Planner] FAILED: {e}")
        raise


async def file_planner_node(state: GraphState) -> GraphState:
    """Run FilePlanner to create generation plan with parallel batches."""
    if state.on_event:
        state.on_event("agent:File Planner:start")

    logger.info("[File Planner] Starting...")
    try:
        agent = FilePlanner()
        result_state = await agent.run(state.model_dump())

        # Get the generation plan (may be a Pydantic model or dict)
        import sys

        plan_dict = result_state.get("generation_plan")
        print(f"[FilePlanner DEBUG] plan_dict type: {type(plan_dict)}", file=sys.stderr)

        # Convert Pydantic model to dict if needed
        if plan_dict and hasattr(plan_dict, "model_dump"):
            plan_dict = plan_dict.model_dump()
            print(f"[FilePlanner DEBUG] Converted to dict: {plan_dict}", file=sys.stderr)

        if plan_dict and isinstance(plan_dict, dict) and plan_dict.get("batches"):
            # Initialize pending files tracking
            all_files = set()
            batches = plan_dict["batches"]
            for batch in batches:
                all_files.update(batch.get("file_paths", []))

            # Calculate total files
            total_files = sum(len(batch.get("file_paths", [])) for batch in batches)

            logger.info(
                f"[File Planner] Created plan with {len(batches)} batches, {total_files} files total"
            )

            if state.on_event:
                for batch in batches:
                    state.on_event(
                        f"batch:planned:{batch.get('batch_id')}:{batch.get('description', '')}"
                    )
                state.on_event("agent:File Planner:end")

            merged = _safe_merge(state, result_state)
            # Remove incremental tracking fields from merged to avoid duplicate kwargs
            merged.pop("generation_plan", None)
            merged.pop("pending_files", None)
            merged.pop("generated_files", None)
            merged.pop("file_progress", None)
            return GraphState(
                **merged,
                user_spec=state.user_spec,
                retry_count=state.retry_count,
                project_root=state.project_root,
                on_event=state.on_event,
                test_mode=state.test_mode,
                # Initialize incremental tracking
                generation_plan=plan_dict,  # Already a dict
                pending_files={f: "" for f in all_files},  # Empty content placeholders
                generated_files={},
                file_progress=(0, total_files),
            )
        else:
            # Fallback: no plan generated, use legacy behavior
            logger.warning("[File Planner] No plan generated, falling back to legacy mode")
            if state.on_event:
                state.on_event("agent:File Planner:end")
            merged = _safe_merge(state, result_state)
            # Remove generation_plan from merged to avoid duplicate kwarg
            merged.pop("generation_plan", None)
            merged.pop("pending_files", None)
            merged.pop("generated_files", None)
            merged.pop("file_progress", None)
            return GraphState(
                **merged,
                user_spec=state.user_spec,
                retry_count=state.retry_count,
                project_root=state.project_root,
                on_event=state.on_event,
                test_mode=state.test_mode,
                generation_plan=None,  # Will trigger legacy behavior
            )

    except Exception as e:
        logger.error(f"[File Planner] FAILED: {e}")
        if state.on_event:
            state.on_event("agent:File Planner:failed")
        raise


async def batch_processor_node(state: GraphState) -> GraphState:
    """Process the next batch of files to generate.

    If no generation plan exists, falls back to legacy code generator behavior.
    """
    # Check for fallback to legacy behavior
    plan_dict = state.generation_plan
    if not plan_dict or not isinstance(plan_dict, dict) or not plan_dict.get("batches"):
        return await _legacy_code_generator_node(state)

    if state.on_event:
        state.on_event("agent:Code Generator:start")

    logger.info("[Code Generator] Starting batch processing...")

    try:
        plan = plan_dict
        batches = plan.get("batches", [])
        completed = set(state.generated_files.keys())
        pending = set(state.pending_files.keys())

        # Find next ready batch (all dependencies satisfied)
        batch_to_process = None
        generation_order = plan.get("generation_order", [])
        for step in generation_order:
            for batch_id in step:
                batch = next((b for b in batches if b.get("batch_id") == batch_id), None)
                if not batch:
                    continue

                # Check if all dependencies are satisfied
                deps_satisfied = True
                for dep_id in batch.get("dependencies", []):
                    dep_batch = next((b for b in batches if b.get("batch_id") == dep_id), None)
                    if dep_batch:
                        # Check if all files in dependency batch are generated
                        for dep_file in dep_batch.get("file_paths", []):
                            if dep_file not in completed:
                                deps_satisfied = False
                                break
                    if not deps_satisfied:
                        break

                # Check if this batch has pending files
                has_pending = any(f in pending for f in batch.get("file_paths", []))

                if deps_satisfied and has_pending:
                    batch_to_process = batch
                    break
            if batch_to_process:
                break

        if not batch_to_process:
            # All batches processed - move to validation
            logger.info("[Code Generator] All batches complete")
            if state.on_event:
                state.on_event("agent:Code Generator:end")

            return GraphState(
                **state.model_dump(),
                pending_files={},
                user_spec=state.user_spec,
                retry_count=state.retry_count,
                project_root=state.project_root,
                on_event=state.on_event,
                test_mode=state.test_mode,
            )

        # Process the batch
        if state.on_event:
            state.on_event(f"batch:start:{batch_to_process.get('batch_id')}")
            for path in batch_to_process.get("file_paths", []):
                if path in pending:
                    state.on_event(f"file:generating:{path}")

        # Run code generator for this batch
        batch_state = {
            **state.model_dump(),
            "current_batch": batch_to_process,
        }

        agent = CodeGenerator()
        result_state = await agent.run(batch_state)

        # Get generated files - filter to only include files from current batch
        all_new_files = result_state.get("files", {})
        batch_files = batch_to_process.get("file_paths", [])
        new_files = {
            path: content for path, content in all_new_files.items() if path in batch_files
        }

        # Write files to disk and emit events
        if new_files and state.project_root:
            file_ops = FileOps(Path(state.project_root))
            for path, content in new_files.items():
                # Preserve declare_id when writing lib.rs
                if "lib.rs" in path and state.project_root:
                    content = _preserve_declare_id(Path(state.project_root), content)
                file_ops.write_file(path, content)
                if state.on_event:
                    state.on_event(f"file:created:{path}:{len(content)}")

        logger.info(
            f"[Code Generator] Generated {len(new_files)} files in batch {batch_to_process.get('batch_id')}"
        )

        # Update progress
        updated_generated = {**state.generated_files, **new_files}
        updated_pending = {k: v for k, v in state.pending_files.items() if k not in new_files}
        progress = (len(updated_generated), state.file_progress[1])

        if state.on_event:
            state.on_event(f"batch:end:{batch_to_process.get('batch_id')}")

        merged = _safe_merge(state, result_state)
        # Remove any duplicate keyword args that are already in merged
        merged.pop("generated_files", None)
        merged.pop("pending_files", None)
        merged.pop("file_progress", None)
        return GraphState(
            **merged,
            generated_files=updated_generated,
            pending_files=updated_pending,
            file_progress=progress,
            user_spec=state.user_spec,
            retry_count=state.retry_count,
            project_root=state.project_root,
            on_event=state.on_event,
            test_mode=state.test_mode,
        )

    except Exception as e:
        logger.error(f"[Code Generator] FAILED: {e}")
        if state.on_event:
            state.on_event("agent:Code Generator:failed")
        raise


async def _legacy_code_generator_node(state: GraphState) -> GraphState:
    """Legacy code generator - generates all files at once (fallback)."""
    if state.on_event:
        state.on_event("agent:Code Generator:start")

    logger.info("[Code Generator] Starting (legacy mode)...")
    try:
        agent = CodeGenerator()
        result_state = await agent.run(state.model_dump())

        # Write generated instruction files to disk
        files = result_state.get("files", {})
        if files:
            file_ops = FileOps(Path(state.project_root))
            for path, content in files.items():
                # Preserve declare_id when writing lib.rs
                if "lib.rs" in path and state.project_root:
                    content = _preserve_declare_id(Path(state.project_root), content)
                file_ops.write_file(path, content)
                if state.on_event:
                    state.on_event(f"file:created:{path}:{len(content)}")

        logger.info(f"[Code Generator] Wrote {len(files)} instruction files (legacy mode)")

        if state.on_event:
            state.on_event("agent:Code Generator:end")

        merged = _safe_merge(state, result_state)
        # Clear incremental tracking fields in legacy mode
        merged.pop("pending_files", None)
        merged.pop("generated_files", None)
        merged.pop("file_progress", None)
        return GraphState(
            **merged,
            user_spec=state.user_spec,
            retry_count=state.retry_count,
            project_root=state.project_root,
            on_event=state.on_event,
            test_mode=state.test_mode,
        )
    except Exception as e:
        logger.error(f"[Code Generator] FAILED: {e}")
        if state.on_event:
            state.on_event("agent:Code Generator:failed")
        raise


async def static_validator_node(state: GraphState) -> GraphState:
    """Run static validation."""
    if state.on_event:
        state.on_event("validation:start")

    logger.info("[Static Validator] Starting...")
    try:
        if not state.project_root:
            raise ValueError("project_root is None")

        validator = StaticValidator(Path(state.project_root))
        result = await validator.run(state.model_dump())

        if state.on_event:
            state.on_event(f"validation:{'success' if result['validation_passed'] else 'failed'}")

        merged = _safe_merge(state, result)
        return GraphState(
            **merged,
            user_spec=state.user_spec,
            retry_count=state.retry_count,
            project_root=state.project_root,
            on_event=state.on_event,
            test_mode=state.test_mode,
        )
    except Exception as e:
        logger.error(f"[Static Validator] FAILED: {e}")
        raise


async def build_node(state: GraphState) -> GraphState:
    """Build the contract."""
    if state.on_event:
        state.on_event("build:start")

    logger.info("[Build Contract] Starting...")
    try:
        if not state.project_root:
            raise ValueError("project_root is None")

        builder = Builder(Path(state.project_root))
        success, output = builder.verify_build()
        artifact = builder.get_build_artifact()

        if state.on_event:
            state.on_event(f"build:{'success' if success else 'failed'}")

        merged = _safe_merge(state, {})
        merged.pop("build_logs", None)  # Remove build_logs to avoid duplicate keyword arg
        return GraphState(
            **merged,
            user_spec=state.user_spec,
            retry_count=state.retry_count,
            project_root=state.project_root,
            build_success=success,
            build_logs=output,
            final_artifact=str(artifact) if artifact else None,
            on_event=state.on_event,
            test_mode=state.test_mode,
        )
    except Exception as e:
        logger.error(f"[Build Contract] FAILED: {e}")
        raise


async def debugger_node(state: GraphState) -> GraphState:
    """Run debugger agent with detailed logging."""
    if state.on_event:
        state.on_event("agent:Debugger:start")

    logger.info("[Debugger] Starting...")

    try:
        agent = Debugger()
        result_state = await agent.run(state.model_dump())

        # Write patched files to disk
        files = result_state.get("files", {})
        if files:
            file_ops = FileOps(Path(state.project_root))
            file_ops.write_files(files)
            logger.info(f"[Debugger] Wrote {len(files)} patched files to disk")

        # Log debugger activity
        if result_state.get("error_message"):
            logger.warning(f"[Debugger] Failed: {result_state['error_message']}")
        else:
            files = result_state.get("files", {})
            changed_files = set(files.keys())
            patches_count = result_state.get("debugger_patches_count", len(changed_files))
            logger.info(f"[Debugger] Applied {patches_count} patches to fix issues")

            # Log analysis
            analysis = result_state.get("debugger_analysis", "")
            if analysis and analysis != "No analysis provided":
                logger.info(f"[Debugger] Analysis: {analysis[:200]}...")

            # Log what was fixed
            logger.info(f"[Debugger] Files modified: {list(changed_files)}")

        if state.on_event:
            state.on_event("agent:Debugger:end")

        merged = _safe_merge(state, result_state)
        return GraphState(
            **merged,
            user_spec=state.user_spec,
            retry_count=state.retry_count + 1,
            project_root=state.project_root,
            on_event=state.on_event,
            test_mode=state.test_mode,
        )
    except Exception as e:
        logger.error(f"[Debugger] FAILED: {e}")
        if state.on_event:
            state.on_event("agent:Debugger:failed")
        raise


async def abort_node(state: GraphState) -> GraphState:
    """Handle abort - final error state."""
    logger.warning(f"[Abort] Workflow aborted. Error: {state.error_message}")
    return state


async def run_workflow(
    user_spec: str,
    on_event: Callable[[str], None] | None = None,
    test_mode: bool = False,
    project_name: str | None = None,
) -> GraphState:
    """Run the complete workflow.

    Args:
        user_spec: Natural language specification
        on_event: Optional callback for progress events
        test_mode: If True, use mock LLM for testing
        project_name: Project name for anchor init

    Returns:
        Final workflow state
    """
    if on_event:
        on_event("workflow:start")
    logger.info("=" * 60)
    logger.info("WORKFLOW STARTED")
    logger.info(f"User specification: {user_spec[:100]}...")
    logger.info(f"Project name: {project_name}")
    logger.info(f"Test mode: {test_mode}")
    logger.info("=" * 60)

    app = create_workflow(test_mode=test_mode).compile()

    initial_state = GraphState(
        user_spec=user_spec,
        test_mode=test_mode,
        project_name=project_name,
        on_event=on_event,
    )

    try:
        result = await app.ainvoke(initial_state)
        logger.info("=" * 60)
        logger.info("WORKFLOW COMPLETED")
        logger.info(f"Build success: {result.get('build_success')}")
        logger.info(f"Project root: {result.get('project_root')}")
        logger.info("=" * 60)
        if on_event:
            on_event("workflow:end")
    except Exception as e:
        logger.error("=" * 60)
        logger.error("WORKFLOW FAILED WITH EXCEPTION")
        logger.error(f"Error: {e}")
        logger.error("=" * 60)
        if on_event:
            on_event("workflow:failed")
        raise

    return GraphState(**result)
