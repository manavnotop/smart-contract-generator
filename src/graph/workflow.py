"""LangGraph workflow for orchestrating contract generation pipeline."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langgraph.graph import END, StateGraph

from src.agents.code_generator import CodeGenerator
from src.agents.debugger import Debugger
from src.agents.project_planner import ProjectPlanner
from src.agents.spec_interpreter import SpecInterpreter
from src.schemas.models import MAX_RETRIES, GraphState
from src.utils.builder import Builder
from src.utils.file_ops import FileOps
from src.validators.static_validator import StaticValidator

# Set up logging
logger = logging.getLogger(__name__)


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
    workflow.add_node("code_generator", code_generator_node)
    workflow.add_node("static_validator", static_validator_node)
    workflow.add_node("build_contract", build_node)
    workflow.add_node("debugger", debugger_node)
    workflow.add_node("abort", abort_node)

    # Define edges - linear flow
    workflow.set_entry_point("spec_interpreter")
    workflow.add_edge("spec_interpreter", "project_planner")
    workflow.add_edge("project_planner", "code_generator")
    workflow.add_edge("code_generator", "static_validator")

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


async def code_generator_node(state: GraphState) -> GraphState:
    """Run code generator agent and write instruction files."""
    if state.on_event:
        state.on_event("agent:Code Generator:start")

    logger.info("[Code Generator] Starting...")
    try:
        agent = CodeGenerator()
        result_state = await agent.run(state.model_dump())

        # Write generated instruction files to disk
        files = result_state.get("files", {})
        if files:
            file_ops = FileOps(Path(state.project_root))
            file_ops.write_files(files)
            logger.info(f"[Code Generator] Wrote {len(files)} instruction files")

        logger.info("[Code Generator] Completed successfully")

        if state.on_event:
            state.on_event("agent:Code Generator:end")

        merged = _safe_merge(state, result_state)
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
