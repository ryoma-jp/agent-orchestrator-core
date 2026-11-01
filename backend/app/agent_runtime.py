from app.openai_client import (
    AgentExecutionContext,
    DEFAULT_MAX_AGENT_TURNS,
    StructuredModelClient,
)
from app.orchestration_state import assert_lease, requeue_if_new_messages
from app.comfyui import ComfyUIImageGenerationTool
from app.config import get_settings
from app.tools import TavilySearchTool, ToolRegistry

model_client = StructuredModelClient()
tool_registry = ToolRegistry([TavilySearchTool().registration()])


def subtask_tool_registry(
    task_id,
    config,
    before_workspace_write,
    *,
    include_image_tool=True,
):
    registrations = [TavilySearchTool().registration()]
    exposed_risk_classes = {"read_only"}
    if (
        include_image_tool
        and get_settings().comfyui_enabled
        and config.get("worker_image_provider", "comfyui") == "comfyui"
    ):
        registrations.append(
            ComfyUIImageGenerationTool(
                task_id=task_id,
                checkpoint_name=config.get("worker_image_model", ""),
                before_workspace_write=before_workspace_write,
            ).registration()
        )
        exposed_risk_classes.add("writes_data")
    return ToolRegistry(
        registrations, exposed_risk_classes=exposed_risk_classes
    )


class InstructionChanged(RuntimeError):
    pass


def agent_options(
    task_id,
    job_id,
    lease_owner,
    config,
    *,
    plan_version=None,
    subtask_id=None,
    conversation_sequence=None,
):
    async def check_lease():
        await assert_lease(job_id, lease_owner)
        if task_id and conversation_sequence is not None:
            if await requeue_if_new_messages(
                task_id, job_id, lease_owner, conversation_sequence
            ):
                raise InstructionChanged("task_requeued")

    return {
        "before_model_call": check_lease,
        "max_turns": config.get("max_agent_turns", DEFAULT_MAX_AGENT_TURNS),
        "execution_context": AgentExecutionContext(
            task_id=str(task_id) if task_id else None,
            plan_version=plan_version,
            subtask_id=str(subtask_id) if subtask_id else None,
            conversation_sequence=conversation_sequence,
        ),
    }
