from dataclasses import replace

from app.openai_client import (
    AgentExecutionContext,
    DEFAULT_MAX_AGENT_TURNS,
    StructuredModelClient,
)
from app.orchestration_state import assert_lease, requeue_if_new_messages
from app.comfyui import ComfyUIImageGenerationTool
from app.config import get_settings
from app.image_generation import OpenAIImageGenerationTool
from app.agent_schemas import ReviewGeneratedImageArguments
from app.tools import TavilySearchTool, ToolRegistration, ToolRegistry

model_client = StructuredModelClient()
tool_registry = ToolRegistry([TavilySearchTool().registration()])


def subtask_tool_registry(
    task_id,
    config,
    before_workspace_write,
    *,
    include_image_tool=True,
    image_review_handler=None,
    image_generation_guard=None,
):
    registrations = [TavilySearchTool().registration()]
    exposed_risk_classes = {"read_only"}
    image_provider = config.get("worker_image_provider", "comfyui")
    settings = get_settings()
    if (
        include_image_tool
        and image_provider == "comfyui"
        and settings.comfyui_enabled
    ):
        registrations.append(
            ComfyUIImageGenerationTool(
                task_id=task_id,
                checkpoint_name=config.get("worker_image_model", ""),
                before_workspace_write=before_workspace_write,
            ).registration()
        )
        exposed_risk_classes.add("writes_data")
    elif include_image_tool and image_provider == "openai":
        registrations.append(
            OpenAIImageGenerationTool(
                task_id=task_id,
                model=config.get("worker_image_model", ""),
                before_workspace_write=before_workspace_write,
            ).registration()
        )
        exposed_risk_classes.add("writes_data")
    if image_review_handler is not None:
        registrations.append(
            ToolRegistration(
                name="review_generated_image",
                version="1",
                description=(
                    "Ask the assigned image-review specialist to inspect a "
                    "generated image before returning it. Review only an "
                    "image_id and artifact_path returned by the image tool."
                ),
                risk_class="read_only",
                timeout_seconds=180,
                arguments_model=ReviewGeneratedImageArguments,
                handler=image_review_handler,
            )
        )
    if image_generation_guard is not None:
        guarded_registrations = []
        for registration in registrations:
            if registration.name not in {
                "comfyui_generate_image",
                "openai_generate_image",
            }:
                guarded_registrations.append(registration)
                continue
            original_handler = registration.handler

            async def guarded_handler(
                arguments, handler=original_handler
            ):
                error = await image_generation_guard(arguments)
                if error is not None:
                    return {"error": error}
                return await handler(arguments)

            guarded_registrations.append(
                replace(registration, handler=guarded_handler)
            )
        registrations = guarded_registrations
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
