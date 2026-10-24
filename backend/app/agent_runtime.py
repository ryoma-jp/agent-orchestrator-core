from app.openai_client import (
    AgentExecutionContext,
    DEFAULT_MAX_AGENT_TURNS,
    StructuredModelClient,
)
from app.orchestration_state import assert_lease, requeue_if_new_messages
from app.tools import ToolRegistry, TavilySearchTool

model_client = StructuredModelClient()
tool_registry = ToolRegistry([TavilySearchTool().registration()])


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
