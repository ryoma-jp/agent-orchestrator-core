from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PlannedSubtask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=300)
    instruction: str = Field(min_length=1, max_length=12_000)
    dependencies: list[int] = Field(max_length=30)
    skill_names: list[str] = Field(max_length=2)


class TaskPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=2_000)
    completion_criteria: list[str] = Field(min_length=1, max_length=10)
    subtasks: list[PlannedSubtask] = Field(min_length=1, max_length=30)

    @model_validator(mode="after")
    def validate_dependencies(self):
        count = len(self.subtasks)
        for index, subtask in enumerate(self.subtasks):
            if len(set(subtask.dependencies)) != len(subtask.dependencies):
                raise ValueError("subtask dependencies must be unique")
            if any(
                dependency < 0 or dependency >= count or dependency == index
                for dependency in subtask.dependencies
            ):
                raise ValueError("subtask dependency index is invalid")

        visiting: set[int] = set()
        visited: set[int] = set()

        def visit(index: int) -> None:
            if index in visiting:
                raise ValueError("subtask dependency graph contains a cycle")
            if index in visited:
                return
            visiting.add(index)
            for dependency in self.subtasks[index].dependencies:
                visit(dependency)
            visiting.remove(index)
            visited.add(index)

        for index in range(count):
            visit(index)
        return self


class SubtaskResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["succeeded", "needs_input", "failed"]
    summary: str = Field(max_length=4_000)
    findings: list[str] = Field(max_length=30)
    sources: list["SourceReference"] = Field(max_length=15)
    tool_runs: list["ToolExecutionRecord"] = Field(max_length=10)
    questions: list[str] = Field(max_length=10)
    error: str

    @model_validator(mode="after")
    def validate_questions(self):
        if self.status == "needs_input":
            if not any(question.strip() for question in self.questions):
                raise ValueError("needs_input requires a question")
        elif self.questions:
            raise ValueError("questions are only valid for needs_input")
        return self


class SourceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=500)
    url: str = Field(min_length=1, max_length=2_048)
    content: str = Field(max_length=1_500)
    score: float | None
    published_date: str | None
    retrieved_at: str = Field(min_length=1, max_length=64)


class ToolExecutionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(min_length=1, max_length=100)
    request_summary: str = Field(max_length=1_000)
    status: Literal["succeeded", "failed"]
    source_count: int = Field(ge=0, le=15)
    error_code: str = Field(max_length=100)


class IntegratedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, max_length=20_000)
    completion_met: bool
    caveats: list[str] = Field(max_length=20)
