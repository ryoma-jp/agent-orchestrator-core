import re
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)


class PlannedSubtask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=300)
    instruction: str = Field(min_length=1, max_length=12_000)
    artifact_goal: str = Field(min_length=1, max_length=2_000)
    acceptance_checklist: list["ChecklistItem"] = Field(
        min_length=1, max_length=20
    )
    dependencies: list[int] = Field(max_length=30)
    skill_names: list[str] = Field(max_length=2)

    @field_validator("skill_names")
    @classmethod
    def validate_skill_names(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("skill_names must be unique")
        if any(
            re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) is None
            for name in value
        ):
            raise ValueError("skill name is invalid")
        return value

    @model_validator(mode="after")
    def validate_checklist_ids(self):
        _validate_checklist_ids(self.acceptance_checklist)
        return self


class TaskPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=100)
    summary: str = Field(min_length=1, max_length=2_000)
    final_checklist: list["ChecklistItem"] = Field(
        min_length=1, max_length=20
    )
    subtasks: list[PlannedSubtask] = Field(min_length=1, max_length=30)

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        title = value.strip()
        if not title:
            raise ValueError("title must not be blank")
        return title

    @model_validator(mode="after")
    def validate_dependencies(self):
        _validate_checklist_ids(self.final_checklist)
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


class TaskTitle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=100)

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        title = value.strip()
        if not title:
            raise ValueError("title must not be blank")
        return title


class ChecklistItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=500)
    required: bool = True
    verification_mode: Literal[
        "model",
        "non_empty",
        "source_present",
        "markdown_citation",
        "artifact_present",
    ] = "model"


def _validate_checklist_ids(items: list[ChecklistItem]) -> None:
    identifiers = [item.id for item in items]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("checklist item ids must be unique")


class ChecklistItemResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=1, max_length=100)
    passed: bool
    evidence: str = Field(max_length=2_000)
    deficiency: str = Field(max_length=1_000)
    requires_additional_tool_calls: bool = False
    revision_target: Literal["answer", "image", "both"] = "answer"
    target_image_paths: list[str] = Field(
        default_factory=list, max_length=5
    )


class ChecklistReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[ChecklistItemResult] = Field(max_length=21)


class ImageGenerationPrompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    positive_prompt: str = Field(min_length=1, max_length=4_000)
    negative_prompt: str = Field(max_length=2_000)

    @field_validator("positive_prompt")
    @classmethod
    def validate_positive_prompt(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("image_positive_prompt_empty")
        return value

    @field_validator("negative_prompt")
    @classmethod
    def normalize_negative_prompt(cls, value: str) -> str:
        return value.strip()


class ImageGenerationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_path: str = Field(min_length=1, max_length=300)
    prompt: str = Field(min_length=1, max_length=4_000)
    negative_prompt: str = Field(max_length=2_000)
    width: int = Field(ge=256, le=3840)
    height: int = Field(ge=256, le=3840)
    steps: int | None = Field(default=None, ge=1, le=60)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    quality: str | None = Field(default=None, max_length=40)
    openai_prompt: str | None = Field(default=None, max_length=6_100)


class SubtaskResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    _image_generations: list[ImageGenerationRecord] = PrivateAttr(
        default_factory=list
    )

    status: Literal["succeeded", "partial", "needs_input", "failed"]
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
    content: str = Field(max_length=10_000)
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
    artifact_paths: list[str] = Field(default_factory=list, max_length=10)


class GeneratedArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1, max_length=200_000)

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        has_separator = any(
            separator in value for separator in ("/", "\\", "\0")
        )
        has_control_character = any(
            ord(character) < 32 or ord(character) == 127
            for character in value
        )
        if value in {".", ".."} or has_separator or has_control_character:
            raise ValueError("filename must be a single path component")
        return value


class IntegratedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, max_length=20_000)
    caveats: list[str] = Field(max_length=20)
    artifacts: list[GeneratedArtifact] = Field(max_length=10)

    @model_validator(mode="after")
    def validate_unique_artifact_filenames(self):
        filenames = [item.filename.casefold() for item in self.artifacts]
        if len(set(filenames)) != len(filenames):
            raise ValueError("artifact filenames must be unique")
        return self
