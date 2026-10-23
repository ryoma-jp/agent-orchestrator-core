import re
from typing import Any

from app.agent_schemas import ChecklistItem, ChecklistReviewResult

_MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\((https?://[^)\s]+)\)")


def normalize_checklist_results(
    checklist: list[ChecklistItem],
    review: ChecklistReviewResult,
    artifact: object,
    source_urls: set[str] | None = None,
) -> list[dict[str, Any]]:
    proposed = {result.item_id: result for result in review.results}
    allowed_sources = source_urls or _artifact_source_urls(artifact)
    artifact_text = _artifact_text(artifact)
    normalized = []

    for item in checklist:
        evidence = ""
        deficiency = ""
        requires_additional_tool_calls = False
        if item.verification_mode == "non_empty":
            passed = bool(artifact_text.strip())
            evidence = "Artifact contains text." if passed else ""
            deficiency = "The artifact is empty." if not passed else ""
        elif item.verification_mode == "source_present":
            passed = bool(allowed_sources)
            requires_additional_tool_calls = not passed
            evidence = (
                ", ".join(sorted(allowed_sources)[:5]) if passed else ""
            )
            deficiency = "No source URL was attached." if not passed else ""
        elif item.verification_mode == "markdown_citation":
            citations = set(_MARKDOWN_LINK.findall(artifact_text))
            verified = citations & allowed_sources
            passed = bool(verified)
            requires_additional_tool_calls = not passed and not allowed_sources
            evidence = ", ".join(sorted(verified)[:5])
            deficiency = (
                "No citation links to an attached source."
                if not passed
                else ""
            )
        elif item.verification_mode == "artifact_present":
            files = (
                artifact.get("artifacts", [])
                if isinstance(artifact, dict)
                else []
            )
            filenames = [
                file["filename"]
                for file in files
                if isinstance(file, dict)
                and isinstance(file.get("filename"), str)
                and isinstance(file.get("content"), str)
                and file["content"].strip()
            ] if isinstance(files, list) else []
            passed = bool(filenames)
            evidence = ", ".join(filenames[:5])
            if not passed:
                deficiency = "No non-empty requested file was generated."
        else:
            proposed_result = proposed.get(item.id)
            passed = bool(proposed_result and proposed_result.passed)
            if proposed_result is not None:
                evidence = proposed_result.evidence.strip()
                deficiency = proposed_result.deficiency.strip()
                requires_additional_tool_calls = (
                    proposed_result.requires_additional_tool_calls
                )
            if passed and not evidence:
                passed = False
                deficiency = "The review did not provide supporting evidence."
            elif not passed and not deficiency:
                deficiency = "The review did not confirm this criterion."

        normalized.append(
            {
                "item_id": item.id,
                "description": item.description,
                "required": item.required,
                "verification_mode": item.verification_mode,
                "passed": passed,
                "evidence": evidence,
                "deficiency": deficiency,
                "requires_additional_tool_calls": (
                    requires_additional_tool_calls
                ),
            }
        )
    return normalized


def required_checklist_items_passed(
    results: list[dict[str, Any]],
) -> bool:
    return all(
        result["passed"]
        for result in results
        if result.get("required", True)
    )


def failed_required_checklist_items(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        result
        for result in results
        if result.get("required", True) and not result["passed"]
    ]


def _artifact_source_urls(artifact: object) -> set[str]:
    if not isinstance(artifact, dict):
        return set()
    sources = artifact.get("sources")
    if not isinstance(sources, list):
        return set()
    return {
        source["url"]
        for source in sources
        if isinstance(source, dict)
        and isinstance(source.get("url"), str)
        and source["url"].startswith(("https://", "http://"))
    }


def _artifact_text(artifact: object) -> str:
    if isinstance(artifact, str):
        return artifact
    if not isinstance(artifact, dict):
        return ""
    parts = [artifact.get("answer"), artifact.get("summary")]
    findings = artifact.get("findings")
    if isinstance(findings, list):
        parts.extend(findings)
    artifacts = artifact.get("artifacts")
    if isinstance(artifacts, list):
        parts.extend(
            item["content"]
            for item in artifacts
            if isinstance(item, dict)
            and isinstance(item.get("content"), str)
        )
    text_parts = [part for part in parts if isinstance(part, str)]
    return "\n".join(text_parts)
