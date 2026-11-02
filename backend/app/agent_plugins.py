"""Validated loader for private specialist Agent definitions."""

import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from app.config import get_settings
from app.tools import list_private_skills, load_private_skill

_AGENT_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_SKILL_LINK = re.compile(
    r"\]\((?:\.\./)?skills/([a-z0-9]+(?:-[a-z0-9]+)*)/SKILL\.md\)"
)
_MAX_AGENT_BYTES = 100_000


@dataclass(frozen=True)
class AgentDefinition:
    id: str
    name: str
    description: str
    instructions: str
    allowed_agents: tuple[str, ...]
    skill_names: tuple[str, ...]
    digest: str
    legacy_digest: str | None = None

    def catalog_entry(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "referenced_agents": list(self.allowed_agents),
            "skills": list(self.skill_names),
        }


def _plugin_root() -> Path | None:
    configured_path = getattr(get_settings(), "agent_plugin_path", None)
    if not configured_path:
        return None
    configured = Path(configured_path).expanduser()
    try:
        if configured.is_symlink():
            raise RuntimeError("agent_plugin_invalid")
        root = configured.resolve(strict=True)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RuntimeError("agent_plugin_unavailable") from exc
    if not root.is_dir():
        raise RuntimeError("agent_plugin_invalid")
    return root


def _parse_agent(path: Path) -> dict[str, object]:
    try:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("agent_definition_invalid")
        raw = path.read_bytes()
    except OSError as exc:
        raise RuntimeError("agent_definition_unavailable") from exc
    if len(raw) > _MAX_AGENT_BYTES:
        raise RuntimeError("agent_definition_invalid")
    try:
        content = raw.decode("utf-8")
    except UnicodeError as exc:
        raise RuntimeError("agent_definition_invalid") from exc
    match = re.match(
        r"\A---\r?\n(?P<metadata>.*?)\r?\n---\r?\n(?P<body>.*)\Z",
        content,
        flags=re.DOTALL,
    )
    if match is None:
        raise RuntimeError("agent_definition_invalid")
    try:
        metadata = yaml.safe_load(match.group("metadata"))
    except yaml.YAMLError as exc:
        raise RuntimeError("agent_definition_invalid") from exc
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str) for key in metadata
    ):
        raise RuntimeError("agent_definition_invalid")

    agent_id = path.name.removesuffix(".agent.md")
    name = metadata.get("name", agent_id)
    description = metadata.get("description")
    allowed_names = metadata.get("agents", [])
    declared_skills = metadata.get("skills", [])
    body = match.group("body").strip()
    if (
        _AGENT_ID.fullmatch(agent_id) is None
        or len(agent_id) > 100
        or not isinstance(name, str)
        or not name.strip()
        or len(name) > 200
        or not isinstance(description, str)
        or not description.strip()
        or len(description) > 1_000
        or not isinstance(allowed_names, list)
        or len(allowed_names) > 10
        or any(
            not isinstance(value, str) or not value.strip()
            for value in allowed_names
        )
        or len(set(allowed_names)) != len(allowed_names)
        or not isinstance(declared_skills, list)
        or any(not isinstance(value, str) for value in declared_skills)
        or not body
    ):
        raise RuntimeError("agent_definition_invalid")
    linked_skills = _SKILL_LINK.findall(body)
    skill_names = tuple(dict.fromkeys([*declared_skills, *linked_skills]))
    if any(_AGENT_ID.fullmatch(skill) is None for skill in skill_names):
        raise RuntimeError("agent_definition_invalid")
    available_skills = {skill["name"] for skill in list_private_skills()}
    if any(skill not in available_skills for skill in skill_names):
        raise RuntimeError("agent_definition_skill_unavailable")
    return {
        "id": agent_id,
        "name": name.strip(),
        "description": description.strip(),
        "instructions": body,
        "allowed_names": tuple(allowed_names),
        "skill_names": skill_names,
        "digest": hashlib.sha256(raw).hexdigest(),
    }


def load_agent_catalog() -> dict[str, AgentDefinition]:
    root = _plugin_root()
    if root is None:
        return {}
    agents_path = root / "agents"
    if not agents_path.exists():
        return {}
    if agents_path.is_symlink() or not agents_path.is_dir():
        raise RuntimeError("agent_catalog_invalid")
    try:
        files = sorted(agents_path.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        raise RuntimeError("agent_catalog_unavailable") from exc
    parsed = [
        _parse_agent(path) for path in files if path.name.endswith(".agent.md")
    ]
    by_name = {item["name"]: item for item in parsed}
    if len(by_name) != len(parsed):
        raise RuntimeError("agent_catalog_invalid")
    definitions: dict[str, AgentDefinition] = {}
    for item in parsed:
        child_ids = []
        for child_name in item["allowed_names"]:
            child = by_name.get(child_name)
            if child is None:
                continue
            child_ids.append(child["id"])
        definition = AgentDefinition(
            id=item["id"],
            name=item["name"],
            description=item["description"],
            instructions=item["instructions"],
            allowed_agents=tuple(child_ids),
            skill_names=item["skill_names"],
            digest=item["digest"],
        )
        if definition.id in definitions:
            raise RuntimeError("agent_catalog_invalid")
        definitions[definition.id] = definition

    resolved_digests: dict[str, str] = {}
    resolving: set[str] = set()

    def resolve_digest(agent_id: str) -> str:
        cached_digest = resolved_digests.get(agent_id)
        if cached_digest is not None:
            return cached_digest
        definition = definitions[agent_id]
        if agent_id in resolving:
            return definition.digest
        resolving.add(agent_id)
        payload = {
            "definition": definition.digest,
            "children": [
                (child_id, resolve_digest(child_id))
                for child_id in definition.allowed_agents
            ],
            "skills": [
                (
                    skill_name,
                    hashlib.sha256(
                        load_private_skill(skill_name).encode("utf-8")
                    ).hexdigest(),
                )
                for skill_name in definition.skill_names
            ],
        }
        digest = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        resolving.remove(agent_id)
        resolved_digests[agent_id] = digest
        return digest

    legacy_digests = {
        agent_id: resolve_digest(agent_id) for agent_id in definitions
    }
    independent_definitions = {}
    for agent_id, definition in definitions.items():
        payload = {
            "definition": definition.digest,
            "children": [],
            "skills": [
                (
                    skill_name,
                    hashlib.sha256(
                        load_private_skill(skill_name).encode("utf-8")
                    ).hexdigest(),
                )
                for skill_name in definition.skill_names
            ],
        }
        independent_digest = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        independent_definitions[agent_id] = replace(
            definition,
            digest=independent_digest,
            legacy_digest=legacy_digests[agent_id],
        )
    return independent_definitions


def list_agent_catalog() -> list[dict[str, object]]:
    return [
        definition.catalog_entry()
        for definition in load_agent_catalog().values()
    ]


def load_agent_definition(
    agent_id: str, expected_digest: str | None = None
) -> AgentDefinition:
    definition = load_agent_catalog().get(agent_id)
    if definition is None or (
        expected_digest is not None
        and expected_digest
        not in {definition.digest, definition.legacy_digest}
    ):
        raise RuntimeError("agent_definition_unavailable")
    return definition
