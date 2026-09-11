import json
from pathlib import Path


def load_source_registry() -> dict:
    """
    Load the trusted source registry from
    data/sources/source_registry.json.
    """

    project_root = Path(__file__).resolve().parents[1]
    registry_path = (
        project_root
        / "data"
        / "sources"
        / "source_registry.json"
    )

    if not registry_path.exists():
        raise FileNotFoundError(
            f"Source registry not found: {registry_path}"
        )

    with registry_path.open("r", encoding="utf-8") as file:
        registry = json.load(file)

    if "sources" not in registry:
        raise ValueError(
            "Source registry must contain a 'sources' field."
        )

    if not isinstance(registry["sources"], list):
        raise ValueError(
            "'sources' must be a list."
        )

    return registry


def find_sources(
    registry: dict,
    topic: str | None = None,
    geographic_scope: str | None = None,
    priority: str | None = None,
    industry: str | None = None,
) -> list[dict]:
    """
    Find trusted sources matching the requested criteria.

    Filters can be applied using:
    - topic
    - geographic scope
    - priority
    - industry
    """

    results = []

    for source in registry["sources"]:

        # Filter by topic
        if topic:
            source_topics = [
                item.lower()
                for item in source.get("topics", [])
            ]

            if topic.lower() not in source_topics:
                continue

        # Filter by geographic scope
        if geographic_scope:
            if (
                source.get("geographic_scope", "").lower()
                != geographic_scope.lower()
            ):
                continue

        # Filter by priority
        if priority:
            if (
                source.get("priority", "").lower()
                != priority.lower()
            ):
                continue

        # Filter by industry
        if industry:
            source_industries = [
                item.lower()
                for item in source.get("industries", [])
            ]

            if industry.lower() not in source_industries:
                continue

        results.append(source)

    return results