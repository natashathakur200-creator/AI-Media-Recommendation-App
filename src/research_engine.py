import os
import json
from urllib.parse import urlparse

from dotenv import load_dotenv
from tavily import TavilyClient


SOURCE_REGISTRY_PATH = "data/sources/source_registry.json"


def load_source_registry() -> list:
    """
    Load approved research sources from the source registry.
    """

    if not os.path.exists(SOURCE_REGISTRY_PATH):
        raise FileNotFoundError(
            f"Source registry not found: {SOURCE_REGISTRY_PATH}"
        )

    with open(
        SOURCE_REGISTRY_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    return data.get("sources", [])


def get_approved_domains(
    industry: str = "",
    state: str = ""
) -> list:
    """
    Return approved domains relevant to the
    requested industry and geographic area.
    """

    sources = load_source_registry()

    approved_domains = []

    industry_normalized = (
        industry.strip().lower()
    )

    state_normalized = (
        state.strip().lower()
    )

    for source in sources:

        source_industries = [
            item.lower()
            for item in source.get(
                "industries",
                []
            )
        ]

        geographic_scope = (
            source.get(
                "geographic_scope",
                ""
            )
            .strip()
            .lower()
        )

        industry_matches = (
            not industry_normalized
            or industry_normalized in source_industries
        )

        geography_matches = (
            not state_normalized
            or state_normalized in geographic_scope
            or geographic_scope in state_normalized
        )

        if industry_matches and geography_matches:

            parsed_url = urlparse(
                source.get("url", "")
            )

            domain = parsed_url.netloc

            if (
                domain
                and domain not in approved_domains
            ):
                approved_domains.append(domain)

    return approved_domains


def get_research_client():
    """
    Create and return a Tavily research client.
    """

    load_dotenv()

    api_key = os.getenv(
        "TAVILY_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "Missing TAVILY_API_KEY in .env"
        )

    return TavilyClient(
        api_key=api_key
    )


def search_market_research(
    query: str,
    max_results: int = 5,
    approved_domains: list | None = None
) -> list:
    """
    Search for market information.

    When approved_domains are supplied,
    Tavily restricts the search to those domains.
    """

    client = get_research_client()

    search_arguments = {
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
    }

    if approved_domains:
        search_arguments["include_domains"] = (
            approved_domains
        )

    response = client.search(
        **search_arguments
    )

    return response.get(
        "results",
        []
    )