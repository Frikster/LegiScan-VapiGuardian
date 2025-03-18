import os
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

DEFAULT_REPORT_STRUCTURE = """Use this structure to create a report on the user-provided topic:

1. Introduction (no research needed)
   - Brief overview of the topic area

2. Main Body Sections:
   - Each section should focus on a sub-topic of the user-provided topic
   
3. Conclusion
   - Aim for 1 structural element (either a list of table) that distills the main body sections 
   - Provide a concise summary of the report"""

DEFAULT_VAPI_CALL_ANALYSIS_PLAN = {
    "structuredDataPlan": {
        "enabled": True,
        "schema": {
            "type": "object",
            "properties": {
                "agreed_to_request": {
                    "type": "boolean",
                    "description": "Whether the politician agreed to the specific request made in the call",
                },
                "level_of_support": {
                    "type": "string",
                    "enum": [
                        "strongly_supportive",
                        "supportive",
                        "neutral",
                        "opposed",
                        "strongly_opposed",
                    ],
                    "description": "The politician's level of support for the specific request made in the call",
                },
                "key_concerns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key concerns or objections raised by the politician",
                },
                "follow_up_needed": {
                    "type": "boolean",
                    "description": "Whether follow-up is needed with this politician",
                },
                "follow_up_type": {
                    "type": "string",
                    "enum": ["call", "email", "meeting", "none"],
                    "description": "Type of follow-up preferred by the politician",
                },
            },
            "required": ["agreed_to_request", "level_of_support"],
        },
    },
    "successEvaluationPlan": {"rubric": "AutomaticRubric"},
}


class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    EXA = "exa"
    ARXIV = "arxiv"
    PUBMED = "pubmed"


class PlannerProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GROQ = "groq"


class WriterProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GROQ = "groq"


@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the application."""

    report_structure: str = (
        DEFAULT_REPORT_STRUCTURE  # Defaults to the default report structure
    )
    number_of_queries: int = 10  # Number of search queries to generate per iteration
    max_search_depth: int = 2  # Maximum number of reflection + search iterations
    planner_provider: PlannerProvider = (
        PlannerProvider.OPENAI
    )  # Defaults to Anthropic as provider
    planner_model: str = "gpt-4o"  # Defaults to claude-3-7-sonnet-latest
    writer_provider: WriterProvider = (
        WriterProvider.OPENAI
    )  # Defaults to Anthropic as provider
    writer_model: str = "gpt-4o-mini"  # Defaults to claude-3-5-sonnet-latest
    search_api: SearchAPI = SearchAPI.TAVILY  # Default to TAVILY
    search_api_config: Optional[Dict[str, Any]] = None

    # New fields for legislation analysis
    number_of_politicians: int = 3  # Number of politicians to research

    # Vapi configuration
    vapi_phone_id: Optional[str] = None  # (if unset will use env var VAPI_PHONE_ID)
    vapi_to_number: Optional[str] = (
        None  # Optional preset to number (if unset will use env var TEST_NUMBER)
    )
    vapi_assistant_name: str = "Jennifer"
    vapi_organization_name: str = "The American Century Institute"
    # TODO: setting below does not work
    # vapi_analysis_plan: Dict[str, Any] = field(default_factory=lambda: DEFAULT_VAPI_CALL_ANALYSIS_PLAN)

    # Vector store configuration for persistence
    vector_store_path: str = "./vector_store"  # Path to store vector database
    cache_path: str = "./cache"  # Path to store embedding and LLM caches

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})
