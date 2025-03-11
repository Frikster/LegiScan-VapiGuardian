import os
from enum import Enum
from dataclasses import dataclass, fields
from typing import Any, Optional, Dict 

from langchain_core.runnables import RunnableConfig
from dataclasses import dataclass

DEFAULT_REPORT_STRUCTURE = """Use this structure to create a report on the user-provided topic:

1. Introduction (no research needed)
   - Brief overview of the topic area

2. Main Body Sections:
   - Each section should focus on a sub-topic of the user-provided topic
   
3. Conclusion
   - Aim for 1 structural element (either a list of table) that distills the main body sections 
   - Provide a concise summary of the report"""

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

class VapiProvider(Enum):
    VAPI = "vapi"

@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the application."""
    report_structure: str = DEFAULT_REPORT_STRUCTURE # Defaults to the default report structure
    number_of_queries: int = 2 # Number of search queries to generate per iteration
    max_search_depth: int = 2 # Maximum number of reflection + search iterations
    planner_provider: PlannerProvider = PlannerProvider.OPENAI  # Defaults to Anthropic as provider
    planner_model: str = "gpt-4o" # Defaults to claude-3-7-sonnet-latest
    writer_provider: WriterProvider = WriterProvider.OPENAI # Defaults to Anthropic as provider
    writer_model: str = "gpt-4o-mini" # Defaults to claude-3-5-sonnet-latest
    search_api: SearchAPI = SearchAPI.TAVILY # Default to TAVILY
    search_api_config: Optional[Dict[str, Any]] = None 

    # New fields for legislation analysis
    number_of_politicians: int = 3  # Number of politicians to research
    max_search_depth: int = 2  # Maximum number of reflection + search iterations
    
    # Vapi configuration
    vapi_provider: VapiProvider = VapiProvider.VAPI
    vapi_assistant_id: Optional[str] = None  # Optional preset assistant ID
    vapi_from_number: Optional[str] = None  # Optional preset from number
    vapi_to_number: Optional[str] = None  # Optional preset to number (for testing)
    
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