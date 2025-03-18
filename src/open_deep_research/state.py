import operator
from typing import Annotated, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

# Report


class Section(BaseModel):
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section.",
    )
    research: bool = Field(
        description="Whether to perform web research for this section of the report."
    )
    content: str = Field(description="The content of the section.")


class Sections(BaseModel):
    sections: List[Section] = Field(
        description="Sections of the report.",
    )


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query for web search.")


class Queries(BaseModel):
    queries: List[SearchQuery] = Field(
        description="List of search queries.",
    )


class Feedback(BaseModel):
    grade: Literal["pass", "fail"] = Field(
        description="Evaluation result indicating whether the response meets requirements ('pass') or needs revision ('fail')."
    )
    follow_up_queries: List[SearchQuery] = Field(
        description="List of follow-up search queries.",
    )


class ReportStateInput(TypedDict):
    topic: Annotated[str, lambda x, y: x or y]  # Report topic
    additional_context: Annotated[dict, lambda x, y: x or y]


class PoliticianReport(BaseModel):
    """Report generated for a specific politician."""

    politician_name: str = Field(description="Name of the politician")
    topic: str = Field(description="Topic of the report")
    final_report: str = Field(description="Final report content")
    tldr_points: List[str] = Field(
        description="TLDR bullet points", default_factory=list
    )
    filename: str = Field(description="Location where file with report has been stored")


class ReportStateOutput(TypedDict):
    final_reports: list[PoliticianReport]


class ReportState(TypedDict):
    topic: Annotated[str, lambda x, y: x or y]  # Report topic   TODO:
    additional_context: Annotated[dict, lambda x, y: x or y]
    feedback_on_report_plan: str  # Feedback on the report plan
    sections: list[Section]  # List of report sections
    completed_sections: Annotated[list, operator.add]  # Send() API key
    report_sections_from_research: Annotated[
        str, lambda x, y: x or y
    ]  # String of any completed sections from research to write final sections
    final_reports: list[PoliticianReport]


class SectionState(TypedDict):
    topic: Annotated[str, lambda x, y: x or y]  # Report topic
    # additional_context: Annotated[dict, lambda x, y: x]
    # section: Section # Report section
    section: Annotated[Section, lambda x, y: x or y]
    search_iterations: Annotated[
        int, lambda x, y: x or y
    ]  # Number of search iterations done
    search_queries: Annotated[
        list[SearchQuery], lambda x, y: x or y
    ]  # List of search queries
    source_str: Annotated[
        str, lambda x, y: x or y
    ]  # String of formatted source content from web search
    report_sections_from_research: Annotated[
        str, lambda x, y: x or y
    ]  # String of any completed sections from research to write final sections
    completed_sections: list[
        Section
    ]  # Final key we duplicate in outer state for Send() API


class SectionOutputState(TypedDict):
    completed_sections: list[
        Section
    ]  # Final key we duplicate in outer state for Send() API


# Legislation

class LegislationAnalysis(BaseModel):
    name_of_legislation: str = Field(
        description="Name of the legislation being analyzed.",
    )
    summary: str = Field(
        description="Summary of the legislation's content and implications.",
    )
    issue_impact: str = Field(
        description="Analysis of how the legislation affects issue of concern.",
    )
    will_have_negative_impact: bool = Field(
        description="Whether the legislation will negatively affect the issue of concern.",
    )
    key_politician_names: List[str] = Field(
        description="List of key politicians that support this legislation.",
    )


class VapiCallConfig(BaseModel):
    assistant_id: Optional[str] = Field(
        description="ID of the Vapi assistant to use for the call.", default=None
    )
    phone_number_id: str = Field(
        description="ID of the phone number to call from.",
    )
    customer_number: str = Field(
        description="Phone number to call.",
    )
    customer_name: str = Field(
        description="Name of the customer being called.",
    )
    first_message: str = Field(
        description="First message the assistant will say when the call connects.",
    )
    system_prompt: str = Field(
        description="System prompt for the assistant to use during the call.",
    )
    assistant_name: str = Field(
        description="Name for the assistant to use when referring to itself.",
    )
    organization_name: str = Field(
        description="Name of the organization the assistant represents.",
    )
    analysis_plan: Optional[dict] = Field(
        description="Configuration for call analysis and outcome reporting.",
    )


class LegislationStateOutput(TypedDict):
    analysis: LegislationAnalysis  # Analysis of the legislation
    vapi_configs: List[VapiCallConfig]  # Vapi call configurations


class VapiTools(BaseModel):
    """Tools created in Vapi for the assistant."""

    legislation_file_id: str = Field(description="ID of the uploaded legislation file")
    tool_ids: List[str] = Field(description="IDs of the created query tools")

class Politician(BaseModel):
    name: str = Field(
        description="Name of the politician.",
    )
    position: str = Field(
        description="Current political position/office.",
    )
    background: str = Field(
        description="Political background and relevant history.",
    )
    stance_on_issue: str = Field(
        description="Known positions on input issue.",
    )
    financial_backing: str = Field(
        description="Information about financial supporters and donors.",
    )
    phone_number: Optional[str] = Field(
        description="Phone number to call, if available.", default=None
    )

class LegislationState(TypedDict):
    legislation_path: str  # Input legislation path
    issue_of_concern: str
    legislation_text: str
    analysis: LegislationAnalysis  # Analysis of the legislation
    politicians: Annotated[list[Politician], operator.add]  # List of politicians
    final_reports: Annotated[
        list[PoliticianReport], operator.add
    ]  # Reports for each politician
    vapi_tools: VapiTools  # Tools created in Vapi
    vapi_configs: Annotated[
        list[VapiCallConfig], operator.add
    ]  # Vapi call configurations
    approved_calls: list[VapiCallConfig]  # Calls approved by the user


class LegislationStateInput(TypedDict):
    legislation_path: str  # Path to legislation PDF file
    issue_of_concern: str  # Input issue that the app user is concerned about


class PoliticianResearchState(TypedDict):
    politician_name: str  # Name of the politician to research
    legislation_text: str  # Original legislation text for context
    issue_of_concern: str
    search_queries: list[SearchQuery]  # Search queries for research
    source_str: str  # String of formatted source content from web search
    politician: Politician  # Politician data structure to be populated

class PoliticianResearchOutput(TypedDict):
    politicians: Annotated[list[Politician], operator.add]  # List of researched politicians
