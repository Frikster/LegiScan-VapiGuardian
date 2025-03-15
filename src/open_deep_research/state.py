import operator
import os
from typing import Annotated, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


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
    content: str = Field(
        description="The content of the section."
    )   

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
    grade: Literal["pass","fail"] = Field(
        description="Evaluation result indicating whether the response meets requirements ('pass') or needs revision ('fail')."
    )
    follow_up_queries: List[SearchQuery] = Field(
        description="List of follow-up search queries.",
    )

class ReportStateInput(TypedDict):
    topic: str # Report topic
    
class ReportStateOutput(TypedDict):
    final_report: str # Final report

class ReportState(TypedDict):
    topic: str # Report topic    
    feedback_on_report_plan: str # Feedback on the report plan
    sections: list[Section] # List of report sections 
    completed_sections: Annotated[list, operator.add] # Send() API key
    report_sections_from_research: str # String of any completed sections from research to write final sections
    final_report: str # Final report

class SectionState(TypedDict):
    topic: str # Report topic
    section: Section # Report section  
    search_iterations: int # Number of search iterations done
    search_queries: list[SearchQuery] # List of search queries
    source_str: str # String of formatted source content from web search
    report_sections_from_research: str # String of any completed sections from research to write final sections
    completed_sections: list[Section] # Final key we duplicate in outer state for Send() API

class SectionOutputState(TypedDict):
    completed_sections: list[Section] # Final key we duplicate in outer state for Send() API

#######

class Politician(BaseModel):
    name: str = Field(
        description="Name of the politician.",
    )
    position: str = Field(
        description="Current political position/office.",
    )
    contact_info: str = Field(
        description="Contact information including phone number if available.",
    )
    background: str = Field(
        description="Political background and relevant history.",
    )
    stance_on_animals: str = Field(
        description="Known positions on animal welfare issues.",
    )
    financial_backing: str = Field(
        description="Information about financial supporters and donors.",
    )
    phone_number: Optional[str] = Field(
        description="Phone number to call, if available.",
        default=None
    )

class CallScript(BaseModel):
    politician: str = Field(
        description="Name of the politician this script is for.",
    )
    first_message: str = Field(
        description="First message the assistant will say when the call connects. Must be brief.",
    )
    key_points: List[str] = Field(
        description="Key talking points tailored to the politician's background.",
    )
    ask: str = Field(
        description="The specific request or action being asked of the politician.",
    )
    end_call_message: str = Field(
        description="Message the assistant will say before ending the call.",
    )
    full_script: str = Field(
        description="Complete call script combining all elements.",
    )

class LegislationAnalysis(BaseModel):
    summary: str = Field(
        description="Summary of the legislation's content and implications.",
    )
    animal_welfare_impact: str = Field(
        description="Analysis of how the legislation affects animal welfare.",
    )
    key_politicians: List[Politician] = Field(
        description="List of politicians relevant to this legislation.",
    )
    recommended_actions: List[str] = Field(
        description="Recommended advocacy actions based on the analysis.",
    )

class VapiCallConfig(BaseModel):
    assistant_id: str = Field(
        description="ID of the Vapi assistant to use for the call.",
    )
    phone_number_id: str = Field(
        description="ID of the phone number to call from.",
    )
    customer_number: str = Field(
        description="Phone number to call.",
        default=os.getenv('TEST_NUMBER'),
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
    enhanced_system_prompt: str = Field(
        description="Enhanced system prompt that is more conversational and concise.",
        default=None
    )
    assistant_name: str = Field(
        description="Name for the assistant to use when referring to itself.",
        default="Jennifer"
    )
    organization_name: str = Field(
        description="Name of the organization the assistant represents.",
        default="The American Century Institute"
    )
    end_call_message: str = Field(
        description="Message the assistant will say before ending the call.",
        default="Thank you for your time. Goodbye!"
    )
    analysis_plan: Optional[dict] = Field(
        description="Configuration for call analysis and outcome reporting.",
        default=None
    )

class LegislationStateInput(TypedDict):
    legislation_path: str  # Path to legislation PDF file
    
class LegislationStateOutput(TypedDict):
    analysis: LegislationAnalysis  # Analysis of the legislation
    call_scripts: List[CallScript]  # Generated call scripts
    vapi_configs: List[VapiCallConfig]  # Vapi call configurations

class LegislationState(TypedDict):
    legislation_path: str  # Input legislation path
    legislation_text: str
    analysis: LegislationAnalysis  # Analysis of the legislation
    politicians: Annotated[list[Politician], operator.add]  # List of politicians
    call_scripts: Annotated[list[CallScript], operator.add]  # Generated call scripts
    vapi_configs: Annotated[list[VapiCallConfig], operator.add]  # Vapi call configurations
    approved_calls: list[VapiCallConfig]  # Calls approved by the user

class PoliticianResearchState(TypedDict):
    politician_name: str  # Name of the politician to research
    legislation_text: str  # Original legislation text for context
    search_queries: list[SearchQuery]  # Search queries for research
    source_str: str  # String of formatted source content from web search
    politician: Politician  # Politician data structure to be populated

class PoliticianResearchOutput(TypedDict):
    politicians: Annotated[list[Politician], operator.add]  # List of researched politicians
