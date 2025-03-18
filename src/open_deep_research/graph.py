import os
from typing import Any, Dict, Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import OpenAIEmbeddings
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from open_deep_research.configuration import (
    DEFAULT_VAPI_CALL_ANALYSIS_PLAN,
    Configuration,
)
from open_deep_research.prompts import (
    final_section_writer_instructions,
    legislation_analysis_prompt,
    query_writer_instructions,
    report_planner_instructions,
    report_planner_query_writer_instructions,
    section_grader_instructions,
    section_writer_instructions,
    tldr_prompt,
    vapi_system_prompt_template,
    politician_query_writer_instructions,
    politician_research_instructions,
    tldr_prompt_with_phone_context,
    report_topic
)
from open_deep_research.state import (
    Feedback,
    LegislationAnalysis,
    LegislationState,
    LegislationStateInput,
    LegislationStateOutput,
    PoliticianReport,
    Queries,
    ReportState,
    ReportStateInput,
    ReportStateOutput,
    SectionOutputState,
    Sections,
    SectionState,
    VapiCallConfig,
    VapiTools,
    PoliticianResearchState,
    PoliticianResearchOutput,
    Politician
)
from open_deep_research.utils import (
    arxiv_search_async,
    create_query_tool,
    create_vapi_assistant,
    deduplicate_and_format_sources,
    exa_search,
    extract_text_from_pdf,
    format_sections,
    get_config_value,
    get_search_params,
    make_vapi_call,
    perplexity_search,
    pubmed_search_async,
    setup_embedding_cache,
    setup_llm_cache,
    setup_persistent_vectorstore,
    tavily_search_async,
    upload_file_to_vapi,
    upload_report_to_trieve
)


# Nodes
async def generate_report_plan(state: ReportState, config: RunnableConfig):
    """Generate the report plan."""
    # Inputs
    print("generate_report_plan state", state)
    print("Type of topic:", type(state["topic"]))
    topic = state["topic"]
    additional_context = state["additional_context"]
    feedback = state.get("feedback_on_report_plan", None)

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    report_structure = configurable.report_structure
    number_of_queries = configurable.number_of_queries
    search_api = get_config_value(configurable.search_api)
    search_api_config = (
        configurable.search_api_config or {}
    )  # Get the config dict, default to empty
    params_to_pass = get_search_params(
        search_api, search_api_config
    )  # Filter parameters

    # Convert JSON object to string if necessary
    if isinstance(report_structure, dict):
        report_structure = str(report_structure)

    # Set writer model (model used for query writing and section writing)
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model = init_chat_model(
        model=writer_model_name, model_provider=writer_provider, temperature=0
    )
    structured_llm = writer_model.with_structured_output(Queries)

    # Format system instructions
    system_instructions_query = report_planner_query_writer_instructions.format(
        topic=topic,
        additional_context=additional_context,
        report_organization=report_structure,
        number_of_queries=number_of_queries,
    )

    # Generate queries
    results = structured_llm.invoke(
        [
            SystemMessage(content=system_instructions_query),
            HumanMessage(
                content="Generate search queries that will help with planning the sections of the report."
            ),
        ]
    )

    # Web search
    query_list = [query.search_query for query in results.queries]

    # Search the web with parameters
    if search_api == "tavily":
        search_results = await tavily_search_async(query_list, **params_to_pass)
        source_str = deduplicate_and_format_sources(
            search_results, max_tokens_per_source=1000, include_raw_content=False
        )
    elif search_api == "perplexity":
        search_results = perplexity_search(query_list, **params_to_pass)
        source_str = deduplicate_and_format_sources(
            search_results, max_tokens_per_source=1000, include_raw_content=False
        )
    elif search_api == "exa":
        search_results = await exa_search(query_list, **params_to_pass)
        source_str = deduplicate_and_format_sources(
            search_results, max_tokens_per_source=1000, include_raw_content=False
        )
    elif search_api == "arxiv":
        search_results = await arxiv_search_async(query_list, **params_to_pass)
        source_str = deduplicate_and_format_sources(
            search_results, max_tokens_per_source=1000, include_raw_content=False
        )
    elif search_api == "pubmed":
        search_results = await pubmed_search_async(query_list, **params_to_pass)
        source_str = deduplicate_and_format_sources(
            search_results, max_tokens_per_source=1000, include_raw_content=False
        )
    else:
        raise ValueError(f"Unsupported search API: {search_api}")

    # Format system instructions
    system_instructions_sections = report_planner_instructions.format(
        topic=topic,
        additional_context=additional_context,
        report_organization=report_structure,
        context=source_str,
        feedback=feedback,
    )

    # Set the planner
    planner_provider = get_config_value(configurable.planner_provider)
    planner_model = get_config_value(configurable.planner_model)

    # Report planner instructions
    planner_message = """Generate the sections of the report. Your response must include a 'sections' field containing a list of sections. 
                        Each section must have: name, description, plan, research, and content fields."""

    # Run the planner
    if planner_model == "claude-3-7-sonnet-latest":
        # Allocate a thinking budget for claude-3-7-sonnet-latest as the planner model
        planner_llm = init_chat_model(
            model=planner_model,
            model_provider=planner_provider,
            max_tokens=20_000,
            thinking={"type": "enabled", "budget_tokens": 16_000},
        )

        # with_structured_output uses forced tool calling, which thinking mode with Claude 3.7 does not support
        # So, we use bind_tools without enforcing tool calling to generate the report sections
        report_sections = planner_llm.bind_tools([Sections]).invoke(
            [
                SystemMessage(content=system_instructions_sections),
                HumanMessage(content=planner_message),
            ]
        )
        tool_call = report_sections.tool_calls[0]["args"]
        report_sections = Sections.model_validate(tool_call)

    else:
        # With other models, we can use with_structured_output
        planner_llm = init_chat_model(
            model=planner_model, model_provider=planner_provider
        )
        structured_llm = planner_llm.with_structured_output(Sections)
        report_sections = structured_llm.invoke(
            [
                SystemMessage(content=system_instructions_sections),
                HumanMessage(content=planner_message),
            ]
        )

    # Get sections
    sections = report_sections.sections

    return {"sections": sections}


def human_feedback(
    state: ReportState, config: RunnableConfig
) -> Command[Literal["generate_report_plan", "build_section_with_web_research"]]:
    """Get feedback on the report plan."""
    # Get sections
    topic = state["topic"]
    # additional_context = state["additional_context"]
    sections = state["sections"]
    sections_str = "\n\n".join(
        f"Section: {section.name}\n"
        f"Description: {section.description}\n"
        f"Research needed: {'Yes' if section.research else 'No'}\n"
        for section in sections
    )

    # Get feedback on the report plan from interrupt
    interrupt_message = f"""Please provide feedback on the following report plan. 
                        \n\nTopic: {topic}\n\n
                        \n\n{sections_str}\n\n
                        \nDoes the report plan meet your needs? Pass "Yes" to approve the report plan or provide feedback to regenerate the report plan:"""

    feedback = interrupt(interrupt_message)

    # If the user approves the report plan, kick off section writing
    if feedback == "Yes":
        # Treat this as approve and kick off section writing
        return Command(
            goto=[
                Send(
                    "build_section_with_web_research",
                    {"topic": topic,
                    #  "additional_context": additional_context,
                     "section": s, "search_iterations": 0},
                )
                for s in sections
                if s.research
            ]
        )

    # If the user provides feedback, regenerate the report plan
    elif isinstance(feedback, str):
        # Treat this as feedback
        return Command(
            goto="generate_report_plan", update={"feedback_on_report_plan": feedback}
        )
    else:
        raise TypeError(f"Interrupt value of type {type(feedback)} is not supported.")


def generate_queries(state: SectionState, config: RunnableConfig):
    """Generate search queries for a report section."""
    # Get state
    topic = state["topic"]
    # additional_context = state["additional_context"]
    section = state["section"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    number_of_queries = configurable.number_of_queries

    # Generate queries
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model = init_chat_model(
        model=writer_model_name, model_provider=writer_provider, temperature=0
    )
    structured_llm = writer_model.with_structured_output(Queries)

    # Format system instructions
    system_instructions = query_writer_instructions.format(
        topic=topic,
        # additional_context=additional_context,
        section_topic=section.description,
        number_of_queries=number_of_queries,
    )

    # Generate queries
    queries = structured_llm.invoke(
        [
            SystemMessage(content=system_instructions),
            HumanMessage(content="Generate search queries on the provided topic."),
        ]
    )

    return {"search_queries": queries.queries}


async def search_web(state: SectionState, config: RunnableConfig):
    """Search the web for each query, then return a list of raw sources and a formatted string of sources."""
    # Get state
    search_queries = state["search_queries"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    search_api = get_config_value(configurable.search_api)
    search_api_config = (
        configurable.search_api_config or {}
    )  # Get the config dict, default to empty
    params_to_pass = get_search_params(
        search_api, search_api_config
    )  # Filter parameters

    # Web search
    query_list = [query.search_query for query in search_queries]

    # Search the web with parameters
    if search_api == "tavily":
        search_results = await tavily_search_async(query_list, **params_to_pass)
        source_str = deduplicate_and_format_sources(
            search_results, max_tokens_per_source=5000, include_raw_content=True
        )
    elif search_api == "perplexity":
        search_results = perplexity_search(query_list, **params_to_pass)
        source_str = deduplicate_and_format_sources(
            search_results, max_tokens_per_source=5000, include_raw_content=False
        )
    elif search_api == "exa":
        search_results = await exa_search(query_list, **params_to_pass)
        source_str = deduplicate_and_format_sources(
            search_results, max_tokens_per_source=1000, include_raw_content=False
        )
    elif search_api == "arxiv":
        search_results = await arxiv_search_async(query_list, **params_to_pass)
        source_str = deduplicate_and_format_sources(
            search_results, max_tokens_per_source=1000, include_raw_content=False
        )
    elif search_api == "pubmed":
        search_results = await pubmed_search_async(query_list, **params_to_pass)
        source_str = deduplicate_and_format_sources(
            search_results, max_tokens_per_source=1000, include_raw_content=False
        )
    else:
        raise ValueError(f"Unsupported search API: {search_api}")

    return {
        "source_str": source_str,
        "search_iterations": state["search_iterations"] + 1,
    }


def write_section(
    state: SectionState, config: RunnableConfig
) -> Command[Literal[END, "search_web"]]:
    """Write a section of the report."""
    # Get state
    topic = state["topic"]
    # additional_context = state["additional_context"]
    section = state["section"]
    source_str = state["source_str"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # Format system instructions
    system_instructions = section_writer_instructions.format(
        topic=topic,
        # additional_context=additional_context,
        section_name=section.name,
        section_topic=section.description,
        context=source_str,
        section_content=section.content,
    )

    # Generate section
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model = init_chat_model(
        model=writer_model_name, model_provider=writer_provider, temperature=0
    )
    section_content = writer_model.invoke(
        [
            SystemMessage(content=system_instructions),
            HumanMessage(
                content="Generate a report section based on the provided sources."
            ),
        ]
    )

    # Write content to the section object
    section.content = section_content.content

    # Grade prompt
    section_grader_message = """Grade the report and consider follow-up questions for missing information.
                               If the grade is 'pass', return empty strings for all follow-up queries.
                               If the grade is 'fail', provide specific search queries to gather missing information."""

    section_grader_instructions_formatted = section_grader_instructions.format(
        topic=topic,
        # additional_context=additional_context,
        section_topic=section.description,
        section=section.content,
        number_of_follow_up_queries=configurable.number_of_queries,
    )

    # Use planner model for reflection
    planner_provider = get_config_value(configurable.planner_provider)
    planner_model = get_config_value(configurable.planner_model)

    # If the planner model is claude-3-7-sonnet-latest, we need to use bind_tools to use thinking when generating the feedback
    if planner_model == "claude-3-7-sonnet-latest":
        # Allocate a thinking budget for claude-3-7-sonnet-latest as the planner model
        reflection_model = init_chat_model(
            model=planner_model,
            model_provider=planner_provider,
            max_tokens=20_000,
            thinking={"type": "enabled", "budget_tokens": 16_000},
        )

        # with_structured_output uses forced tool calling, which thinking mode with Claude 3.7 does not support
        # So, we use bind_tools without enforcing tool calling to generate the report sections
        reflection_result = reflection_model.bind_tools([Feedback]).invoke(
            [
                SystemMessage(content=section_grader_instructions_formatted),
                HumanMessage(content=section_grader_message),
            ]
        )
        tool_call = reflection_result.tool_calls[0]["args"]
        feedback = Feedback.model_validate(tool_call)

    else:
        reflection_model = init_chat_model(
            model=planner_model, model_provider=planner_provider
        ).with_structured_output(Feedback)

        feedback = reflection_model.invoke(
            [
                SystemMessage(content=section_grader_instructions_formatted),
                HumanMessage(content=section_grader_message),
            ]
        )

    # If the section is passing or the max search depth is reached, publish the section to completed sections
    if (
        feedback.grade == "pass"
        or state["search_iterations"] >= configurable.max_search_depth
    ):
        # Publish the section to completed sections
        return Command(update={"completed_sections": [section]}, goto=END)
    # Update the existing section with new content and update search queries
    else:
        return Command(
            update={"search_queries": feedback.follow_up_queries, "section": section},
            goto="search_web",
        )


def write_final_sections(state: SectionState, config: RunnableConfig):
    """Write final sections of the report, which do not require web search and use the completed sections as context."""
    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # Get state
    topic = state["topic"]
    # additional_context = state["additional_context"]
    section = state["section"]
    completed_report_sections = state["report_sections_from_research"]

    # Format system instructions
    system_instructions = final_section_writer_instructions.format(
        topic=topic,
        # additional_context=additional_context,
        section_name=section.name,
        section_topic=section.description,
        context=completed_report_sections,
    )

    # Generate section
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model_name = get_config_value(configurable.writer_model)
    writer_model = init_chat_model(
        model=writer_model_name, model_provider=writer_provider, temperature=0
    )
    section_content = writer_model.invoke(
        [
            SystemMessage(content=system_instructions),
            HumanMessage(
                content="Generate a report section based on the provided sources."
            ),
        ]
    )

    # Write content to section
    section.content = section_content.content

    # Write the updated section to completed sections
    return {"completed_sections": [section]}


def gather_completed_sections(state: ReportState):
    """Gather completed sections from research and format them as context for writing the final sections."""
    # List of completed sections
    completed_sections = state["completed_sections"]

    # Format completed section to str to use as context for final sections
    completed_report_sections = format_sections(completed_sections)

    return {"report_sections_from_research": completed_report_sections}


def initiate_final_section_writing(state: ReportState):
    """Write any final sections using the Send API to parallelize the process."""
    # Kick off section writing in parallel via Send() API for any sections that do not require research
    return [
        Send(
            "write_final_sections",
            {
                "topic": state["topic"],
                "section": s,
                "report_sections_from_research": state["report_sections_from_research"],
            },
        )
        for s in state["sections"]
        if not s.research
    ]


def compile_final_report(state: ReportState, config: RunnableConfig):
    """Compile the final report and create a PoliticianReport object."""
    # Get sections
    print(f"state in compile_final_report: {state}")
    sections = state["sections"]
    completed_sections = {s.name: s.content for s in state["completed_sections"]}
    topic = state["topic"]
    additional_context = state["additional_context"]
    politician_name = additional_context["politician"].name
    name_of_legislation = additional_context["analysis"].name_of_legislation

    # Update sections with completed content while maintaining original order
    for section in sections:
        section.content = completed_sections[section.name]

    # Compile final report
    all_sections = "\n\n".join([s.content for s in sections])

    # Replace spaces with underscores and remove special characters for filename
    unsafe_filename = f"Report_on_{politician_name}_and_{name_of_legislation}"
    safe_filename = unsafe_filename.replace(" ", "_").replace(",", "").replace(".", "")
    filename = f"data/{safe_filename}.txt"

    # Ensure data directory exists
    if not os.path.exists("data"):
        os.makedirs("data")

    # Write the report to a file
    with open(filename, "w") as f:
        f.write(all_sections)

    print(f"Report written to {filename}")

    # Upload the file to Trieve
    # upload_report_to_trieve(filename, topic)

    # Generate TLDR points
    configurable = Configuration.from_runnable_config(config)
    llm = init_chat_model(
        model=get_config_value(configurable.planner_model),
        model_provider=get_config_value(configurable.planner_provider),
    )
    response = llm.invoke(
        [SystemMessage(content=tldr_prompt_with_phone_context.format(input=all_sections))]
    )

    # Extract bullet points from response
    bullet_points = [
        point.strip("- ").strip()
        for point in response.content.split("\n")
        if point.strip().startswith("-")
    ]

    # Create a PoliticianReport object
    politician_report = PoliticianReport(
        politician_name=politician_name,
        topic=state["topic"],
        final_report=all_sections,
        tldr_points=bullet_points,
        filename=filename
    )

    return {
        # "final_report": all_sections,  # Keep for backward compatibility
        "final_reports": [politician_report]  # Return as a list for proper aggregation
    }


# legislation analysis workflow


def analyze_legislation(state: LegislationState, config: RunnableConfig):
    """Analyze the legislation for issue implications."""
    legislation_path = state["legislation_path"]
    issue_of_concern = state["issue_of_concern"]

    # If we have a path but no text, extract text from the PDF
    # if not legislation_text and legislation_path:
    legislation_text = extract_text_from_pdf(legislation_path)

    # If we still don't have text, raise an error
    if not legislation_text:
        raise ValueError("No legislation text or valid PDF path provided")

    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # Set up LLM
    planner_provider = get_config_value(configurable.planner_provider)
    planner_model = get_config_value(configurable.planner_model)
    llm = init_chat_model(model=planner_model, model_provider=planner_provider)

    # Analyze legislation - use structured output directly
    system_message = legislation_analysis_prompt.format(
        legislation_text=legislation_text, issue_of_concern=issue_of_concern
    )

    # Only make one LLM call with structured output
    structured_llm = llm.with_structured_output(LegislationAnalysis)
    analysis = structured_llm.invoke(
        [
            SystemMessage(content=system_message),
            HumanMessage(
                content=f"Provide a structured analysis of this legislation with focus on {issue_of_concern} implications."
            ),
        ]
    )

    # Return analysis
    return {
        "analysis": analysis,
        "legislation_text": legislation_text,
        # "politicians": analysis.key_politicians,
    }

def identify_politicians_to_research(state: LegislationState, config: RunnableConfig) -> Command[Literal["research_politician"]]:
    """Identify politicians to research and kick off parallel research."""
    # Get analysis
    analysis = state["analysis"]
    issue_of_concern = state["issue_of_concern"]
    legislation_text = state["legislation_text"]
     
    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    max_politicians = configurable.number_of_politicians
    
    # Limit to the specified number of politicians
    politicians_to_research_names = analysis.key_politician_names[:max_politicians]
     
    # Kick off politician research in parallel
    return Command(goto=[
        Send("research_politician", {
             "politician_name": name,
             "legislation_text": legislation_text,
             "issue_of_concern": issue_of_concern
        }) for name in politicians_to_research_names
    ])
 
def generate_politician_queries(state: PoliticianResearchState, config: RunnableConfig):
    """Generate search queries for researching a politician."""
    # Get state
    politician_name = state["politician_name"]
    legislation_text = state["legislation_text"]
    issue_of_concern = state["issue_of_concern"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    number_of_queries = configurable.number_of_queries

    # Set up LLM
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model = get_config_value(configurable.writer_model)
    llm = init_chat_model(model=writer_model, model_provider=writer_provider)
    structured_llm = llm.with_structured_output(Queries)

    # Generate queries
    system_instructions = politician_query_writer_instructions.format(
        politician_name=politician_name,
        legislation_context=legislation_text,
        number_of_queries=number_of_queries,
        issue_of_concern=issue_of_concern
    )

    # TODO: Improve to get multiple variants of arguments based on different conversation paths
    queries = structured_llm.invoke([
        SystemMessage(content=system_instructions),
        HumanMessage(content="Generate search queries to research this politician.")
    ])

    return {"search_queries": queries.queries}

async def search_politician_info(state: PoliticianResearchState, config: RunnableConfig):
    """Search for information about the politician."""
    # Get state
    search_queries = state["search_queries"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    search_api = get_config_value(configurable.search_api)

    # Web search
    query_list = [query.search_query for query in search_queries]
    # TODO
    # query_list = [query.search_query for query in results.queries]
    # params_to_pass = get_search_params(search_api, search_api_config)  # Filter parameters

    # Search the web
    if search_api == "tavily":
        search_results = await tavily_search_async(query_list)
        source_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=5000)
    elif search_api == "perplexity":
        search_results = perplexity_search(query_list)
        source_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=5000)
    elif search_api == "exa":
        search_results = await exa_search(query_list)
        source_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=5000)
    else:
        raise ValueError(f"Unsupported search API: {search_api}")

    return {"source_str": source_str}

def research_politician(state: PoliticianResearchState, config: RunnableConfig):
    """Research a politician based on search results."""
    # Get state
    politician_name = state["politician_name"]
    legislation_text = state["legislation_text"]
    source_str = state["source_str"]
    issue_of_concern = state["issue_of_concern"]

    # Get configuration
    configurable = Configuration.from_runnable_config(config)

    # Set up LLM
    writer_provider = get_config_value(configurable.writer_provider)
    writer_model = get_config_value(configurable.writer_model)
    llm = init_chat_model(model=writer_model, model_provider=writer_provider)
    structured_llm = llm.with_structured_output(Politician)

    # Research politician
    system_instructions = politician_research_instructions.format(
        politician_name=politician_name,
        legislation_context=legislation_text,
        context=source_str,
        issue_of_concern=issue_of_concern
    )

    politician = structured_llm.invoke([
        SystemMessage(content=system_instructions),
        HumanMessage(content="Research this politician based on the provided sources.")
    ])

    return {"politicians": [politician]}  # Return as a list for proper aggregation


def initiate_politician_reports(
    state: LegislationState, config: RunnableConfig
) -> Command[Literal["generate_report"]]:
    """Identify politicians to research and process them sequentially."""
    analysis = state["analysis"]
    politicians = state["politicians"]
    
    print("initiate_politician_reports state: ", state)

    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    max_politicians = configurable.number_of_politicians

    # Limit to the specified number of politicians
    politicians_to_research = politicians[:max_politicians]
    
    # Format the topic for each politician using the report_topic template
    formatted_politician_topics = []
    for politician in politicians_to_research:
        formatted_topic = report_topic.format(
            legislation_name=analysis.name_of_legislation,
            name=politician.name
        )
        formatted_politician_topics.append((politician, formatted_topic))

    # Spawn a report generation task for each politician
    return Command(
        goto=[
            Send(
                "generate_report",
                {
                    "topic": formatted_politician_topic[1],  # Use the formatted topic from the template
                    #  TODO: add adiditional context? Seems to degrade peroformance
                    "additional_context": {
                        "politician": formatted_politician_topic[0],
                        "analysis": analysis,
                    },
                },
            )
            for formatted_politician_topic in formatted_politician_topics
        ]
    )

def add_report_to_trieve(state: ReportState):
    """Add the final report to Trieve for future retrieval.

    Args:
        state: Current state containing the final report

    Returns:
        dict: Empty dict as the function is the final node
    """
    import os
    import re
    from datetime import datetime

    import requests

    final_report = state["final_report"]
    topic = state["topic"]

    # Configure Trieve API request
    url = "https://api.trieve.ai/api/chunk"
    headers = {
        "Authorization": os.getenv("TRIEVE_API_KEY"),
        "Content-Type": "application/json",
        "TR-Dataset": os.getenv("TRIEVE_DATASET_ID"),
    }

    # Split the report into sections based on markdown headers
    sections = re.split(r"(#+\s+.*?\n)", final_report)
    chunks = []

    # Recombine headers with their content
    i = 0
    while i < len(sections):
        if i + 1 < len(sections) and re.match(r"#+\s+.*?\n", sections[i]):
            # This is a header, combine with the next section
            chunks.append(sections[i] + sections[i + 1])
            i += 2
        else:
            # This is content without a header or a trailing section
            if sections[i].strip():  # Only add non-empty sections
                chunks.append(sections[i])
            i += 1

    # Add metadata to each chunk
    for i, chunk in enumerate(chunks):
        if chunk.strip():  # Skip empty chunks
            payload = {
                "chunk_html": chunk,
                "metadata": {
                    "topic": topic,
                    "chunk_number": i + 1,
                    "total_chunks": len(chunks),
                    "date_added": datetime.now().isoformat(),
                },
            }

            # Make the API request for each chunk
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()

    print(f"Successfully added report to Trieve in {len(chunks)} chunks")
    return {}  # Return empty dict as this is the final node in this branch (it just adds the chunks to trieve)


# Create a new node for generating TLDR points
def generate_tldr_points(state: ReportState, config: RunnableConfig):
    """Generate TLDR bullet points from a completed report."""
    final_report = state["final_report"]
    configurable = Configuration.from_runnable_config(config)
    llm = init_chat_model(
        model=get_config_value(configurable.planner_model),
        model_provider=get_config_value(configurable.planner_model),
    )
    response = llm.invoke(
        [SystemMessage(content=tldr_prompt_with_phone_context.format(input=final_report))]
    )

    # Extract bullet points from response
    bullet_points = [
        point.strip("- ").strip()
        for point in response.content.split("\n")
        if point.strip().startswith("-")
    ]
    return {"tldr_points": bullet_points}


# Replaced with full research report code. Left as comments in case useful before demo

# def generate_call_scripts(state: LegislationState, config: RunnableConfig):
#     """Generate call scripts for each politician and enhance them to be more conversational and concise."""
#     # Get state
#     analysis = state["analysis"]
#     politicians = state["politicians"]

#     # Get configuration
#     configurable = Configuration.from_runnable_config(config)

#     # Set up LLM
#     writer_provider = get_config_value(configurable.writer_provider)
#     writer_model = get_config_value(configurable.writer_model)
#     llm = init_chat_model(model=writer_model, model_provider=writer_provider)
#     structured_llm = llm.with_structured_output(CallScript)

#     # Generate call scripts for each politician
#     call_scripts = []
#     for politician in politicians:
#         system_instructions = call_script_generation_instructions.format(
#             assistant_name=configurable.vapi_assistant_name,
#             politician_profile=politician.model_dump_json(),
#             legislation_summary=analysis.summary
#         )

#         script = structured_llm.invoke([
#             SystemMessage(content=system_instructions),
#             HumanMessage(content="Generate a call script for this politician.")
#         ])

#         call_scripts.append(script)

#     return {"call_scripts": call_scripts}


def create_vapi_tools(
    state: LegislationState, config: RunnableConfig
) -> Dict[str, Any]:
    """Create Vapi tools for the assistant."""
    legislation_path = state["legislation_path"]
    final_reports = state["final_reports"]
    
    legislation_file_id = upload_file_to_vapi(legislation_path)
    query_legislation_tool_id = create_query_tool(legislation_file_id)
    process_email_tool_id = os.getenv("VAPI_PROCESS_EMAIL_INFO_TOOL_ID") or "cd38c02a-70e1-4e62-bfe1-cc51dc7899cf"
    for report in final_reports:
        upload_report_to_trieve(report.filename, report.topic)
    return {"vapi_tools": VapiTools(legislation_file_id=legislation_file_id, tool_ids=[query_legislation_tool_id, process_email_tool_id])}


def prepare_vapi_configs(
    state: LegislationState, config: RunnableConfig
) -> Dict[str, Any]:
    """Prepare Vapi call configurations for each politician."""
    # Get state and configuration
    configurable = Configuration.from_runnable_config(config)
    politicians = state["politicians"]
    final_reports = state["final_reports"]
    analysis = state["analysis"]
    legislation_name = analysis.name_of_legislation

    # Create a lookup dictionary for politician reports
    report_by_politician = {report.politician_name: report for report in final_reports}

    vapi_configs = []
    for politician in politicians:
        # if politician.phone_number and politician.name in report_by_politician:
        if politician.name in report_by_politician:
            report = report_by_politician[politician.name]

            # Format key points as a bulleted list
            key_points_formatted = "\n".join(
                [f"- {point}" for point in report.tldr_points]
            )
            
            # Generate TLDR for politician profile
            configurable = Configuration.from_runnable_config(config)
            llm = init_chat_model(
                model=get_config_value(configurable.planner_model),
                model_provider=get_config_value(configurable.planner_provider),
            )
            politician_tldr_response = llm.invoke(
                [SystemMessage(content=tldr_prompt.format(input=politician.model_dump_json()))]
            )

            # Create system prompt with all context
            system_prompt = vapi_system_prompt_template.format(
                assistant_name=configurable.vapi_assistant_name,
                organization_name=configurable.vapi_organization_name,
                ask=f"You want to urge the politician to reconsider their support for {legislation_name}",  # TODO: Make this configurable
                politician_profile=politician_tldr_response.content,
                tldr_points=key_points_formatted,
            )

            # Configure the call
            vapi_config = VapiCallConfig(
                phone_number_id=configurable.vapi_phone_id,
                customer_number=configurable.vapi_to_number or os.getenv("TEST_NUMBER"),
                customer_name=politician.name,
                first_message=f"Hi, have I reached {politician.name}'s office?",
                system_prompt=system_prompt,
                assistant_name=configurable.vapi_assistant_name,
                organization_name=configurable.vapi_organization_name,
                analysis_plan=DEFAULT_VAPI_CALL_ANALYSIS_PLAN,
            )

            vapi_configs.append(vapi_config)

    return {"vapi_configs": vapi_configs}


def create_vapi_assistants(
    state: LegislationState, config: RunnableConfig
) -> Dict[str, Any]:
    """Create Vapi assistants for each configuration."""
    vapi_configs = state["vapi_configs"]
    tool_ids = state["vapi_tools"].tool_ids

    updated_configs = []
    for config in vapi_configs:
        # Create a new Vapi assistant
        assistant_id = create_vapi_assistant(
            name=f"{config.assistant_name} - {config.customer_name}",
            system_prompt=config.system_prompt,
            first_message=config.first_message,
            analysis_plan=config.analysis_plan,
            tool_ids=tool_ids,
        )

        # Create updated config with assistant_id
        updated_config = config.model_copy(update={"assistant_id": assistant_id})
        updated_configs.append(updated_config)

    return {"vapi_configs": updated_configs}


# def configure_vapi_calls(state: LegislationState, config: RunnableConfig):
#     """Configure Vapi calls for each politician with a phone number."""
#     # Get state
#     analysis = state["analysis"]
#     politicians = state["politicians"]
#     call_scripts = state["call_scripts"]
#     legislation_path = state["legislation_path"]

#     # Get configuration
#     configurable = Configuration.from_runnable_config(config)
#     phone_number_id = configurable.vapi_phone_id
#     assistant_name = configurable.vapi_assistant_name
#     organization_name = configurable.vapi_organization_name

#     # If no phone number ID is configured, we can't make calls
#     if not phone_number_id:
#         return {"vapi_configs": []}

#     # Upload the legislation file directly from the path
#     file_id = upload_file_to_vapi(legislation_path)

#     # Create a query tool with the uploaded file
#     tool_id = create_query_tool(file_id)

#     analysis_plan = configurable.vapi_analysis_plan.to_dict()

#     # Configure Vapi calls for politicians with phone numbers
#     vapi_configs = []
#     for politician, script in zip(politicians, call_scripts):
#         if politician.phone_number:
#             # Format key points as a bulleted list
#             key_points_formatted = "\n".join([f"- {point}" for point in script.key_points])

#             # Format legislation analysis as a structured summary
#             legislation_analysis_formatted = f"""
#                 Summary: {analysis.summary}

#                 Animal Welfare Impact: {analysis.animal_welfare_impact}

#                 Recommended Actions:
#                 {chr(10).join([f"- {action}" for action in analysis.recommended_actions])}
#                 """

#             # Create system prompt with all context
#             system_prompt = vapi_system_prompt_template.format(
#                 assistant_name=assistant_name,
#                 organization_name=organization_name,
#                 ask=script.ask,
#                 politician_profile=politician.model_dump_json(),
#                 tldr_points=key_points_formatted, # TODO: add tldr points
#                 # tldr_points="\n".join([f"- {point}" for point in report.tldr_points]) TODO
#             )

#             # Set up LLM to enhance the system prompt
#             planner_provider = get_config_value(configurable.planner_provider)
#             planner_model = get_config_value(configurable.planner_model)
#             llm = init_chat_model(model=planner_model, model_provider=planner_provider)

#             # Prompt to enhance the system prompt
#             enhancement_prompt = f"""The below is the system prompt given to a vapi call agent. Improve it so that the agent behaves more humanlike and conversational and is sure to get to the point fast.
# Consider using a method of Extreme TLDR generation, a new form of extreme summarization for paragraphs. TLDR generation involves high source compression, removes stop words and summarizes the paragraph whilst retaining meaning. The result is the shortest possible summary that retains all of the original meaning and context of the paragraph.

# SYSTEM PROMPT TO IMPROVE:
# {system_prompt}

# IMPROVED SYSTEM PROMPT:"""

#             # Generate enhanced system prompt
#             enhanced_prompt_response = llm.invoke([
#                 HumanMessage(content=enhancement_prompt)
#             ])

#             # Extract the enhanced system prompt
#             enhanced_system_prompt = enhanced_prompt_response.content

#             # Create a new Vapi assistant for this call with the query tool
#             assistant_id = create_vapi_assistant(
#                 name=f"{assistant_name} - {politician.name}",
#                 system_prompt=system_prompt,
#                 first_message=script.first_message,
#                 end_call_message=script.end_call_message,
#                 analysis_plan=analysis_plan,
#                 tool_id=tool_id
#             )

#             # Configure the call
#             vapi_config = VapiCallConfig(
#                 assistant_id=assistant_id,
#                 phone_number_id=phone_number_id,
#                 customer_number=configurable.vapi_to_number or os.getenv('TEST_NUMBER'),
#                 customer_name=politician.name,
#                 first_message=script.first_message,
#                 system_prompt=system_prompt,
#                 enhanced_system_prompt=enhanced_system_prompt,
#                 assistant_name=assistant_name,
#                 organization_name=organization_name,
#                 end_call_message=script.end_call_message,
#                 analysis_plan=analysis_plan
#             )

#             vapi_configs.append(vapi_config)

#     return {"vapi_configs": vapi_configs}


def call_scripts_human_feedback(
    state: LegislationState,
) -> Command[Literal[END, "make_calls"]]:
    """Get human feedback on call scripts and configurations."""
    # Get state
    # analysis = state["analysis"]
    # politicians = state["politicians"]
    # call_scripts = state["call_scripts"]
    vapi_configs = state["vapi_configs"]

    review_info = f"""
        {vapi_configs}
    Approve these call scripts and proceed with calls? Reply 'yes' to approve or provide feedback for changes.
    """

    # # Format information for human review
    # review_info = f"""
    # ## Legislation Analysis

    # {analysis.summary}

    # ## Impact on Animal Welfare

    # {analysis.animal_welfare_impact}

    # ## Politicians and Call Scripts

    # """

    # for i, (politician, script) in enumerate(zip(politicians, call_scripts)):
    #     review_info += f"""
    #     ### {i+1}. {politician.name} ({politician.position})

    #     **Phone:** {politician.phone_number or "Not available"}

    #     **Script:**
    #     {script.full_script}

    #     """

    # Get feedback
    feedback = interrupt(f"""
    Please review the legislation analysis and call scripts:
    
    {review_info}
    
    Approve these call scripts and proceed with calls? Reply 'yes' to approve or provide feedback for changes.
    """)

    # Process feedback
    if isinstance(feedback, str) and feedback.lower() == "yes":
        # Approved - proceed with calls
        return Command(goto="make_calls", update={"approved_calls": vapi_configs})
    else:
        # Not approved - end workflow
        return Command(goto=END)


def make_calls(state: LegislationState, config: RunnableConfig):
    """Make approved calls using Vapi."""
    configurable = Configuration.from_runnable_config(config)

    # Get approved calls
    approved_calls = state["approved_calls"]

    # Make calls
    call_results = []
    for call_config in approved_calls:
        from_number = configurable.vapi_phone_id or os.getenv("VAPI_PHONE_ID")
        if not from_number:
            raise ValueError("No phone number ID provided. Please set vapi_phone_id in configuration or VAPI_PHONE_ID environment variable.")
            
        to_number = configurable.vapi_to_number or os.getenv("TEST_NUMBER")
        if not to_number:
            raise ValueError("No destination phone number provided. Please set vapi_to_number in configuration or TEST_NUMBER environment variable.")
        if call_config.assistant_id:
            result = make_vapi_call(
                assistant_id=call_config.assistant_id,
                phone_number_id=from_number,
                customer_number=to_number,  # call_config.customer_number,
            )
            call_results.append(result)

    return {"call_results": call_results}


# Report section sub-graph --

# Add nodes
section_builder = StateGraph(SectionState, output=SectionOutputState)
section_builder.add_node("generate_queries", generate_queries)
section_builder.add_node("search_web", search_web)
section_builder.add_node("write_section", write_section)

# Add edges
section_builder.add_edge(START, "generate_queries")
section_builder.add_edge("generate_queries", "search_web")
section_builder.add_edge("search_web", "write_section")

# Replaced with full research report code. Left as comments in case useful before demo

# Politician research sub-graph
politician_research = StateGraph(PoliticianResearchState, output=PoliticianResearchOutput)
politician_research.add_node("generate_politician_queries", generate_politician_queries)
politician_research.add_node("search_politician_info", search_politician_info)
politician_research.add_node("research_politician", research_politician)

# Add edges to the politician research sub-graph
politician_research.add_edge(START, "generate_politician_queries")
politician_research.add_edge("generate_politician_queries", "search_politician_info")
politician_research.add_edge("search_politician_info", "research_politician")
politician_research.add_edge("research_politician", END)

# Compile the politician research sub-graph
politician_research_graph = politician_research.compile()

# Inner graph for report generation --

# Add nodes
builder = StateGraph(
    ReportState,
    input=ReportStateInput,
    output=ReportStateOutput,
    config_schema=Configuration,
)
builder.add_node("generate_report_plan", generate_report_plan)
builder.add_node("human_feedback", human_feedback)
builder.add_node("build_section_with_web_research", section_builder.compile())
builder.add_node("gather_completed_sections", gather_completed_sections)
builder.add_node("write_final_sections", write_final_sections)
builder.add_node("compile_final_report", compile_final_report)

# Add edges
builder.add_edge(START, "generate_report_plan")
builder.add_edge("generate_report_plan", "human_feedback")
builder.add_edge("build_section_with_web_research", "gather_completed_sections")
builder.add_conditional_edges(
    "gather_completed_sections",
    initiate_final_section_writing,
    ["write_final_sections"],
)
builder.add_edge("write_final_sections", "compile_final_report")
builder.add_edge("compile_final_report", END)

# report_graph = builder.compile()

# politician_report_builder = StateGraph(PoliticianReport, output=ReportStateOutput)
# politician_report_graph = politician_report_builder.compile()

# Legislation analysis graph
legislation_analysis = StateGraph(
    LegislationState,
    input=LegislationStateInput,
    output=LegislationStateOutput,
    config_schema=Configuration,
)
legislation_analysis.add_node("analyze_legislation", analyze_legislation)
legislation_analysis.add_node("identify_politicians_to_research", identify_politicians_to_research)
legislation_analysis.add_node("research_politician", politician_research_graph)
legislation_analysis.add_node(
    "initiate_politician_reports", initiate_politician_reports
)
# legislation_analysis.add_node("generate_politician_report", politician_report_graph)
legislation_analysis.add_node("generate_report", builder.compile())
# legislation_analysis.add_node("add_report_to_trieve", add_report_to_trieve)
# legislation_analysis.add_node("generate_tldr_points", generate_tldr_points)
legislation_analysis.add_node("create_vapi_tools", create_vapi_tools)
legislation_analysis.add_node("prepare_vapi_configs", prepare_vapi_configs)
legislation_analysis.add_node("create_vapi_assistants", create_vapi_assistants)
legislation_analysis.add_node(
    "call_scripts_human_feedback", call_scripts_human_feedback
)
legislation_analysis.add_node("make_calls", make_calls)
# legislation_analysis.add_node("process_next_politician", process_next_politician)

# Add edges to the legislation analysis graph
legislation_analysis.add_edge(START, "analyze_legislation")
# legislation_analysis.add_edge("analyze_legislation", "initiate_politician_reports")
legislation_analysis.add_edge("analyze_legislation", "identify_politicians_to_research")
legislation_analysis.add_edge("research_politician", "initiate_politician_reports")
# legislation_analysis.add_edge("initiate_politician_reports", "generate_report")
# legislation_analysis.add_edge("analyze_legislation", politician_report_graph)
legislation_analysis.add_edge("generate_report", "create_vapi_tools")
legislation_analysis.add_edge("create_vapi_tools", "prepare_vapi_configs")
legislation_analysis.add_edge("prepare_vapi_configs", "create_vapi_assistants")
legislation_analysis.add_edge("create_vapi_assistants", "call_scripts_human_feedback")
legislation_analysis.add_edge("make_calls", END)

legislation_graph = legislation_analysis.compile()


# Initialize caching and persistence
def init_caching():
    """Initialize caching and persistence for the application."""
    # Set up LLM cache
    setup_llm_cache()

    # Set up embeddings with cache
    base_embeddings = OpenAIEmbeddings()
    cached_embeddings = setup_embedding_cache(base_embeddings, "legislation_analysis")

    # Set up persistent vector store
    vectorstore = setup_persistent_vectorstore(
        cached_embeddings, "legislation_analysis"
    )

    return vectorstore


# Choose which graph to use based on the application mode
graph = legislation_graph  # Default to legislation analysis
