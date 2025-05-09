An WIP MVP done for a Stealth Startup (I have permission to make this public). This application is a multi-agent RAG system that analyzes input legislation, deep-researches politicians that sponsored the bill (their voting history, their political stances, ways of reaching said politician etc), and generates personalized AI voice assistants calibrated to be agile during conversation and to be precisely calibrated to the deep-research uncovered on the politician being called.


[Video demonstration](https://youtu.be/wNLGqJwOW9A?si=jb5RM8aOvsNTelhA)

# Running Locally

- `cp .env.example .env` and fill out required env variables. You will need to have accounts on OpenAI, [Vapi](https://vapi.ai/), [Langsmith](https://www.langchain.com/langsmith), [Trieve](https://trieve.ai/) and at least one of [Tavily](https://tavily.com/), [Perplexity](https://www.perplexity.ai/) or [Exa](https://exa.ai/) to retrieve required API keys to run locally.
- `uv sync`
- `uv run langgraph dev`
- Add a Legislation Path and Issue of Concern for the AI to focus its research on

<img width="558" alt="image" src="https://github.com/user-attachments/assets/63eaa741-1ada-4620-aac4-edf3ed46d897" />


# Proposed Architecture

Below is a proposed architecture that guided development. The code here currently implements Politician Analysis, Call Orchestration, Two-Way Conversation. Preliminary Logging & Reporting was additionally done through [Vapi](https://vapi.ai/), [Make](https://www.make.com/en) and Google Sheets.

![Architecture](https://github.com/Frikster/Pressure.AI/blob/main/architecture.png)
