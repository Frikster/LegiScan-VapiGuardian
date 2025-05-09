Developed for a stealth-mode startup. This application is an advanced multi-agent RAG system that analyzes legislation, researches sponsoring politicians' backgrounds, and generates personalized AI voice assistants. It is implemented using a Trieve vectorstore for efficient knowledge retrieval, enabling contextualized call scripts calibrated for political advocacy conversations tailored to each politician's specific positions and voting history. The call agents themselves are calibrated to be agile during conversation.

If you have information on what tech I can use to make the call agents more robustly adhere to guardrails please do reach out! This is a technical question I have yet to solve and am very eager to learn about. We reached an acceptable milestone with this MVP improving on what was currently being used by the Stealth Startup, nonetheless this MVP is not yet production-ready.


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
