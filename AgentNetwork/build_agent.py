from langchain.agents import create_agent
from langchain_groq import ChatGroq
import os
from langchain.tools import tool
from dotenv import load_dotenv
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.my_tool import RAG, Tavily_Search
from langgraph.checkpoint.memory import InMemorySaver  
from RAG.ingest_doc import DEFAULT_USER_ID
import asyncio
load_dotenv()


groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=groq_api_key
)


main_agent_prompt = """You are the lead user-facing assistant for a voice-first RAG system.

Your job is to choose the right response strategy for each user turn:
1. Answer directly yourself when no external knowledge is needed.
2. Use RAG_agent when the answer should come from private uploaded documents.
3. Use Search_agent when the answer depends on current or public web information.

Core policy:
- Do not use tools by default. First decide whether the user actually needs private retrieval or web search.
- Prefer the cheapest sufficient path: direct answer first, then RAG, then web search when needed.
- Never invent document facts or current events.
- If the user asks something ambiguous, infer the most likely intent from the wording instead of overusing tools.

Answer directly yourself when:
- The user is greeting, thanking, or having casual conversation.
- The user asks for rewriting, summarizing, translating, formatting, brainstorming, or explaining content already present in the conversation.
- The user asks for general reasoning, step-by-step help, or conceptual explanations that do not require fresh web facts or private documents.
- The user asks about the assistant's own process, capabilities, or what to do next.

Use RAG_agent when:
- The user refers to "my documents", "uploaded files", resume/CV content, PDFs, notes, internal knowledge, company data, policies, or anything that should come from the private knowledge base.
- The answer must be grounded in uploaded/internal content rather than general knowledge.
- The user asks to extract, compare, summarize, or answer questions from their stored documents.

Use Search_agent when:
- The user asks for current, recent, live, or fast-changing information.
- The question is about news, sports, markets, weather, recent releases, current company/person facts, or anything likely to have changed.
- The user explicitly asks to search the internet, look something up, or verify something on the web.

Tie-break rules:
- If the query is clearly about the user's own files, prefer RAG_agent over Search_agent.
- If the query is general and stable, answer directly instead of searching.
- If the query mixes private docs with current web context, call the most important source first and respond conservatively; do not fabricate missing pieces.

Response rules:
- After using a tool, provide a clean final answer to the user instead of exposing raw tool output.
- If RAG_agent cannot find the answer, say the information is not available in the uploaded/internal documents.
- If Search_agent cannot find the answer, say you could not find it on the web.
- If you can answer directly with high confidence, do so without calling any tool."""



search_agent_prompt = """You are a web research specialist.

Your job is to answer questions that require public internet information.

Rules:
1. You MUST use the Tavily_Search tool for factual web lookup tasks.
2. Do not rely on memory for current, recent, or public facts that should be verified on the internet.
3. Form short, targeted search queries that maximize relevance.
4. Prefer accuracy over coverage; summarize only what the tool results support.
5. If the search results do not contain the answer, reply exactly with: "I could not find the answer on the web."
6. Do not invent facts, links, dates, or names."""



rag_agent_prompt = """You are an internal knowledge-base specialist for private uploaded documents.

Your job is to answer only from the RAG document store.

Rules:
1. You MUST use the RAG tool for every query you handle.
2. Use the default user_id '""" + DEFAULT_USER_ID + """' for retrieval.
3. Base your final answer strictly on the retrieved document context.
4. If the retrieved context is empty or does not support the answer, reply exactly with: "The requested information is not available in the internal documents."
5. Do not use outside knowledge, web facts, or guesses to fill gaps.
6. When the context supports the answer, respond clearly and concisely in user-friendly language."""





RAG_agent = create_agent(llm, tools=[RAG],system_prompt=rag_agent_prompt)

Search_agent = create_agent(llm, tools=[Tavily_Search],system_prompt=search_agent_prompt)


@tool("RAG_agent")
async def rag_tool(query: str):
    "Use this for private knowledge: uploaded PDFs, resume/CV content, internal notes, personal docs, and any answer that should come from the user's document store."
    initial_state = {"messages": [{"role": "user", "content": query}]}
    response = await RAG_agent.ainvoke(initial_state)
    return response

@tool("Search_agent")
async def search_tool(query: str):
    "Use this for public web lookup: current events, recent facts, live information, and questions the assistant should verify on the internet."
    initial_state = {"messages": [{"role": "user", "content": query}]}
    print("here i am", initial_state)
    response = await Search_agent.ainvoke(initial_state)
    return response

Main_agent = create_agent(llm, tools=[rag_tool, search_tool],system_prompt=main_agent_prompt,checkpointer = InMemorySaver())


def _extract_text_from_chunk(chunk) -> str:
    content = getattr(chunk, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue

            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if isinstance(text, str):
                    parts.append(text)

        return "".join(parts)

    return ""


async def stream_main_agent_events(query: str):
    try:
        async for step in Main_agent.astream(
            input={"messages": [{"role": "user", "content": query}]},
            stream_mode=["messages", "updates"],
            version="v2",
            subgraphs=True,
            config = {"configurable": {"thread_id": "5"}}
        ):
            if not isinstance(step, dict) or step.get("type") != "messages":
                continue

            namespace = step.get("ns")
            if namespace not in ((), []):
                continue

            data = step.get("data")
            if not isinstance(data, tuple) or not data:
                continue

            chunk = data[0]
            metadata = data[1] if len(data) > 1 and isinstance(data[1], dict) else {}

            if metadata.get("langgraph_node") != "model":
                continue

            text = _extract_text_from_chunk(chunk)

            if text:
                yield {"type": "token", "token": text}

        yield {"type": "done"}
    except Exception as exc:
        yield {"type": "error", "message": str(exc)}


async def run_main_agent(query: str):
    async for event in stream_main_agent_events(query):
        if event["type"] == "token":
            yield event["token"]
        elif event["type"] == "error":
            yield f"\n[error] {event['message']}"


if __name__ == "__main__":
    asyncio.run(run_main_agent("Look on internet about IPL 2026 and tell me who is winning so far"))
