from langchain.tools import tool
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langchain_tavily import TavilySearch
from qdrant_client import QdrantClient, models
import ollama
from RAG.config import client, collection_name, ensure_collection_ready
from RAG.ingest_doc import DEFAULT_USER_ID
from dotenv import load_dotenv
load_dotenv()

@tool
def RAG(query: str, user_id: str = DEFAULT_USER_ID) -> str:
    """
    This tool performs Retrieval-Augmented Generation (RAG) by querying a vector database and generating a response based on the retrieved information.
    
    Args:
        query (str): The input query for which you want to perform RAG.
        user_id (str): The unique identifier for the user, used to filter the search results to ensure data isolation.
    Returns:
        str: The generated response based on the retrieved information.
    """
    user_id = DEFAULT_USER_ID
    ensure_collection_ready()
    print("Calling RAG Tool",file=sys.stderr)
    query_vector = ollama.embed(
                model='bge-m3',
                input=query,
            )
    search_results = client.query_points(
        collection_name=collection_name,
        query=query_vector.embeddings[0],  
        limit=3,
        # This Filter is where the Payload Index proves its worth.
        # It guarantees this search ONLY operates within this specific user's isolated data.
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=user_id),
                )
            ]
        )
    )
    
    print(f"\n--- Results for {user_id} ---")

    points = getattr(search_results, "points", search_results)
    if not points:
        return "The requested information is not available in the internal documents."

    summaries: list[str] = []
    for hit in points:
        payload = getattr(hit, "payload", {}) or {}
        text = payload.get("text", "")
        if text:
            summaries.append(text)

    if not summaries:
        return "The requested information is not available in the internal documents."

    return "\n\n".join(summaries)

from typing import Any
@tool
def Tavily_Search(query: Any) -> str:
    """
    This tool performs a web search operation based on the input query.
    """
    print("Calling Tavily Search",file=sys.stderr)
    tavily_search = TavilySearch(api_key=os.getenv("TAVILY_API_KEY"),max_results=2)
    response = tavily_search.invoke({"query": query})
    return {"results": response}
