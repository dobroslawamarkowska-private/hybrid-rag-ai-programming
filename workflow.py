"""
Advanced RAG LangGraph workflow following the diagram:
  Pre-Retrieval (smart LLM) → Retrieval (cheap embeddings) → Post-Retrieval (smart LLM) → Generate (frozen LLM)

Retrieval: async + shared retriever – równoległe ainvoke dla każdego expanded query (asyncio.gather).
"""

import asyncio
from typing import TypedDict

from dotenv import load_dotenv

load_dotenv(override=True)  # przed importem LangChain (LangSmith observability)

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from config import (
    GRADER_LLM_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    RETRIEVAL_MAX_WORKERS,
    SMART_LLM_MODEL,
)
from retriever import get_shared_retriever


# --- State ---
class RAGState(TypedDict, total=False):
    query: str
    route: str  # "direct" | "rag"
    expanded_queries: list[str]
    raw_docs: list
    reranked_docs: list
    context: str
    answer: str
    retrieval_attempt: int


# --- Models ---
def _get_smart_llm():
    return ChatOpenAI(
        model=SMART_LLM_MODEL,
        temperature=0,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )


def _get_grader_llm():
    """LLM do gradera (Check & Refine) – lepszy model do oceny relewancji."""
    return ChatOpenAI(
        model=GRADER_LLM_MODEL,
        temperature=0,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )


# --- Route: direct answer vs RAG ---
ROUTE_PROMPT = """Decide whether this question needs Docker documentation search or can be answered from general knowledge.

Answer DIRECT if: the question is general (e.g. "What is Docker?", "Co to jest kontener?"), introductory, or about basic concepts.
Answer RAG if: the question asks for specific instructions, commands, configuration, API details, step-by-step guides, or anything that requires looking up documentation.

Question: {query}

Reply with exactly one word: DIRECT or RAG"""


def route_query(state: RAGState) -> dict:
    """LLM decides: answer directly (no retrieval) or run RAG pipeline."""
    query = state["query"]
    print("\n[DEBUG route_query] IN:  query =", repr(query))
    llm = _get_smart_llm()
    prompt = ChatPromptTemplate.from_messages([("human", ROUTE_PROMPT)])
    chain = prompt | llm
    response = chain.invoke({"query": query})
    route = "rag"
    if response.content and "direct" in response.content.strip().lower():
        route = "direct"
    print("[DEBUG route_query] OUT: route =", route)
    return {"route": route}


def _route_to_direct_or_rag(state: RAGState) -> str:
    """Route to direct answer or RAG pipeline."""
    return "generate_direct" if state.get("route") == "direct" else "pre_retrieval"


# --- Pre-Retrieval: Query Routing, Rewriting, Expansion ---
PRE_RETRIEVAL_PROMPT = """You are a query optimizer for a Docker documentation search system.

Given the user query, produce 1-3 optimized search queries that will retrieve the most relevant documentation chunks.
- Keep the original intent
- Add technical keywords if helpful (e.g. "Dockerfile", "volume", "compose")
- Use different phrasings to improve recall
- Output ONLY the queries, one per line, no numbering or bullets"""


def pre_retrieval(state: RAGState) -> dict:
    """Query rewriting and expansion using smart LLM."""
    print("\n[DEBUG pre_retrieval] IN:  query =", repr(state["query"]))
    llm = _get_smart_llm()
    prompt = ChatPromptTemplate.from_messages([("human", PRE_RETRIEVAL_PROMPT + "\n\nUser query: {query}")])
    chain = prompt | llm
    response = chain.invoke({"query": state["query"]})
    lines = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
    queries = lines[:3] if lines else [state["query"]]
    print("[DEBUG pre_retrieval] OUT: expanded_queries =", queries)
    return {"expanded_queries": queries}


# --- Retrieval: Async + shared retriever (asyncio.gather) ---
async def _retrieval_task(retriever, query: str) -> list[Document]:
    """Async task: retriever.ainvoke dla jednego query."""
    return await retriever.ainvoke(query)


async def retrieval_async(state: RAGState) -> dict:
    """
    Async retrieval: współdzielony retriever (thread-safe singleton), równoległe ainvoke
    przez asyncio.gather dla każdego expanded query.
    """
    queries = state["expanded_queries"]
    n_parallel = min(len(queries), RETRIEVAL_MAX_WORKERS)
    print("\n[DEBUG retrieval] IN:  expanded_queries =", queries, "| async tasks =", n_parallel)

    retriever = get_shared_retriever(k=6)
    tasks = [_retrieval_task(retriever, q) for q in queries]
    results = await asyncio.gather(*tasks)

    all_docs: list[Document] = []
    seen_ids: set[int] = set()
    for docs in results:
        for d in docs:
            cid = hash(d.page_content[:200])
            if cid not in seen_ids:
                seen_ids.add(cid)
                all_docs.append(d)

    titles = [d.metadata.get("title", "?") for d in all_docs[:6]]
    print("[DEBUG retrieval] OUT: raw_docs count =", len(all_docs), "| titles (first 6) =", titles)
    return {"raw_docs": all_docs}


def retrieval(state: RAGState) -> dict:
    """Sync wrapper – uruchamia retrieval_async."""
    return asyncio.run(retrieval_async(state))


# --- Check relevance & refine query if docs are bad (grader 0–1) ---
RELEVANCE_THRESHOLD = 0.5  # score >= 0.5 → docs ok; score < 0.5 → refine

CHECK_AND_REFINE_PROMPT = """You are a grader evaluating whether retrieved Docker documentation chunks are relevant to the user's question.

User question: {query}

First 3 retrieved chunk titles (preview):
{chunk_preview}

Answer in exactly 2 lines:
1. SCORE: a number from 0.00 to 1.00 (0.00=completely irrelevant, 1.00=perfectly relevant), exactly 2 decimal places
2. REFINED: if SCORE < 0.50, write an improved question that:
   - clarifies the user's intent
   - adds missing context
   - removes ambiguity
   - specifies attributes/details
   If SCORE >= 0.50, write the original question unchanged.

Example format:
SCORE: 0.35
REFINED: How to install Docker Engine on Ubuntu Linux step by step?"""


def _parse_grader_response(response_text: str) -> tuple[float, str]:
    """Parse SCORE (0.00–1.00) and REFINED from LLM response. Returns (score, refined_query)."""
    lines = response_text.strip().split("\n")
    score = 0.5  # default: pass
    refined = ""
    for line in lines:
        line_upper = line.strip().upper()
        if line_upper.startswith("SCORE:"):
            try:
                raw = line.split(":", 1)[1].strip().strip(".")
                num = float(raw.replace(",", "."))
                score = round(max(0.0, min(1.0, num)), 2)
            except (ValueError, IndexError):
                pass
        elif line_upper.startswith("REFINED:"):
            refined = line.split(":", 1)[1].strip()
    return score, refined


def check_and_refine_query(state: RAGState) -> dict:
    """Grader 0–1: if score < threshold, LLM refines the query."""
    raw_docs = state.get("raw_docs", [])
    query = state["query"]
    attempt = state.get("retrieval_attempt", 0)
    print("\n[DEBUG check_and_refine] IN:  raw_docs count =", len(raw_docs), "| retrieval_attempt =", attempt)

    if not raw_docs or attempt >= 1:
        print("[DEBUG check_and_refine] OUT: docs_ok = True (no retry)")
        return {"retrieval_attempt": 0}

    chunk_preview = "\n".join(
        f"- {d.metadata.get('title', '?')}: {d.page_content[:80]}..." for d in raw_docs[:3]
    )
    llm = _get_grader_llm()
    prompt = ChatPromptTemplate.from_messages([("human", CHECK_AND_REFINE_PROMPT)])
    chain = prompt | llm
    response = chain.invoke({"query": query, "chunk_preview": chunk_preview})
    score, refined = _parse_grader_response(response.content)
    docs_ok = score >= RELEVANCE_THRESHOLD

    if docs_ok:
        print("[DEBUG check_and_refine] OUT: docs_ok = True | score =", score)
        return {"retrieval_attempt": 0}

    if not refined:
        refined = query
    print("[DEBUG check_and_refine] OUT: docs_ok = False | score =", score, "| refined_query =", repr(refined))
    return {"query": refined, "retrieval_attempt": 1, "expanded_queries": [refined]}


def _route_after_check(state: RAGState) -> str:
    """Route: retrieval (retry) if we just refined, else post_retrieval."""
    # If retrieval_attempt was set to 1, we refined the query and need to re-retrieve
    if state.get("retrieval_attempt") == 1:
        return "retrieval"
    return "post_retrieval"
def post_retrieval(state: RAGState) -> dict:
    """Rerank and prepare context. Use smart LLM to compress if needed."""
    docs = state["raw_docs"]
    print("\n[DEBUG post_retrieval] IN:  raw_docs count =", len(docs))
    # RRF: treat each query's results as a list (simplified: we merged already, so just take top by diversity)
    # Keep top 6 most relevant
    reranked = docs[:8]
    context = "\n\n---\n\n".join(
        f"[{i+1}] (from: {d.metadata.get('title', '?')})\n{d.page_content}" for i, d in enumerate(reranked[:6])
    )
    print("[DEBUG post_retrieval] OUT: reranked count =", len(reranked), "| context length =", len(context), "chars")
    return {"reranked_docs": reranked, "context": context}


# --- Generate: Frozen LLM ---
GENERATE_PROMPT = """You are a helpful Docker documentation assistant. Answer the user's question based ONLY on the provided context.

If the context does NOT contain the exact information asked for:
1. Clearly state that such information is not in the documentation.
2. Suggest the closest related information from the context (e.g. "Najbliższa propozycja:" / "The closest match:").

If the context contains relevant information, answer concisely and technically.

Context:
{context}

User question: {query}

Answer:"""


def generate(state: RAGState) -> dict:
    """Generate final answer using frozen smart LLM."""
    llm = _get_smart_llm()
    prompt = ChatPromptTemplate.from_messages([("human", GENERATE_PROMPT)])
    chain = prompt | llm
    response = chain.invoke({"context": state["context"], "query": state["query"]})
    return {"answer": response.content}


# --- Build graph ---
def build_rag_graph():
    builder = StateGraph(RAGState)

    builder.add_node("pre_retrieval", pre_retrieval)
    builder.add_node("retrieval", retrieval)
    builder.add_node("check_and_refine", check_and_refine_query)
    builder.add_node("post_retrieval", post_retrieval)
    builder.add_node("generate", generate)

    builder.add_edge(START, "pre_retrieval")
    builder.add_edge("pre_retrieval", "retrieval")
    builder.add_edge("retrieval", "check_and_refine")
    builder.add_conditional_edges("check_and_refine", _route_after_check, ["retrieval", "post_retrieval"])
    builder.add_edge("post_retrieval", "generate")
    builder.add_edge("generate", END)

    return builder.compile()


# --- Entry point ---
_graph = None


def get_rag_graph():
    global _graph
    if _graph is None:
        _graph = build_rag_graph()
    return _graph


def ask(query: str) -> str:
    """Run the RAG workflow and return the answer."""
    graph = get_rag_graph()
    result = graph.invoke({"query": query})
    return result.get("answer", "")


if __name__ == "__main__":
    q = "How can I persist data in Docker containers?"
    print("Query:", q)
    print("\nAnswer:\n", ask(q))
