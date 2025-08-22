from typing import TypedDict, List, Dict, Any, Optional
import os
import json
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv() 

# LLM imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

# LangGraph
from langgraph.graph import StateGraph, END

# --- Config ---
SEARCH_BACKEND = os.getenv("SEARCH_BACKEND", "tavily")
TOP_K = int(os.getenv("TOP_K", "8"))
MAX_REVISIONS = 3

# --- Search helpers (Tavily) ---
def _search_tavily(query: str, k: int) -> List[Dict[str,str]]:
    from tavily import TavilyClient
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    resp = client.search(query=query, max_results=k)
    items = []
    for r in resp.get("results", []):
        items.append({
            "title": r.get("title") or "",
            "url": r.get("url") or r.get("link") or "",
            "snippet": r.get("content") or r.get("snippet") or ""
        })
    return items

def run_search(query: str, k: int = TOP_K) -> List[Dict[str,str]]:
    if SEARCH_BACKEND == "tavily":
        return _search_tavily(query, k)
    # Add other backends here (duckduckgo, serpapi, etc.)
    raise RuntimeError(f"SEARCH_BACKEND={SEARCH_BACKEND} not implemented")

# --- Typed State ---
class LitState(TypedDict, total=False):
    query: str
    search_results: List[Dict[str,str]]
    fetched_documents: List[Dict[str, Any]]
    draft_summary: str
    reviews: Dict[str, Any]
    formatted: str
    iterations: int

# --- Search node ---
def search_node(state: LitState) -> LitState:
    q = state.get("query", "").strip()
    if not q:
        return {}
    results = run_search(q, TOP_K)
    return {"search_results": results}

# --- LLM instances ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
review_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
formatter_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# --- fetch + clean pages ---
def fetch_page_text(url: str, max_chars: int = 2000) -> str:
    try:
        headers = {"User-Agent": "AcademicHelper/1.0 (+https://example.org)"}
        r = requests.get(url, timeout=7, headers=headers)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for script in soup(["script", "style"]):
            script.extract()
        text = " ".join(s.strip() for s in soup.stripped_strings)
        return text[:max_chars]
    except Exception:
        return ""

# --- Summarizer node ---
def summarizer_node(state: LitState) -> LitState:
    # Build fetched_docs from search_results if not already present
    fetched_docs: List[Dict[str, Any]] = []
    for r in state.get("search_results", []):
        url = r.get("url", "")
        if not url:
            continue
        text = fetch_page_text(url)
        if text:
            fetched_docs.append({"url": url, "text": text})
    combined_text = "\n\n".join([d["text"] for d in fetched_docs])[:15000]  # safety trunc

    # Incorporate reviewer feedback if this is a revision
    feedback = state.get("reviews", {}).get("feedback", "")
    feedback_section = ""
    if feedback:
        feedback_section = f"\n\nPrevious reviewer feedback:\n{feedback}\n\nPlease revise the summary to address the feedback."

    prompt = f"""
You are an academic assistant. Summarize the following research content into 200–300 words for a literature review.
Capture key findings, methods, and results and keep it concise. Preserve references by mentioning URLs where relevant.

{feedback_section}

Content:
{combined_text}

Write the summary ONLY (no extra commentary).
"""
    resp = llm.invoke([HumanMessage(content=prompt)])
    draft_summary = resp.content.strip() if getattr(resp, "content", None) is not None else str(resp)

    return {
        "fetched_documents": fetched_docs,
        "draft_summary": draft_summary,
        # if iterations already in state keep same, we'll increment in reviewer_node if needed
        "iterations": state.get("iterations", 0)
    }

# --- Reviewer node (LLM-powered) ---
review_prompt_template = ChatPromptTemplate.from_template("""
You are a strict academic reviewer. Evaluate the draft summary below.

Draft:
{draft}

Evaluate on:
- Clarity and readability
- Coverage of key research aspects (methods, results, implications)
- Accuracy (highlight if factual claims seem unsupported)
- Length appropriateness (<= 200 words preferred)

Respond STRICTLY in JSON exactly like:
{{
  "decision": "revise" or "approve",
  "feedback": "<short natural language critique; if approve, keep short>"
}}
""")

def reviewer_node(state: LitState) -> LitState:
    draft = state.get("draft_summary", "")
    if not draft:
        return {"reviews": {"decision": "revise", "feedback": "No draft provided"}}

    prompt_text = review_prompt_template.format(draft=draft)
    resp = review_llm.invoke([HumanMessage(content=prompt_text)])
    raw = getattr(resp, "content", str(resp))

    # Try to extract JSON from model output (robust)
    parsed = None
    try:
        parsed = json.loads(raw)
    except Exception:
        # Try to find a JSON substring
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            parsed = json.loads(raw[start:end])
        except Exception:
            parsed = {"decision": "revise", "feedback": "Could not parse reviewer output. Please improve draft."}

    # Ensure required fields exist
    decision = parsed.get("decision", "revise")
    feedback = parsed.get("feedback", "")

    # Increment iterations if the reviewer asked for revise
    iterations = state.get("iterations", 0)
    if decision == "revise":
        iterations += 1
        # If we've hit the max, override to approve with a warning
        if iterations >= MAX_REVISIONS:
            decision = "approve"
            feedback += " (Reached max revision attempts; accepting current draft.)"

    return {
        "reviews": {"decision": decision, "feedback": feedback},
        "iterations": iterations
    }

# --- Formatter node ---
formatter_prompt_template = ChatPromptTemplate.from_template(
    "Polish the following draft into a clean final summary suitable for academic writing (one paragraph):\n\n{draft}"
)

def formatter_node(state: LitState) -> LitState:
    draft = state.get("draft_summary", "")
    if not draft:
        return {"formatted": ""}
    prompt_text = formatter_prompt_template.format(draft=draft)
    resp = formatter_llm.invoke([HumanMessage(content=prompt_text)])
    final_summary = getattr(resp, "content", str(resp)).strip()
    return {"formatted": final_summary}

# --- Output node (returns formatted summary and metadata) ---
def output_node(state: LitState) -> LitState:
    return {
        # keep useful data for user / downstream:
        "formatted": state.get("formatted", ""),
        "draft_summary": state.get("draft_summary", ""),
        "reviews": state.get("reviews", {}),
        "iterations": state.get("iterations", 0),
        "search_results": state.get("search_results", [])[:TOP_K]
    }

# --- Build the graph and wire nodes ---
workflow = StateGraph(LitState)
workflow.add_node("search", search_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("formatter", formatter_node)
workflow.add_node("output", output_node)

# Entry: search -> summarizer
workflow.set_entry_point("search")
workflow.add_edge("search", "summarizer")

# Summarizer -> Reviewer
workflow.add_edge("summarizer", "reviewer")

# Reviewer branching: if reviews.decision == "revise" and iterations < MAX_REVISIONS -> summarizer else -> formatter
def reviewer_branch_key(state: LitState) -> str:
    reviews = state.get("reviews", {})
    decision = reviews.get("decision", "revise")
    iterations = state.get("iterations", 0)
    if decision == "revise" and iterations < MAX_REVISIONS:
        return "revise"
    return "approve"

workflow.add_conditional_edges(
    "reviewer",
    reviewer_branch_key,
    {"revise": "summarizer", "approve": "formatter"}
)

# Formatter -> Output -> END
workflow.add_edge("formatter", "output")
workflow.add_edge("output", END)

graph = workflow.compile()

# --- Example usage (main) ---
if __name__ == "__main__":
    initial_state: LitState = {
        "query": input("Enter your search query: ")
    }

    # Run the search summarization pipeline
    # First, run 'search' entry which cascades through the graph
    print("=== Running pipeline (search -> summarizer -> reviewer -> formatter -> output) ===")
    result = graph.invoke(initial_state)
    print("\n=== FINAL OUTPUT ===")
    print("Formatted summary:\n", result.get("formatted"))
    print("\nDraft summary (raw):\n", result.get("draft_summary"))
    print("\nReview metadata:\n", result.get("reviews"))
    print("\nIterations:\n", result.get("iterations"))
    print("\nSource links:")
    for i, r in enumerate(result.get("search_results", [])[:TOP_K], 1):
        print(f"{i}. {r.get('title','').strip()} - {r.get('url')}")
