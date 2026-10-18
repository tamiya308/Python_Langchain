import importlib

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from qdrant import OPTIMIZED_WINDOWS_COLLECTION_NAME, MINERAL_COLLECTION_NAME, get_mineral_summaries

# reuse the retrieval & table logic from main5 (the filename has spaces, so a normal import won't work)
rag = importlib.import_module("main5 - RAG")

load_dotenv()

MAX_STEPS = 6  # safety limit on how many rounds of tool calls the agent may make


# ---------------------------------------------------------------------------
# Tools: each Qdrant collection is exposed as a tool. The LLM reads the
# docstrings to decide WHICH tool to call, with WHAT arguments, and how often.
# ---------------------------------------------------------------------------


@tool
def search_drill_logs(query: str, top_k: int = 3) -> str:
    """Semantic search over the raw drill-log text (the 'optimized_windows' collection).
    Use for descriptive questions: lithology, alteration, veining, what was observed at
    a given depth or in a given hole. Returns only the few passages closest to the query,
    so it is NOT suitable for comparing all holes. You may call it several times with
    different, more specific queries (e.g. include the hole id)."""
    windows = rag.retrieve_windows(query, top_k=top_k)
    if not windows:
        return f"No data found in '{OPTIMIZED_WINDOWS_COLLECTION_NAME}'."
    return "\n\n=====\n\n".join(
        f"[Source: {w.get('source', 'Unknown')} | similarity {w['score']:.3f}]\n{w['content']}"
        for w in windows
    )


@tool
def get_mineral_table() -> str:
    """Exact mineral figures for EVERY drill hole (the 'mineral_summary' collection):
    sample count, metres sampled, highest grade and length-weighted average per mineral.
    Use for comparison questions (most, least, highest, which hole, rank the holes) and
    for any question needing exact assay numbers. Figures are computed in Python, so trust them."""
    summaries = get_mineral_summaries()
    if not summaries:
        return f"No mineral summaries found in '{MINERAL_COLLECTION_NAME}'."
    return rag.build_mineral_table(summaries)


TOOLS = {t.name: t for t in [search_drill_logs, get_mineral_table]}

SYSTEM_PROMPT = (
    "You are a geologist's assistant answering questions about drill logs.\n"
    "You have tools that query different Qdrant collections. Decide which tool(s) you need, "
    "call them, and look at the results before answering. Call more tools if the results "
    "are insufficient (e.g. search again with a more specific query).\n"
    "- For comparisons across holes, ALWAYS use get_mineral_table: search_drill_logs only "
    "returns a few passages and cannot see every hole.\n"
    "- Holes marked 'no minerals recorded' have no RECORDED minerals, which is not proof of zero.\n"
    "- If 'most' is ambiguous, give both the highest grade and the length-weighted average.\n"
    "- When you use drill-log passages, mention the source file.\n"
    "- Answer using ONLY information returned by the tools. If it doesn't contain the answer, "
    "say you don't know."
)


def run_agent(question: str) -> str:
    """The agent loop: the LLM picks tools, we run them, feed results back, repeat until it answers."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0).bind_tools(list(TOOLS.values()))
    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(question)]

    for step in range(1, MAX_STEPS + 1):
        ai_message = llm.invoke(messages)
        messages.append(ai_message)

        # no tool calls means the LLM has decided it has enough information to answer
        if not ai_message.tool_calls:
            return ai_message.content

        for call in ai_message.tool_calls:
            args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
            print(f"🛠️  Step {step}: {call['name']}({args})")
            # invoking a tool with the tool-call dict returns a ToolMessage linked to that call
            messages.append(TOOLS[call["name"]].invoke(call))

    return f"Stopped after {MAX_STEPS} steps without a final answer."


if __name__ == "__main__":
    question = input("Ask a question about the drill logs: ").strip()
    if not question:
        raise ValueError("The question cannot be empty.")

    print()
    answer = run_agent(question)
    print("\n💬 Answer:")
    print(answer)
