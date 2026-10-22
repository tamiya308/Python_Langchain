from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from qdrant import (
    EMBEDDING_MODEL,
    MINERAL_COLLECTION_NAME,
    OPTIMIZED_WINDOWS_COLLECTION_NAME,
    TEXT_VECTOR_NAME,
    VECTOR_SIZE,
    get_mineral_summaries,
    get_qdrant_client,
)

load_dotenv()

MAX_STEPS = 6  # safety limit on how many rounds of tool calls the agent may make
TOP_K = 3  # default number of closest windows search_drill_logs returns


def retrieve_windows(question: str, top_k: int = TOP_K) -> list[dict]:
    """Step 1 (Retrieval): find the windows in Qdrant closest in meaning to the question."""
    # Embed the question with the SAME model & size used to store the windows,
    # otherwise the numbers aren't comparable (and Qdrant rejects the size)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, dimensions=VECTOR_SIZE)
    question_vector = embeddings.embed_query(question)

    client = get_qdrant_client()
    response = client.query_points(
        collection_name=OPTIMIZED_WINDOWS_COLLECTION_NAME,
        query=question_vector,
        using=TEXT_VECTOR_NAME,  # search the drill log text, not the drill images
        limit=top_k,
        with_payload=True,
    )

    return [{"score": point.score, **point.payload} for point in response.points]


def build_mineral_table(hole_summaries: list[dict]) -> str:
    """
    Turns every hole's mineral summary into a short table of exact figures.
    The maths is done here in Python (not by the LLM), so comparisons like
    "most gold" are based on correct numbers for EVERY hole.
    """
    lines = []
    for summary in sorted(hole_summaries, key=lambda s: s["hole_id"]):
        hole = summary["hole_id"]
        if not summary["minerals"]:
            lines.append(f"{hole} | no minerals recorded")
            continue

        # group the rows for this hole by mineral (e.g. all the Au samples together)
        groups: dict[tuple, list[dict]] = {}
        for row in summary["minerals"]:
            key = (row["mineral"], row["unit"], row["basis"])
            groups.setdefault(key, []).append(row)

        for (mineral, unit, basis), rows in groups.items():
            measured = [r for r in rows if r["amount"] is not None]
            if not measured:
                depths = ", ".join(f"{r['from_depth']:g}-{r['to_depth']:g}m" for r in rows)
                lines.append(f"{hole} | {mineral} | {basis} | noted ({unit}) at {depths}")
                continue

            best = max(measured, key=lambda r: r["amount"])
            metres = sum(r["to_depth"] - r["from_depth"] for r in measured)
            weighted_avg = (
                sum(r["amount"] * (r["to_depth"] - r["from_depth"]) for r in measured) / metres
                if metres
                else best["amount"]
            )
            lines.append(
                f"{hole} | {mineral} | {basis} | {len(measured)} samples over {metres:g}m | "
                f"highest {best['amount']:g} {unit} at {best['from_depth']:g}-{best['to_depth']:g}m | "
                f"length-weighted average {weighted_avg:.2f} {unit}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tools: each Qdrant collection is exposed as a tool. The LLM reads the
# docstrings to decide WHICH tool to call, with WHAT arguments, and how often.
# ---------------------------------------------------------------------------


@tool
def search_drill_logs(query: str, top_k: int = TOP_K) -> str:
    """Semantic search over the raw drill-log text (the 'optimized_windows' collection).
    Use for descriptive questions: lithology, alteration, veining, what was observed at
    a given depth or in a given hole. Returns only the few passages closest to the query,
    so it is NOT suitable for comparing all holes. You may call it several times with
    different, more specific queries (e.g. include the hole id)."""
    windows = retrieve_windows(query, top_k=top_k)
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
    return build_mineral_table(summaries)


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


def run_agent(question: str) -> dict:
    """The agent loop: the LLM picks tools, we run them, feed results back, repeat until it answers.
    Returns the answer plus the tool calls made along the way."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0).bind_tools(list(TOOLS.values()))
    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(question)]
    steps = []

    for step in range(1, MAX_STEPS + 1):
        ai_message = llm.invoke(messages)
        messages.append(ai_message)

        # no tool calls means the LLM has decided it has enough information to answer
        if not ai_message.tool_calls:
            return {"answer": ai_message.content, "steps": steps}

        for call in ai_message.tool_calls:
            args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
            print(f"🛠️  Step {step}: {call['name']}({args})")
            steps.append({"step": step, "tool": call["name"], "args": call["args"]})
            # invoking a tool with the tool-call dict returns a ToolMessage linked to that call
            messages.append(TOOLS[call["name"]].invoke(call))

    return {"answer": f"Stopped after {MAX_STEPS} steps without a final answer.", "steps": steps}
