from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient

from qdrant import COLLECTION_NAME, EMBEDDING_MODEL, VECTOR_SIZE, get_mineral_summaries

load_dotenv()

TOP_K = 3  # how many of the closest windows to hand to the LLM


def retrieve_windows(question: str, top_k: int = TOP_K) -> list[dict]:
    """Step 1 (Retrieval): find the windows in Qdrant closest in meaning to the question."""
    # Embed the question with the SAME model & size used to store the windows,
    # otherwise the numbers aren't comparable (and Qdrant rejects the size)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, dimensions=VECTOR_SIZE)
    question_vector = embeddings.embed_query(question)

    client = QdrantClient(host="localhost", port=6333)
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=question_vector,
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


def answer_question(question: str, mineral_table: str, windows: list[dict]) -> str:
    """Steps 2 & 3 (Augment + Generate): give the mineral table + retrieved windows to the LLM."""
    context = "\n\n=====\n\n".join(
        f"[Source: {w.get('source', 'Unknown')}]\n{w['content']}" for w in windows
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a geologist's assistant. Answer using ONLY the information provided.\n"
                    "- The MINERAL SUMMARY TABLE covers EVERY drill hole. Use it for comparison questions "
                    "(most, least, highest, which hole...). Holes marked 'no minerals recorded' have no minerals logged or assayed, which is not "
                    "proof of zero, so say they have no RECORDED gold (or other mineral), not that they have none.\n"
                    "- The DRILL LOG EXTRACTS are only the few passages closest to the question. "
                    "Use them for descriptive detail, and mention the source file.\n"
                    "- If 'most' is ambiguous, give both the highest grade and the length-weighted average.\n"
                    "- If the information doesn't contain the answer, say you don't know."
                ),
            ),
            (
                "human",
                (
                    "MINERAL SUMMARY TABLE:\n{mineral_table}\n\n"
                    "DRILL LOG EXTRACTS:\n\n{context}\n\nQuestion: {question}"
                ),
            ),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    chain = prompt | llm | StrOutputParser()

    return chain.invoke(
        {"mineral_table": mineral_table, "context": context, "question": question}
    )


if __name__ == "__main__":
    question = input("Ask a question about the drill logs: ").strip()
    if not question:
        raise ValueError("The question cannot be empty.")

    # every hole's mineral summary (small: one record per hole), so comparisons see ALL holes
    hole_summaries = get_mineral_summaries()
    if not hole_summaries:
        raise ValueError("No mineral summaries found. Run 'main2 - import.py' first.")
    mineral_table = build_mineral_table(hole_summaries)
    print(f"\n⛏️  Loaded mineral summaries for {len(hole_summaries)} drill holes:")
    print(mineral_table)

    windows = retrieve_windows(question)
    if not windows:
        raise ValueError(
            f"No data found in '{COLLECTION_NAME}'. Run 'main2 - import.py' first."
        )

    print(f"\n🔎 Retrieved {len(windows)} closest windows:")
    for idx, w in enumerate(windows, start=1):
        print(f"   {idx}. {w.get('source', 'Unknown')} (similarity {w['score']:.3f})")

    print("\n💬 Answer:")
    print(answer_question(question, mineral_table, windows))
