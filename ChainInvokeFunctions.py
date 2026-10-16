import uuid
from typing import Any  # Added Optional import here

from langchain_classic.retrievers import MultiVectorRetriever
from langchain_classic.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field


class LithologyExtraction(BaseModel):
    hole_id: str = Field(
        description="The unique identifier or borehole code extracted from log header."
    )
    rock_type: str = Field(
        description="Dominant rock type or soil composition identified."
    )
    texture: str = Field(
        description="Rock matrix texture profile (e.g., foliated, aphanitic, clastic)."
    )
    grain_size: str = Field(description="Grain classification range observed.")
    key_anomalies: list[str] = Field(
        description="Notable geological structural observations, mineralization, or fractures."
    )


# Rebuild schema to resolve any internal type hints before LangChain schema conversion
LithologyExtraction.model_rebuild()


class LogScanRecord(BaseModel):
    drill_hole_id: str | None = Field(
        default="Unknown", description="The ID of the drill hole (e.g., DDH-26-01)"
    )
    main_rock_type: str = Field(
        description="The primary lithology / rock type identified"
    )
    # raw_content: str = Field(description="The original unstructured message content")
    # total_tokens_used: int = Field(default=0, description="Tokens consumed for this specific inference window")


# Rebuild schema to safely handle typing references and fix the PydanticUserError crash
LogScanRecord.model_rebuild()

from pydantic import BaseModel, Field


class IntervalLog(BaseModel):
    start_depth: float = Field(
        description="The starting depth of the interval in meters."
    )
    end_depth: float = Field(description="The ending depth of the interval in meters.")
    rock_type: str = Field(description="The classification or type of rock/lithology.")
    description: str = Field(
        description="Full text describing texture, grain size, and alteration."
    )


class DrillHoleLog(BaseModel):
    hole_id: str = Field(
        description="The identifier of the drill hole, e.g., DDH-26-01."
    )
    prospect: str = Field(description="The name of the exploration prospect area.")
    eastings: float = Field(description="The grid Easting coordinate.")
    northings: float = Field(description="The grid Northing coordinate.")
    elevation: float = Field(description="The elevation or RL value in meters.")
    azimuth: float = Field(description="The azimuth direction in degrees.")
    dip: float = Field(description="The dip angle in degrees (usually negative).")
    total_depth: float = Field(description="The total depth of the hole in meters.")
    intervals: list[IntervalLog] = Field(
        description="The sequential list of all logged geological depth intervals."
    )


class MineralOccurrence(BaseModel):
    hole_id: str = Field(description="The identifier of the drill hole, e.g., DDH-26-01.")
    mineral: str = Field(
        description="Element symbol and name, or mineral name, e.g., 'Au (gold)', 'As (arsenic)', 'pyrite'."
    )
    amount: float | None = Field(
        description="The amount found as a number, or null if only noted without a value (e.g., 'trace')."
    )
    unit: str = Field(
        description="Unit of the amount, e.g., 'g/t', 'ppm', '%', or 'trace' when no number is given."
    )
    from_depth: float = Field(description="Start depth of the interval in meters.")
    to_depth: float = Field(description="End depth of the interval in meters.")
    basis: str = Field(
        description="'assay' for laboratory results, or 'visual' for minerals noted in the geological log."
    )


class MineralSummary(BaseModel):
    hole_id: str = Field(description="The identifier of the drill hole, e.g., DDH-26-01.")
    occurrences: list[MineralOccurrence] = Field(
        description="Every mineral or element occurrence found anywhere in the drill log, one per interval or sample."
    )


def retrieve_parent_documents(
    parent_docs: list[Document], semantic_docs: list[Document], query: str, k: int = 4
) -> list[Document]:
    """Index semantic child segments and return their full-file parents."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = InMemoryVectorStore(embeddings)
    docstore = InMemoryStore()

    parent_ids_by_source = {}
    parent_entries = []
    for parent in parent_docs:
        source = parent.metadata.get("source")
        if not source:
            raise ValueError("Every parent document must have a source filename.")
        if source in parent_ids_by_source:
            raise ValueError(f"Duplicate parent source filename: {source}")

        parent_id = str(uuid.uuid4())
        parent_ids_by_source[source] = parent_id
        parent_entries.append((parent_id, parent))

    child_docs = []
    for segment in semantic_docs:
        source = segment.metadata.get("source")
        if source not in parent_ids_by_source:
            raise ValueError(
                f"No full-file parent document found for child segment source: {source}"
            )
        child_docs.append(
            Document(
                page_content=segment.page_content,
                metadata={**segment.metadata, "doc_id": parent_ids_by_source[source]},
            )
        )

    if not child_docs:
        raise ValueError("No semantic child segments were provided for retrieval.")

    docstore.mset(parent_entries)
    vector_store.add_documents(child_docs)
    retriever = MultiVectorRetriever(
        vectorstore=vector_store,
        docstore=docstore,
        id_key="doc_id",
        search_kwargs={"k": k},
    )
    return retriever.invoke(query)


def execute_drill_log_scan(
    window_payload: dict[str, Any], model_name: str = "gpt-4o-mini"
) -> dict[str, Any]:
    """Extract structured geological attributes from one text window."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert production geologist mining parser. Analyse the following raw drill log window segment. "
                "Extract the core structural geological attributes cleanly into JSON format according to schema instructions. "
                "If metadata fields or classifications are missing from this specific window block, label as 'Context Missing'.",
            ),
            ("human", "Log Segment Source: {source}\n\nContent:\n{content}"),
        ]
    )

    llm = ChatOpenAI(model=model_name, temperature=0.0)
    chain = prompt | llm.with_structured_output(LithologyExtraction)

    try:
        result = chain.invoke(
            {
                "source": window_payload["source"],
                "content": window_payload["content"],
            }
        )
        return result.model_dump()
    except Exception as e:
        return {"error": f"Failed extraction window: {e!s}"}


def execute_hole_data_scan(
    window_payload: dict[str, Any], model_name: str = "gpt-4o-mini"
) -> dict[str, Any]:
    """Extract structured geological attributes from one text window."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "What is the total depth of the hole, and what rock type is found at the End of Hole (EOH)?",
            ),
            ("human", "Log Segment Source: {source}\n\nContent:\n{content}"),
        ]
    )

    llm = ChatOpenAI(model=model_name, temperature=0.0)
    chain = prompt | llm.with_structured_output(LogScanRecord)

    try:
        print(f"window_payload: {window_payload}\n")
        result = chain.invoke(
            {"source": window_payload["source"], "content": window_payload["content"]}
        )
        return result.model_dump()
    except Exception as e:
        return {"error": f"Failed extraction window: {e!s}"}


def execute_search_mineral_conversational_output(
    window_payload: dict[str, Any], model_name: str = "gpt-4o-mini"
) -> dict[str, Any]:
    """Extract structured geological attributes from one text window."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "What is the main rock type found in this drill log?"),
            ("human", "Log Segment Source: {source}\n\nContent:\n{content}"),
        ]
    )

    llm = ChatOpenAI(model=model_name, temperature=0.0)
    chain = prompt | llm | StrOutputParser()

    try:
        # print(f"window_payload: {window_payload}\n")
        result = chain.invoke(
            {"source": window_payload["source"], "content": window_payload["content"]}
        )
        return result
    except Exception as e:
        return {"error": f"Failed extraction window: {e!s}"}


def execute_search_mineral_structured_output(
    window_payload: dict[str, Any], model_name: str = "gpt-4o-mini"
) -> dict[str, Any]:
    """Extract structured geological attributes from one text window."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract EVERY depth interval from the provided drill log document into the structured schema. No conversational output.",
            ),
            ("human", "Log Segment Source: {source}\n\nContent:\n{content}"),
        ]
    )

    llm = ChatOpenAI(model=model_name, temperature=0.0, max_completion_tokens=4000)

    try:
        chain = prompt | llm.with_structured_output(DrillHoleLog, strict=False)

        # print(f"window_payload: {window_payload}\n")
        result = chain.invoke(
            {"source": window_payload["source"], "content": window_payload["content"]}
        )
        print(f"***raw output: {result}\n")
        return result
    except Exception as e:
        print(f"error: {e!s} \n")
        return ""


def execute_mineral_summary(
    source: str, content: str, model_name: str = "gpt-4o-mini"
) -> dict[str, Any]:
    """Extract every mineral occurrence (assay or visual) from one complete drill log file.

    Returns {"hole_id": ..., "source": ..., "minerals": [one dict per occurrence]}.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an expert exploration geologist. Read the COMPLETE drill log and list EVERY mineral "
                    "or element occurrence into the structured schema, one row per sample or interval. "
                    "Include laboratory assay results (e.g., Au, As) with basis 'assay', and minerals noted in the "
                    "geological descriptions (e.g., pyrite, chalcopyrite) with basis 'visual'. "
                    "Do not include rock types or alteration minerals such as chlorite, biotite or sericite. "
                    "Only use values written in the log. If the log gives a range (e.g., '3-5%'), use the midpoint "
                    "as the amount. If no minerals are recorded, return an empty list."
                ),
            ),
            ("human", "Log Source: {source}\n\nContent:\n{content}"),
        ]
    )

    llm = ChatOpenAI(model=model_name, temperature=0.0)
    chain = prompt | llm.with_structured_output(MineralSummary)

    try:
        result = chain.invoke({"source": source, "content": content})
        rows = [occurrence.model_dump() for occurrence in result.occurrences]
        for row in rows:
            row["hole_id"] = result.hole_id
            # a "trace" mention has no measured amount, so don't let the LLM report it as 0
            if row["unit"] == "trace":
                row["amount"] = None
        return {"hole_id": result.hole_id, "source": source, "minerals": rows}
    except Exception as e:
        print(f"error: Failed mineral summary for {source}: {e!s} \n")
        return {"hole_id": "Unknown", "source": source, "minerals": []}
