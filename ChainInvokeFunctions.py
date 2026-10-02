from unittest import result
import uuid
from typing import Any, List, Optional  # Added Optional import here

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_classic.retrievers import MultiVectorRetriever
from langchain_classic.storage import InMemoryStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field


class LithologyExtraction(BaseModel):
    hole_id: str = Field(description="The unique identifier or borehole code extracted from log header.")
    rock_type: str = Field(description="Dominant rock type or soil composition identified.")
    texture: str = Field(description="Rock matrix texture profile (e.g., foliated, aphanitic, clastic).")
    grain_size: str = Field(description="Grain classification range observed.")
    key_anomalies: List[str] = Field(description="Notable geological structural observations, mineralization, or fractures.")

# Rebuild schema to resolve any internal type hints before LangChain schema conversion
LithologyExtraction.model_rebuild()


class LogScanRecord(BaseModel):
    drill_hole_id: Optional[str] = Field(default="Unknown", description="The ID of the drill hole (e.g., DDH-26-01)")
    main_rock_type: str = Field(description="The primary lithology / rock type identified")
    # raw_content: str = Field(description="The original unstructured message content")
    total_tokens_used: int = Field(default=0, description="Tokens consumed for this specific inference window")

# Rebuild schema to safely handle typing references and fix the PydanticUserError crash
LogScanRecord.model_rebuild()


def retrieve_parent_documents(
    parent_docs: List[Document],
    semantic_docs: List[Document],
    query: str,
    k: int = 4
) -> List[Document]:
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
                metadata={
                    **segment.metadata,
                    "doc_id": parent_ids_by_source[source]
                }
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
        search_kwargs={"k": k}
    )
    return retriever.invoke(query)


def execute_drill_log_scan(
    window_payload: dict[str, Any],
    model_name: str = "gpt-4o-mini"
) -> dict[str, Any]:
    """Extract structured geological attributes from one text window."""
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert production geologist mining parser. Analyse the following raw drill log window segment. "
            "Extract the core structural geological attributes cleanly into JSON format according to schema instructions. "
            "If metadata fields or classifications are missing from this specific window block, label as 'Context Missing'."
        ),
        ("human", "Log Segment Source: {source}\n\nContent:\n{content}")
    ])

    llm = ChatOpenAI(model=model_name, temperature=0.0)
    chain = prompt | llm.with_structured_output(LithologyExtraction)

    try:
        result = chain.invoke({
            "source": window_payload["source"],
            "content": window_payload["content"],
        })
        return result.model_dump()
    except Exception as e:
        return {"error": f"Failed extraction window: {str(e)}"}


def execute_mineral_scan(
    window_payload: dict[str, Any],
    model_name: str = "gpt-4o-mini"
) -> dict[str, Any]:
    """Extract structured geological attributes from one text window."""
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "What is the main rock type found in this drill log?"
        ),
        ("human", "Log Segment Source: {source}\n\nContent:\n{content}")
    ])

    llm = ChatOpenAI(model=model_name, temperature=0.0)
    chain = prompt | llm.with_structured_output(LogScanRecord)

    try:
        result = chain.invoke({
            "source": window_payload["source"],
            "content": window_payload["content"]
        })
        return result.model_dump()
    except Exception as e:
        return {"error": f"Failed extraction window: {str(e)}"}