import glob
import os
from typing import Any

import tiktoken
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

from ChainInvokeFunctions import execute_mineral_summary
from qdrant import (
    MINERAL_COLLECTION_NAME,
    OPTIMIZED_WINDOWS_COLLECTION_NAME,
    initialize_qdrant_collection,
    save_mineral_summary_to_qdrant,
    save_windows_to_qdrant,
)

load_dotenv()

TOKENIZER = tiktoken.encoding_for_model("gpt-4o-mini")


def count_tokens(text: str) -> int:
    """Helper function to keep precise track of token counts."""
    return len(TOKENIZER.encode(text))


def load_drill_log_document(file_path: str) -> Document:
    """Read a drill log as one parent document."""
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    return Document(
        page_content=raw_text, metadata={"source": os.path.basename(file_path)}
    )


def semantic_chunk_document(base_doc: Document) -> list[Document]:
    """Split one drill log document into semantic child segments."""
    # Initialize the semantic chunker using OpenAI Embeddings
    # Uses a percentile threshold to split chunks dynamically when context shifts
    # this sends it to openAi
    # breaks it down into high dimensional vectors and then looks for semantic shifts in the text
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")

    return chunker.split_documents([base_doc])


def build_dynamic_sliding_windows(
    semantic_docs: list[Document],
    max_window_tokens: int = 1500,
    overlap_tokens: int = 250,
) -> list[dict[str, Any]]:
    """
    Bundles semantic chunks into an optimized sliding window to prevent LLM fatigue
    and ensure geological boundaries aren't aggressively truncated across prompts.
    """
    windows = []
    current_window_docs = []
    current_window_tokens = 0

    for doc in semantic_docs:
        doc_tokens = count_tokens(doc.page_content)

        # If a single semantic chunk is exceptionally long, handle separately
        if doc_tokens > max_window_tokens:
            windows.append(
                {
                    "source": doc.metadata.get("source", "Unknown"),
                    "content": doc.page_content,
                    "token_count": doc_tokens,
                }
            )
            continue

        if current_window_tokens + doc_tokens > max_window_tokens:
            # Consolidate the active window text before slicing forward
            window_text = "\n\n--- [Semantic Context Shift] ---\n\n".join(
                [d.page_content for d in current_window_docs]
            )

            windows.append(
                {
                    # FIXED: Pulls source safely from the first document inside the active window queue
                    "source": (
                        current_window_docs[0].metadata.get("source", "Unknown")
                        if current_window_docs
                        else "Unknown"
                    ),
                    "content": window_text,
                    "token_count": current_window_tokens,
                }
            )

            # Slide window back by applying overlap to retain continuity
            overlap_docs = []
            accumulated_overlap_tokens = 0
            for d in reversed(current_window_docs):
                t = count_tokens(d.page_content)
                if accumulated_overlap_tokens + t <= overlap_tokens:
                    overlap_docs.insert(0, d)
                    accumulated_overlap_tokens += t
                else:
                    break

            current_window_docs = overlap_docs + [doc]
            current_window_tokens = accumulated_overlap_tokens + doc_tokens
        else:
            current_window_docs.append(doc)
            current_window_tokens += doc_tokens

    # Flush remaining documents
    if current_window_docs:
        window_text = "\n\n--- [Semantic Context Shift] ---\n\n".join(
            [d.page_content for d in current_window_docs]
        )
        windows.append(
            {
                # FIXED: Added safe .get() access to prevent crashes on the last chunk
                "source": (
                    semantic_docs[0].metadata.get("source", "Unknown")
                    if semantic_docs
                    else "Unknown"
                ),
                "content": window_text,
                "token_count": current_window_tokens,
            }
        )

        print(f"***windows: {windows}")

    return windows


if __name__ == "__main__":
    # Point this to your directory where files are stored (e.g. current path)
    log_files_directory = "./Drill Logs"
    target_files = glob.glob(os.path.join(log_files_directory, "drill_hole_*.txt"))

    print(f"📦 Discovered {len(target_files)} target log files for analysis.\n")

    all_extracted_insights = []
    all_parent_documents = []
    all_semantic_segments = []

    mineralSummary_client = initialize_qdrant_collection(MINERAL_COLLECTION_NAME)
    qdrant_client = initialize_qdrant_collection(OPTIMIZED_WINDOWS_COLLECTION_NAME)

    for file_path in target_files:
        print(f"📖 Scanning: {os.path.basename(file_path)}")

        # Keep the complete file as the parent; use its semantic segments downstream.
        parent_document = load_drill_log_document(file_path)
        all_parent_documents.append(parent_document)

        # Step A: Mineral summary - scan the WHOLE file so no samples are missed
        hole_summary = execute_mineral_summary(
            parent_document.metadata["source"], parent_document.page_content
        )
        print(f"   ├─ Found {len(hole_summary['minerals'])} mineral occurrences.")

        # save one summary per drill hole to Qdrant, so "main5 - RAG.py" can compare every hole
        save_mineral_summary_to_qdrant(hole_summary, mineralSummary_client)

        semantic_segments = semantic_chunk_document(parent_document)
        all_semantic_segments.extend(semantic_segments)

        # Step B: Dynamic Window grouping (ensures text fits target token boundaries safely)
        optimized_windows = build_dynamic_sliding_windows(
            semantic_segments,
            max_window_tokens=1000,  # Conservative chunk to prevent context fragmentation
            overlap_tokens=200,  # 200 token overlap cushion
        )

        print(f"   ├─ Extracted {len(semantic_segments)} semantic segments.")
        print(f"   ├─ Packed into {len(optimized_windows)} sliding window prompts.")

        # save to qDrant vector database
        save_windows_to_qdrant(optimized_windows, qdrant_client)

    print("\n✅ Execution Finished! Sample Data Extracted:")
