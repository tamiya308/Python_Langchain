import glob
import os
from typing import Any

import tiktoken
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

from ChainInvokeFunctions import (
    execute_drill_image_description,
    execute_mineral_summary,
)
from qdrant import (
    IMAGE_VECTOR_NAME,
    MINERAL_COLLECTION_NAME,
    OPTIMIZED_WINDOWS_COLLECTION_NAME,
    TEXT_VECTOR_NAME,
    initialize_qdrant_collection,
    save_drill_image_to_qdrant,
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

    def close_window() -> None:
        """Join the active window's chunks into one window record."""
        window_text = "\n\n--- [Semantic Context Shift] ---\n\n".join(
            [d.page_content for d in current_window_docs]
        )
        windows.append(
            {
                "source": current_window_docs[0].metadata.get("source", "Unknown"),
                "content": window_text,
                "token_count": current_window_tokens,
            }
        )

    for doc in semantic_docs:
        doc_tokens = count_tokens(doc.page_content)

        # Close the active window when this chunk won't fit. A chunk longer than
        # max_window_tokens is never split: it closes the active window first (so windows
        # stay in log order) and then gets a window of its own, with the usual overlap.
        if (
            current_window_docs
            and current_window_tokens + doc_tokens > max_window_tokens
        ):
            close_window()

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
        close_window()

        print(f"***windows: {windows}")

    return windows


def run_import(
    log_files_directory: str = "./Drill Logs", images_directory: str = "./Drill Images"
) -> dict[str, Any]:
    """
    Rebuilds both Qdrant collections from the drill logs and drill images.
    WARNING: the collections are deleted and recreated first, so any existing data is lost.
    """
    target_files = glob.glob(os.path.join(log_files_directory, "drill_hole_*.txt"))

    print(f"📦 Discovered {len(target_files)} target log files for analysis.\n")

    imported_logs = []
    imported_images = []
    skipped_images = []

    mineralSummary_client = initialize_qdrant_collection(MINERAL_COLLECTION_NAME)
    # text windows and drill images share this collection, each under its own named vector
    qdrant_client = initialize_qdrant_collection(
        OPTIMIZED_WINDOWS_COLLECTION_NAME,
        vector_names=[TEXT_VECTOR_NAME, IMAGE_VECTOR_NAME],
    )

    for file_path in target_files:
        print(f"📖 Scanning: {os.path.basename(file_path)}")

        # Keep the complete file as the parent; use its semantic segments downstream.
        parent_document = load_drill_log_document(file_path)

        # Step A: Mineral summary - scan the WHOLE file so no samples are missed
        hole_summary = execute_mineral_summary(
            parent_document.metadata["source"], parent_document.page_content
        )

        # save one summary per drill hole to Qdrant, so the agent (agent.py) can compare every hole
        save_mineral_summary_to_qdrant(hole_summary, mineralSummary_client)

        semantic_segments = semantic_chunk_document(parent_document)

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

        imported_logs.append(
            {
                "source": parent_document.metadata["source"],
                "mineral_occurrences": len(hole_summary["minerals"]),
                "semantic_segments": len(semantic_segments),
                "windows": len(optimized_windows),
            }
        )

    # Step C: Drill images - describe each strip log with the vision model, then embed the description
    image_files = glob.glob(os.path.join(images_directory, "drill_hole_*.jpg"))
    print(f"\n🖼️ Discovered {len(image_files)} drill images.\n")

    # only import images whose drill hole has a log in log_files_directory
    # (e.g. drill_hole_ddh_26_01.jpg needs drill_hole_ddh_26_01.txt)
    logged_holes = {os.path.splitext(os.path.basename(f))[0] for f in target_files}

    for image_path in image_files:
        source = os.path.basename(image_path)
        if os.path.splitext(source)[0] not in logged_holes:
            print(f"⏭️ Skipping {source}: no matching drill log.")
            skipped_images.append(source)
            continue

        print(f"🖼️ Describing: {source}")
        description = execute_drill_image_description(source, image_path)
        save_drill_image_to_qdrant(source, image_path, description, qdrant_client)
        imported_images.append(source)

    print("\n✅ Import finished!")

    return {
        "logs": imported_logs,
        "images": imported_images,
        "skipped_images": skipped_images,
    }
