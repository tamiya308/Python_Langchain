import uuid
from typing import Any, Dict, List

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


COLLECTION_NAME = "optimized_windows"
MINERAL_COLLECTION_NAME = "mineral_summary"
VECTOR_SIZE = 384
EMBEDDING_MODEL = "text-embedding-3-small"


def initialize_qdrant_collection(collection_name: str = COLLECTION_NAME) -> QdrantClient:
    qdrant_client = QdrantClient(host="localhost", port=6333)

    if not qdrant_client.collection_exists(collection_name=collection_name):
        print(f"Collection '{collection_name}' not found. Creating a new one...")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
        print(f"Collection '{collection_name}' created successfully.")

    return qdrant_client


def save_mineral_summaries_to_qdrant(
    hole_summaries: List[Dict[str, Any]], qdrant_client: QdrantClient
) -> None:
    """
    Saves one point per drill hole into the mineral summary collection.
    The point id is based on the source filename, so re-running the import
    replaces a hole's summary instead of adding a duplicate.
    """
    if not hole_summaries:
        print("No mineral summaries to upload.")
        return

    # A short text version of each summary, so the summaries can also be searched by meaning
    texts = []
    for summary in hole_summaries:
        minerals = sorted({row["mineral"] for row in summary["minerals"]})
        texts.append(
            f"Drill hole {summary['hole_id']} minerals: {', '.join(minerals) or 'none recorded'}"
        )

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, dimensions=VECTOR_SIZE)
    vectors = embeddings.embed_documents(texts)

    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, summary["source"])),
            vector=vector,
            payload=summary,
        )
        for summary, vector in zip(hole_summaries, vectors)
    ]

    print(f"Uploading {len(points)} mineral summaries to Qdrant...")
    qdrant_client.upsert(
        collection_name=MINERAL_COLLECTION_NAME, wait=True, points=points
    )


def get_mineral_summaries(host="localhost", port=6333) -> List[Dict[str, Any]]:
    """Returns every drill hole's mineral summary (one record per hole)."""
    client = QdrantClient(host=host, port=port)
    if not client.collection_exists(collection_name=MINERAL_COLLECTION_NAME):
        return []

    summaries = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=MINERAL_COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        summaries.extend(record.payload for record in records)
        if offset is None:
            break

    return summaries


def save_windows_to_qdrant(
    optimized_windows: List[Dict[str, Any]], qdrant_client: QdrantClient
) -> None:
    """
    Embeds each window and uploads it to the local Qdrant instance.
    """
    if not optimized_windows:
        print("No windows to upload.")
        return

    contents = [window.get("content", "") for window in optimized_windows]
    if any(not content.strip() for content in contents):
        raise ValueError("Every window must contain non-empty text in 'content'.")

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=VECTOR_SIZE,
    )
    vectors = embeddings.embed_documents(contents)

    if len(vectors) != len(optimized_windows):
        raise RuntimeError(
            f"Expected {len(optimized_windows)} embeddings, received {len(vectors)}."
        )

    points = []
    for idx, (window, vector) in enumerate(zip(optimized_windows, vectors)):
        payload = {
            "window_id": window.get("id", idx),
            "start_time": window.get("start_time"),
            "end_time": window.get("end_time"),
            "optimization_score": window.get("score"),
            "source": window.get("source", "Unknown"),
            "content": window["content"],
            "token_count": window.get("token_count"),
            "metadata": window.get("metadata", {}),
        }

        points.append(
            PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
        )

    print(f"Uploading {len(points)} windows to Qdrant...")
    operation_info = qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        wait=True,
        points=points,
    )
    print("Upload complete!", operation_info)


def get_all_text_from_qdrant(collection_name="optimized_windows", host="localhost", port=6333):
    """
    Connects to the local Qdrant instance, triggers an internal collection 
    synchronization, and extracts all text metadata records safely.
    """
    # 1. Connect to your running Docker container
    client = QdrantClient(host=host, port=port)
    
    # 💡 FORCE ENGINE REFRESH: This triggers a backend state update across your shards
    print(f"Synchronizing engine state for collection: '{collection_name}'...")
    try:
        # Requesting collection info forces the database to evaluate unindexed points in the WAL
        client.get_collection(collection_name=collection_name)
    except Exception as e:
        print(f"Warning during cluster synchronization: {e}")

    all_payloads = []
    offset = None
    
    print(f"--- Fetching all data from collection: '{collection_name}' ---")
    
    # 2. Pull points page by page
    while True:
        records, offset = client.scroll(
            collection_name=collection_name,
            limit=100,           
            offset=offset,       
            with_payload=True,   
            with_vectors=False   
        )
        
        for point in records:
            if point.payload:
                record_data = {"qdrant_id": point.id, **point.payload}
            else:
                record_data = {"qdrant_id": point.id, "_status": "Empty payload dictionary"}
                
            all_payloads.append(record_data)
        
        if offset is None:
            break
            
    print(f"Successfully retrieved {len(all_payloads)} records.")
    return all_payloads 