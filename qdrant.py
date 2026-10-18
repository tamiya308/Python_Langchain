import uuid
from typing import Any

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

OPTIMIZED_WINDOWS_COLLECTION_NAME = "optimized_windows"
MINERAL_COLLECTION_NAME = "mineral_summary"
VECTOR_SIZE = 384
EMBEDDING_MODEL = "text-embedding-3-small"


def initialize_qdrant_collection(collection_name: str) -> QdrantClient:
    """
    Gives a fresh, empty collection. If the collection already exists it is deleted
    first, which removes all of its points (and the segments/shards that store them),
    so a re-import never mixes old and new data.
    """
    qdrant_client = QdrantClient(host="localhost", port=6333)

    if qdrant_client.collection_exists(collection_name=collection_name):
        print(f"Collection '{collection_name}' exists. Deleting it and all its points...")
        qdrant_client.delete_collection(collection_name=collection_name)

    # recreating also picks up any change to VECTOR_SIZE
    print(f"Creating collection '{collection_name}'...")
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )
    print(f"Collection '{collection_name}' created successfully.")

    return qdrant_client


def save_mineral_summary_to_qdrant(
    hole_summary: dict[str, Any], qdrant_client: QdrantClient
) -> None:
    """
    Saves one drill hole's summary as a point in the mineral summary collection.
    The point id is based on the source filename, so re-running the import
    replaces a hole's summary instead of adding a duplicate.
    """
    # A short text version of the summary, so the summaries can also be searched by meaning
    minerals = sorted({row["mineral"] for row in hole_summary["minerals"]})
    text = f"Drill hole {hole_summary['hole_id']} minerals: {', '.join(minerals) or 'none recorded'}"

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, dimensions=VECTOR_SIZE)
    vector = embeddings.embed_query(text)

    point = PointStruct(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, hole_summary["source"])),
        vector=vector,
        payload=hole_summary,
    )

    print(f"Uploading mineral summary for {hole_summary['source']} to Qdrant...")
    qdrant_client.upsert(
        collection_name=MINERAL_COLLECTION_NAME, wait=True, points=[point]
    )


def get_mineral_summaries(host="localhost", port=6333) -> list[dict[str, Any]]:
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
    optimized_windows: list[dict[str, Any]], qdrant_client: QdrantClient
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

        points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))

    print(f"Uploading {len(points)} windows to Qdrant...")
    operation_info = qdrant_client.upsert(
        collection_name=OPTIMIZED_WINDOWS_COLLECTION_NAME,
        wait=True,
        points=points,
    )
    print("Upload complete!", operation_info)


def get_all_text_from_qdrant(
    collection_name="optimized_windows", host="localhost", port=6333
):
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
            with_vectors=False,
        )

        for point in records:
            if point.payload:
                record_data = {"qdrant_id": point.id, **point.payload}
            else:
                record_data = {
                    "qdrant_id": point.id,
                    "_status": "Empty payload dictionary",
                }

            all_payloads.append(record_data)

        if offset is None:
            break

    print(f"Successfully retrieved {len(all_payloads)} records.")
    return all_payloads
