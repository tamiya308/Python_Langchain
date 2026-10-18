import uuid
from typing import Any

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

OPTIMIZED_WINDOWS_COLLECTION_NAME = "optimized_windows"
MINERAL_COLLECTION_NAME = "mineral_summary"
VECTOR_SIZE = 384
EMBEDDING_MODEL = "text-embedding-3-small"

# named vectors inside the optimized_windows collection
TEXT_VECTOR_NAME = "text"  # drill log text windows
IMAGE_VECTOR_NAME = "image"  # drill hole strip-log images (embedded from their description)


def initialize_qdrant_collection(
    collection_name: str, vector_names: list[str] | None = None
) -> QdrantClient:
    """
    Gives a fresh, empty collection. If the collection already exists it is deleted
    first, which removes all of its points (and the segments/shards that store them),
    so a re-import never mixes old and new data.

    Pass vector_names to create one named vector per name (e.g. "text" and "image");
    otherwise the collection has a single unnamed vector.
    """
    qdrant_client = QdrantClient(host="localhost", port=6333)

    if qdrant_client.collection_exists(collection_name=collection_name):
        print(f"Collection '{collection_name}' exists. Deleting it and all its points...")
        qdrant_client.delete_collection(collection_name=collection_name)

    # recreating also picks up any change to VECTOR_SIZE
    print(f"Creating collection '{collection_name}'...")
    vector_params = VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=(
            {name: vector_params for name in vector_names}
            if vector_names
            else vector_params
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
            "type": "text",
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
            PointStruct(
                id=str(uuid.uuid4()),
                vector={TEXT_VECTOR_NAME: vector},
                payload=payload,
            )
        )

    print(f"Uploading {len(points)} windows to Qdrant...")
    operation_info = qdrant_client.upsert(
        collection_name=OPTIMIZED_WINDOWS_COLLECTION_NAME,
        wait=True,
        points=points,
    )
    print("Upload complete!", operation_info)


def save_drill_image_to_qdrant(
    source: str, image_path: str, description: str, qdrant_client: QdrantClient
) -> None:
    """
    Saves one drill hole image as a point in the optimized_windows collection, using only
    the "image" named vector (embedded from the image's text description). The point id is
    based on the filename, so re-running the import replaces the image instead of duplicating it.
    """
    if not description.strip():
        print(f"No description for {source}, skipping.")
        return

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, dimensions=VECTOR_SIZE)
    vector = embeddings.embed_query(description)

    point = PointStruct(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, source)),
        vector={IMAGE_VECTOR_NAME: vector},
        payload={
            "type": "image",
            "source": source,
            "image_path": image_path,
            "content": description,
        },
    )

    print(f"Uploading drill image {source} to Qdrant...")
    qdrant_client.upsert(
        collection_name=OPTIMIZED_WINDOWS_COLLECTION_NAME, wait=True, points=[point]
    )
