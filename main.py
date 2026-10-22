import threading
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent import run_agent
from drill_import import run_import

load_dotenv()

app = FastAPI(title="Drill Log RAG")

# the import wipes and rebuilds the Qdrant collections, so only one may run at a time
import_lock = threading.Lock()


class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="A question about the drill logs.")


class ToolStep(BaseModel):
    step: int
    tool: str
    args: dict[str, Any]


class AskResponse(BaseModel):
    answer: str
    steps: list[ToolStep]


class ImportRequest(BaseModel):
    log_files_directory: str = "./Drill Logs"
    images_directory: str = "./Drill Images"


class ImportedLog(BaseModel):
    source: str
    mineral_occurrences: int
    semantic_segments: int
    windows: int


class ImportResponse(BaseModel):
    logs: list[ImportedLog]
    images: list[str]
    skipped_images: list[str]


# Plain `def` (not `async def`) routes: the work is blocking, so FastAPI runs them in a thread pool.


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> dict:
    """Agentic RAG: answers a question about the drill logs using the Qdrant collections."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="The question cannot be empty.")
    return run_agent(question)


@app.post("/import", response_model=ImportResponse)
def import_data(request: ImportRequest | None = None) -> dict:
    """Rebuilds the Qdrant collections from the drill logs and images (existing data is deleted)."""
    request = request or ImportRequest()
    if not import_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="An import is already running.")
    try:
        return run_import(request.log_files_directory, request.images_directory)
    finally:
        import_lock.release()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
