import asyncio
from pathlib import Path
import uuid

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from AgentNetwork.build_agent import run_main_agent, stream_main_agent_events
from fastapi.responses import StreamingResponse
from RAG.ingest_doc import DEFAULT_USER_ID, UPLOADS_DIR, ingest_pdfs


app = FastAPI()
MAX_UPLOAD_FILES = 5
ALLOWED_UPLOAD_SUFFIXES = {".pdf"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/health")
def get_health():
    return {"status": "ok"}




@app.post("/run-agent")
async def run_agent(query: str=Body(...,embed=True)):
    return StreamingResponse(run_main_agent(query), media_type="text/plain")


@app.post("/rag/upload")
async def upload_rag_documents(
    files: list[UploadFile] = File(...),
    user_id: str = Form(DEFAULT_USER_ID),
):
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF is required.")

    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum upload is {MAX_UPLOAD_FILES} PDF files at a time.",
        )

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    original_names: list[str] = []
    try:
        for file in files:
            suffix = Path(file.filename or "").suffix.lower()
            if suffix not in ALLOWED_UPLOAD_SUFFIXES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Only PDF uploads are supported. Invalid file: {file.filename}",
                )

            original_name = Path(file.filename or "document.pdf").name
            safe_name = f"{uuid.uuid4().hex}_{original_name}"
            destination = UPLOADS_DIR / safe_name
            file_bytes = await file.read()
            destination.write_bytes(file_bytes)
            saved_paths.append(str(destination))
            original_names.append(original_name)

        result = await asyncio.to_thread(ingest_pdfs, saved_paths, user_id)
        for file_result, original_name in zip(result.get("files", []), original_names):
            file_result["file_name"] = original_name
        return {
            "status": "completed",
            "message": "All uploaded files were ingested successfully.",
            **result,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        for file in files:
            await file.close()


@app.websocket("/ws/agent")
async def run_agent_ws(websocket: WebSocket):
    await websocket.accept()
    current_stream_task: asyncio.Task | None = None

    async def stop_current_stream(send_status: bool = False):
        nonlocal current_stream_task

        if current_stream_task is None:
            return

        current_stream_task.cancel()
        try:
            await current_stream_task
        except asyncio.CancelledError:
            pass
        finally:
            current_stream_task = None

        if send_status:
            await websocket.send_json({"type": "status", "status": "interrupted"})

    async def stream_query(query: str):
        try:
            await websocket.send_json({"type": "status", "status": "started"})

            async for event in stream_main_agent_events(query):
                await websocket.send_json(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await websocket.send_json({"type": "error", "message": str(exc)})

    try:
        while True:
            payload = await websocket.receive_json()
            action = str(payload.get("action", "query")).strip().lower()

            if action == "interrupt":
                await stop_current_stream(send_status=True)
                continue

            query = str(payload.get("query", "")).strip()

            if not query:
                await websocket.send_json(
                    {"type": "error", "message": "Query is required."}
                )
                continue

            await stop_current_stream(send_status=False)
            current_stream_task = asyncio.create_task(stream_query(query))
    except WebSocketDisconnect:
        if current_stream_task is not None:
            current_stream_task.cancel()
        return
