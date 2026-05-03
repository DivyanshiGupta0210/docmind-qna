import os
import uuid
import json
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from qa_engine import DocumentQAEngine

app = FastAPI(title="Document Q&A System")
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

qa_engine = DocumentQAEngine()

sessions: dict[str, dict] = {}


class QuestionRequest(BaseModel):
    session_id: str
    question: str


@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("static/index.html")


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    session_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{session_id}.pdf"

    content = await file.read()
    file_path.write_bytes(content)

    try:
        doc_data = qa_engine.process_document(str(file_path))
        sessions[session_id] = {
            "filename": file.filename,
            "doc_data": doc_data,
        }
        return {
            "session_id": session_id,
            "filename": file.filename,
            "page_count": doc_data["page_count"],
            "chunk_count": len(doc_data["chunks"]),
        }
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")


@app.post("/ask")
async def ask_question(req: QuestionRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a document first.")

    session = sessions[req.session_id]
    try:
        result = qa_engine.answer_question(
            question=req.question,
            doc_data=session["doc_data"],
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {str(e)}")


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        file_path = UPLOAD_DIR / f"{session_id}.pdf"
        file_path.unlink(missing_ok=True)
    return {"status": "deleted"}
