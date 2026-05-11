from fastapi import APIRouter, UploadFile, File
import shutil
import os

from app.services.parser import load_pdf
from app.services.embeddings import split_documents
from app.services.vectorstore import vectorstore

router = APIRouter()

UPLOAD_DIR = "data"

os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    documents = load_pdf(file_path)

    chunks = split_documents(documents)

    for chunk in chunks:
        chunk.metadata["source"] = file.filename

    vectorstore.add_documents(chunks)

    return {
        "message": "Document uploaded successfully",
        "chunks_indexed": len(chunks)
    }