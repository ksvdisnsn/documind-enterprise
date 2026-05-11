from fastapi import APIRouter

from app.schemas import QueryRequest
from app.services.rag_chain import ask_question

router = APIRouter()


@router.post("/query")
async def query_documents(request: QueryRequest):

    result = ask_question(request.question)

    return result