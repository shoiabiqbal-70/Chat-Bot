"""Question answering API."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.services.answer_service import AnswerError, AnswerResult, RateLimitError, answer_question
from app.services.vector_store import VectorStore, VectorStoreError

router = APIRouter(tags=["Questions"])


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_id: UUID | None = None


@router.post("/ask", response_model=AnswerResult)
async def ask(request: QuestionRequest,
              session: Annotated[AsyncSession, Depends(get_db)]) -> AnswerResult:
    """Return a grounded answer and the inspected retrieval evidence."""
    try:
        return await answer_question(request.question, session, VectorStore(),
                                     document_id=request.document_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except (AnswerError, VectorStoreError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
