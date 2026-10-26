import sys
import os
from pathlib import Path
import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass

# Add backend to path so imports work
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from vector_store import SearchResults


@pytest.fixture
def mock_vector_store():
    """Create a mocked VectorStore with search capabilities."""
    store = MagicMock()
    store.search = MagicMock()
    store.get_lesson_link = MagicMock(return_value=None)
    store.get_course_link = MagicMock(return_value="https://example.com/course")
    return store


def sample_search_results(docs, metas):
    """Factory for creating SearchResults."""
    return SearchResults(
        documents=docs,
        metadata=metas,
        distances=[0.1] * len(docs)
    )


@pytest.fixture
def mock_anthropic_client():
    """Create a mocked Anthropic client."""
    client = MagicMock()
    return client


def make_text_block(text):
    """Create a mock TextBlock."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def make_tool_use_block(tool_name, tool_id, input_dict):
    """Create a mock ToolUseBlock."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.id = tool_id
    block.input = input_dict
    return block


def make_text_response(text):
    """Create a mock response with text content."""
    response = MagicMock()
    response.content = [make_text_block(text)]
    response.stop_reason = "end_turn"
    response.stop_details = None
    return response


def make_tool_use_response(tool_name, tool_id, input_dict):
    """Create a mock response with tool use."""
    response = MagicMock()
    response.content = [make_tool_use_block(tool_name, tool_id, input_dict)]
    response.stop_reason = "tool_use"
    response.stop_details = None
    return response


@pytest.fixture
def sample_search_results_factory():
    """Factory fixture for creating SearchResults."""
    return sample_search_results


@pytest.fixture
def make_text_response_fixture():
    """Fixture to create text responses."""
    return make_text_response


@pytest.fixture
def make_tool_use_response_fixture():
    """Fixture to create tool use responses."""
    return make_tool_use_response


@pytest.fixture
def make_text_block_fixture():
    """Fixture to create text blocks."""
    return make_text_block


@pytest.fixture
def mock_rag_system():
    """Mocked RAGSystem for API endpoint tests."""
    rag = MagicMock()
    rag.query.return_value = ("Test answer", [])
    rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Course A", "Course B"],
    }
    rag.session_manager.create_session.return_value = "session_1"
    return rag


@pytest.fixture
def test_client(mock_rag_system):
    """FastAPI TestClient with mocked RAGSystem; no static files mounted."""
    from fastapi import FastAPI, HTTPException
    from fastapi.testclient import TestClient
    from pydantic import BaseModel
    from typing import List, Optional

    test_app = FastAPI()

    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class Source(BaseModel):
        text: str
        link: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Source]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    @test_app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()
            answer, sources = mock_rag_system.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @test_app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @test_app.get("/")
    async def root():
        return {"status": "ok"}

    return TestClient(test_app)
