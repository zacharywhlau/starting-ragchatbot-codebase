import pytest
from unittest.mock import MagicMock
from search_tools import CourseSearchTool
from vector_store import SearchResults


def sample_search_results(docs, metas):
    """Factory for creating SearchResults."""
    return SearchResults(
        documents=docs,
        metadata=metas,
        distances=[0.1] * len(docs)
    )


class TestCourseSearchTool:
    """Tests for CourseSearchTool.execute()"""

    def test_execute_with_results_returns_formatted_string(self):
        """Execute with search results returns formatted string with headers."""
        # Arrange
        mock_vector_store = MagicMock()
        docs = ["Introduction to AI", "Core concepts"]
        metas = [
            {"course_title": "AI Basics", "lesson_number": 1},
            {"course_title": "AI Basics", "lesson_number": 1}
        ]
        mock_vector_store.search.return_value = sample_search_results(docs, metas)
        mock_vector_store.get_lesson_link.return_value = None
        mock_vector_store.get_course_link.return_value = "https://example.com"

        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(query="What is AI?")

        # Assert
        assert isinstance(result, str)
        assert "[AI Basics - Lesson 1]" in result
        assert "Introduction to AI" in result

    def test_execute_empty_results_returns_no_content_message(self):
        """Execute with empty results returns clear message."""
        mock_vector_store = MagicMock()
        mock_vector_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[], error=None
        )

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="nonexistent topic")

        assert "No relevant content found" in result

    def test_execute_with_course_filter_no_results_includes_filter_in_message(self):
        """No results message includes course name when filtered."""
        mock_vector_store = MagicMock()
        mock_vector_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[], error=None
        )

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="topic", course_name="Python 101")

        assert "No relevant content found in course 'Python 101'" in result

    def test_execute_search_error_returns_error_string(self):
        """Execute with error from search returns error message."""
        mock_vector_store = MagicMock()
        mock_vector_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[],
            error="Course not found"
        )

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="topic")

        assert result == "Course not found"

    def test_execute_passes_course_name_to_search(self):
        """Execute passes course_name parameter to store.search."""
        mock_vector_store = MagicMock()
        mock_vector_store.search.return_value = sample_search_results([], [])

        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="test", course_name="Python 101")

        mock_vector_store.search.assert_called_once()
        call_kwargs = mock_vector_store.search.call_args[1]
        assert call_kwargs.get("course_name") == "Python 101"

    def test_execute_passes_lesson_number_to_search(self):
        """Execute passes lesson_number parameter to store.search."""
        mock_vector_store = MagicMock()
        mock_vector_store.search.return_value = sample_search_results([], [])

        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="test", lesson_number=3)

        mock_vector_store.search.assert_called_once()
        call_kwargs = mock_vector_store.search.call_args[1]
        assert call_kwargs.get("lesson_number") == 3

    def test_execute_populates_last_sources(self):
        """Execute populates last_sources after successful search."""
        mock_vector_store = MagicMock()
        docs = ["Content"]
        metas = [{"course_title": "Course A", "lesson_number": 1}]
        mock_vector_store.search.return_value = sample_search_results(docs, metas)
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"

        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="topic")

        assert len(tool.last_sources) > 0
        assert tool.last_sources[0]["text"] == "Course A - Lesson 1"

    def test_execute_source_link_prefers_lesson_link(self):
        """Execute prefers lesson link over course link."""
        mock_vector_store = MagicMock()
        docs = ["Content"]
        metas = [{"course_title": "Course A", "lesson_number": 1}]
        mock_vector_store.search.return_value = sample_search_results(docs, metas)
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        mock_vector_store.get_course_link.return_value = "https://example.com/course"

        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="topic")

        assert tool.last_sources[0]["link"] == "https://example.com/lesson1"

    def test_execute_source_link_falls_back_to_course_link(self):
        """Execute uses course link when lesson link unavailable."""
        mock_vector_store = MagicMock()
        docs = ["Content"]
        metas = [{"course_title": "Course A", "lesson_number": None}]
        mock_vector_store.search.return_value = sample_search_results(docs, metas)
        mock_vector_store.get_lesson_link.return_value = None
        mock_vector_store.get_course_link.return_value = "https://example.com/course"

        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="topic")

        assert tool.last_sources[0]["link"] == "https://example.com/course"

    def test_execute_deduplicates_sources(self):
        """Execute deduplicates sources with same text and link."""
        mock_vector_store = MagicMock()
        docs = ["Content 1", "Content 2"]
        metas = [
            {"course_title": "Course A", "lesson_number": 1},
            {"course_title": "Course A", "lesson_number": 1}
        ]
        mock_vector_store.search.return_value = sample_search_results(docs, metas)
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"

        tool = CourseSearchTool(mock_vector_store)
        tool.execute(query="topic")

        # Should have only one source despite two matching documents
        assert len(tool.last_sources) == 1
