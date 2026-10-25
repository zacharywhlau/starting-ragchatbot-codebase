import pytest
from unittest.mock import MagicMock, patch, Mock
from rag_system import RAGSystem


class TestRAGSystem:
    """Integration tests for RAGSystem"""

    def create_mock_rag_system(self):
        """Create a RAGSystem with mocked dependencies."""
        with patch("rag_system.DocumentProcessor"), \
             patch("rag_system.VectorStore"), \
             patch("rag_system.AIGenerator"), \
             patch("rag_system.SessionManager"):

            mock_config = MagicMock()
            mock_config.CHUNK_SIZE = 800
            mock_config.CHUNK_OVERLAP = 100
            mock_config.ANTHROPIC_API_KEY = "test-key"
            mock_config.ANTHROPIC_MODEL = "claude-haiku-4-5"
            mock_config.ANTHROPIC_BASE_URL = ""
            mock_config.EMBEDDING_MODEL = "test-model"
            mock_config.MAX_RESULTS = 5
            mock_config.MAX_HISTORY = 2
            mock_config.CHROMA_PATH = "./test_chroma"

            system = RAGSystem(mock_config)
            return system

    def test_query_returns_tuple_of_response_and_sources(self):
        """query() returns a tuple of (response_string, sources_list)."""
        system = self.create_mock_rag_system()
        system.ai_generator.generate_response = MagicMock(return_value="Test response")
        system.tool_manager.get_last_sources = MagicMock(return_value=[])
        system.session_manager.get_conversation_history = MagicMock(return_value=None)

        response, sources = system.query("What is AI?", session_id="test_session")

        assert isinstance(response, str)
        assert isinstance(sources, list)
        assert response == "Test response"

    def test_query_sources_reset_after_retrieval(self):
        """Sources are reset after query() completes."""
        system = self.create_mock_rag_system()
        system.ai_generator.generate_response = MagicMock(return_value="Response")
        system.tool_manager.get_last_sources = MagicMock(return_value=[{"text": "Source 1"}])
        system.tool_manager.reset_sources = MagicMock()
        system.session_manager.get_conversation_history = MagicMock(return_value=None)

        response, sources = system.query("Query", session_id="session_1")

        system.tool_manager.reset_sources.assert_called_once()

    def test_query_updates_session_history(self):
        """query() calls session_manager.add_exchange()."""
        system = self.create_mock_rag_system()
        system.ai_generator.generate_response = MagicMock(return_value="Response")
        system.tool_manager.get_last_sources = MagicMock(return_value=[])
        system.session_manager.get_conversation_history = MagicMock(return_value=None)
        system.session_manager.add_exchange = MagicMock()

        system.query("What is AI?", session_id="session_123")

        system.session_manager.add_exchange.assert_called_once()
        call_args = system.session_manager.add_exchange.call_args[0]
        assert call_args[0] == "session_123"
        assert "What is AI?" in call_args[1]

    def test_search_tool_registered_in_tool_manager(self):
        """CourseSearchTool is registered in tool_manager."""
        system = self.create_mock_rag_system()

        # search_tool should have been registered during __init__
        assert "search_course_content" in system.tool_manager.tools

    def test_outline_tool_registered_in_tool_manager(self):
        """CourseOutlineTool is registered in tool_manager."""
        system = self.create_mock_rag_system()

        # outline_tool should have been registered during __init__
        assert "get_course_outline" in system.tool_manager.tools

    def test_query_propagates_ai_generator_exception(self):
        """Exceptions from generate_response() propagate up."""
        system = self.create_mock_rag_system()
        system.ai_generator.generate_response = MagicMock(side_effect=ValueError("API Error"))
        system.session_manager.get_conversation_history = MagicMock(return_value=None)

        with pytest.raises(ValueError, match="API Error"):
            system.query("test", session_id="session_1")
