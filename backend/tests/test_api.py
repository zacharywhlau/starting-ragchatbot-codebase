import pytest


class TestQueryEndpoint:
    """Tests for POST /api/query"""

    def test_returns_200_with_valid_query(self, test_client):
        response = test_client.post("/api/query", json={"query": "What is Python?"})
        assert response.status_code == 200

    def test_response_has_required_fields(self, test_client):
        response = test_client.post("/api/query", json={"query": "What is Python?"})
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data

    def test_auto_creates_session_when_not_provided(self, test_client, mock_rag_system):
        mock_rag_system.session_manager.create_session.return_value = "auto_session"
        response = test_client.post("/api/query", json={"query": "test"})
        data = response.json()
        assert data["session_id"] == "auto_session"
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_uses_provided_session_id(self, test_client, mock_rag_system):
        response = test_client.post(
            "/api/query", json={"query": "test", "session_id": "my_session"}
        )
        assert response.json()["session_id"] == "my_session"

    def test_provided_session_id_skips_create(self, test_client, mock_rag_system):
        test_client.post("/api/query", json={"query": "test", "session_id": "my_session"})
        mock_rag_system.session_manager.create_session.assert_not_called()

    def test_answer_matches_rag_response(self, test_client, mock_rag_system):
        mock_rag_system.query.return_value = ("Python is a language", [])
        response = test_client.post("/api/query", json={"query": "What is Python?"})
        assert response.json()["answer"] == "Python is a language"

    def test_sources_returned_from_rag(self, test_client, mock_rag_system):
        mock_rag_system.query.return_value = (
            "Answer",
            [{"text": "Python docs - Lesson 1", "link": "https://python.org"}],
        )
        response = test_client.post("/api/query", json={"query": "Python?"})
        sources = response.json()["sources"]
        assert len(sources) == 1
        assert sources[0]["text"] == "Python docs - Lesson 1"
        assert sources[0]["link"] == "https://python.org"

    def test_source_without_link_returns_null_link(self, test_client, mock_rag_system):
        mock_rag_system.query.return_value = ("Answer", [{"text": "Some source"}])
        response = test_client.post("/api/query", json={"query": "test"})
        assert response.json()["sources"][0]["link"] is None

    def test_rag_called_with_query_and_session_id(self, test_client, mock_rag_system):
        mock_rag_system.session_manager.create_session.return_value = "s1"
        test_client.post("/api/query", json={"query": "What is AI?"})
        mock_rag_system.query.assert_called_once_with("What is AI?", "s1")

    def test_returns_500_when_rag_raises(self, test_client, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("service unavailable")
        response = test_client.post("/api/query", json={"query": "test"})
        assert response.status_code == 500

    def test_500_detail_contains_error_message(self, test_client, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("service unavailable")
        response = test_client.post("/api/query", json={"query": "test"})
        assert "service unavailable" in response.json()["detail"]

    def test_missing_query_returns_422(self, test_client):
        response = test_client.post("/api/query", json={})
        assert response.status_code == 422

    def test_empty_sources_list_is_valid(self, test_client, mock_rag_system):
        mock_rag_system.query.return_value = ("Answer", [])
        response = test_client.post("/api/query", json={"query": "test"})
        assert response.status_code == 200
        assert response.json()["sources"] == []


class TestCoursesEndpoint:
    """Tests for GET /api/courses"""

    def test_returns_200(self, test_client):
        response = test_client.get("/api/courses")
        assert response.status_code == 200

    def test_response_has_required_fields(self, test_client):
        response = test_client.get("/api/courses")
        data = response.json()
        assert "total_courses" in data
        assert "course_titles" in data

    def test_total_courses_matches_mock(self, test_client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 5,
            "course_titles": ["A", "B", "C", "D", "E"],
        }
        response = test_client.get("/api/courses")
        assert response.json()["total_courses"] == 5

    def test_course_titles_match_mock(self, test_client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 2,
            "course_titles": ["Course A", "Course B"],
        }
        response = test_client.get("/api/courses")
        assert response.json()["course_titles"] == ["Course A", "Course B"]

    def test_empty_catalog_returns_zero_count(self, test_client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }
        response = test_client.get("/api/courses")
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_returns_500_when_analytics_raises(self, test_client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = RuntimeError("DB error")
        response = test_client.get("/api/courses")
        assert response.status_code == 500

    def test_500_detail_contains_error_message(self, test_client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = RuntimeError("DB error")
        response = test_client.get("/api/courses")
        assert "DB error" in response.json()["detail"]


class TestRootEndpoint:
    """Tests for GET /"""

    def test_root_returns_200(self, test_client):
        response = test_client.get("/")
        assert response.status_code == 200
