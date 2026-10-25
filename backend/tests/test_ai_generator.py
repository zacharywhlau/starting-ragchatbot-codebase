import pytest
from unittest.mock import MagicMock, patch, call
from ai_generator import AIGenerator


def make_text_response(text):
    """Create a mock response with text content."""
    response = MagicMock()
    block = MagicMock()
    block.type = "text"
    block.text = text
    response.content = [block]
    response.stop_reason = "end_turn"
    response.stop_details = None
    return response


def make_tool_use_response(tool_name, tool_id, input_dict):
    """Create a mock response with tool use."""
    response = MagicMock()
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.id = tool_id
    block.input = input_dict
    response.content = [block]
    response.stop_reason = "tool_use"
    response.stop_details = None
    return response


class TestAIGenerator:
    """Tests for AIGenerator"""

    def test_base_params_do_not_include_temperature(self):
        """Verify that base_params does NOT include temperature (API change 2026-10-21)."""
        gen = AIGenerator(api_key="test-key", model="claude-haiku-4-5")

        # DIAGNOSTIC: This test will FAIL on the current code, confirming the bug
        assert "temperature" not in gen.base_params, \
            "temperature parameter was removed from Anthropic API on 2026-10-21. Remove it from base_params."
        assert "max_tokens" in gen.base_params

    def test_direct_response_returns_text(self):
        """Single API call with text response returns the text."""
        with patch("ai_generator.anthropic.Anthropic") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.messages.create.return_value = make_text_response("Hello, world!")

            gen = AIGenerator(api_key="test-key", model="claude-haiku-4-5")
            result = gen.generate_response(query="Hello?", tools=None, tool_manager=None)

            assert result == "Hello, world!"
            mock_client.messages.create.assert_called_once()

    def test_tool_use_triggers_second_api_call(self):
        """Tool use response triggers a second API call."""
        with patch("ai_generator.anthropic.Anthropic") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # First call returns tool use, second returns text
            mock_client.messages.create.side_effect = [
                make_tool_use_response("my_tool", "tool_1", {"arg": "value"}),
                make_text_response("Final answer")
            ]

            mock_tool_manager = MagicMock()
            mock_tool_manager.execute_tool.return_value = "tool result"

            gen = AIGenerator(api_key="test-key", model="claude-haiku-4-5")
            result = gen.generate_response(
                query="test",
                tools=[{"name": "my_tool", "description": "test"}],
                tool_manager=mock_tool_manager
            )

            # Should have called messages.create twice
            assert mock_client.messages.create.call_count == 2
            assert result == "Final answer"

    def test_tool_result_appended_as_user_message(self):
        """Tool results are appended as user message."""
        with patch("ai_generator.anthropic.Anthropic") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.messages.create.side_effect = [
                make_tool_use_response("my_tool", "tool_1", {"arg": "value"}),
                make_text_response("Done")
            ]

            mock_tool_manager = MagicMock()
            mock_tool_manager.execute_tool.return_value = "tool result"

            gen = AIGenerator(api_key="test-key", model="claude-haiku-4-5")
            gen.generate_response(
                query="test",
                tools=[{"name": "my_tool"}],
                tool_manager=mock_tool_manager
            )

            # Check second call's messages parameter
            second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
            messages = second_call_kwargs["messages"]

            # Should have user, assistant (with tool use), and user (with tool result)
            assert len(messages) >= 3
            assert messages[-1]["role"] == "user"
            assert "tool_result" in str(messages[-1]["content"])

    def test_run_tools_returns_list_of_tool_result_dicts(self):
        """_run_tools returns a list of tool_result dicts."""
        with patch("ai_generator.anthropic.Anthropic") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_tool_manager = MagicMock()
            mock_tool_manager.execute_tool.return_value = "result string"

            gen = AIGenerator(api_key="test-key", model="claude-haiku-4-5")
            response = make_tool_use_response("test_tool", "id_123", {})

            result = gen._run_tools(response, mock_tool_manager)

            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0]["type"] == "tool_result"
            assert result[0]["tool_use_id"] == "id_123"
            assert result[0]["content"] == "result string"

    def test_run_tools_calls_tool_manager_with_correct_args(self):
        """_run_tools calls tool_manager.execute_tool with correct name and input."""
        with patch("ai_generator.anthropic.Anthropic") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_tool_manager = MagicMock()
            mock_tool_manager.execute_tool.return_value = "result"

            gen = AIGenerator(api_key="test-key", model="claude-haiku-4-5")
            response = make_tool_use_response("my_tool", "id_1", {"param": "value"})

            gen._run_tools(response, mock_tool_manager)

            mock_tool_manager.execute_tool.assert_called_once_with("my_tool", param="value")

    def test_extract_text_joins_multiple_text_blocks(self):
        """_extract_text joins all text blocks."""
        with patch("ai_generator.anthropic.Anthropic") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            response = MagicMock()
            block1 = MagicMock()
            block1.type = "text"
            block1.text = "Hello "

            block2 = MagicMock()
            block2.type = "text"
            block2.text = "world!"

            response.content = [block1, block2]

            gen = AIGenerator(api_key="test-key", model="claude-haiku-4-5")
            result = gen._extract_text(response)

            assert result == "Hello world!"

    def test_extract_text_fallback_on_empty_content(self):
        """_extract_text returns fallback message when no text blocks."""
        with patch("ai_generator.anthropic.Anthropic") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            response = MagicMock()
            response.content = []

            gen = AIGenerator(api_key="test-key", model="claude-haiku-4-5")
            result = gen._extract_text(response)

            assert "wasn't able to generate a response" in result

    def test_system_prompt_includes_history_when_provided(self):
        """System prompt includes conversation history when provided."""
        with patch("ai_generator.anthropic.Anthropic") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.messages.create.return_value = make_text_response("response")

            gen = AIGenerator(api_key="test-key", model="claude-haiku-4-5")
            history = "User: Hi\nAssistant: Hello"

            gen.generate_response(
                query="Follow up?",
                conversation_history=history,
                tools=None,
                tool_manager=None
            )

            call_kwargs = mock_client.messages.create.call_args[1]
            system_content = call_kwargs["system"]

            assert "Follow up?" in system_content or history in system_content

    def test_api_params_do_not_send_temperature(self):
        """Verify that temperature is not sent in API params."""
        with patch("ai_generator.anthropic.Anthropic") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.messages.create.return_value = make_text_response("response")

            gen = AIGenerator(api_key="test-key", model="claude-haiku-4-5")
            gen.generate_response(query="test", tools=None, tool_manager=None)

            call_kwargs = mock_client.messages.create.call_args[1]

            assert "temperature" not in call_kwargs, \
                "temperature should not be sent to API (removed 2026-10-21)"
            assert call_kwargs["max_tokens"] == 2048
