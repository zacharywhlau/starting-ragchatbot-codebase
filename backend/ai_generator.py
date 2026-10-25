import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **Up to 2 sequential searches per query** — use a second search when the first reveals you need more specific information (e.g., get a course outline first, then search for a specific lesson topic)
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives
- Use **get_course_outline** for questions about a course's structure, syllabus, or lesson list

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **Outline queries** (e.g. "what lessons are in X", "show me the syllabus for X"): call get_course_outline, then return the course title, course link, and a numbered list of all lesson titles — include every lesson, no truncation
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    # Safety cap on how many tool-use rounds a single query may trigger
    MAX_TOOL_ROUNDS = 2

    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = anthropic.Anthropic(**client_kwargs)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "max_tokens": 2048
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [{"role": "user", "content": query}]

        # Tool-use loop: keep answering tool calls until the model stops
        # requesting them (bounded so a runaway loop can't spin forever)
        for _ in range(self.MAX_TOOL_ROUNDS):
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content
            }
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            response = self.client.messages.create(**api_params)

            if response.stop_reason == "tool_use" and tool_manager:
                tool_results, had_error = self._run_tools(response, tool_manager)
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
                if had_error:
                    break
                continue

            return self._extract_text(response)

        # Tool rounds exhausted - make one final request for a text answer
        final_response = self.client.messages.create(**{
            **self.base_params,
            "messages": messages,
            "system": system_content
        })
        return self._extract_text(final_response)

    def _run_tools(self, response, tool_manager) -> tuple[List[Dict[str, Any]], bool]:
        """Execute every tool_use block in a response and return (tool_result blocks, had_error)."""
        tool_results = []
        had_error = False
        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name,
                        **content_block.input
                    )
                except Exception as e:
                    tool_result = f"Error: {e}"
                    had_error = True
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
        return tool_results, had_error

    def _extract_text(self, response) -> str:
        """Join all text blocks, with a fallback if the model returned none."""
        text = "".join(
            block.text for block in response.content
            if getattr(block, "type", None) == "text"
        )
        return text or "I wasn't able to generate a response for that query. Please try rephrasing it."