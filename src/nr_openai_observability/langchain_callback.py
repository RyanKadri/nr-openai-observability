from typing import Any, Dict, List, Optional, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, BaseMessage, LLMResult

from nr_openai_observability import monitor
import newrelic.agent


class NewRelicCallbackHandler(BaseCallbackHandler):
    def __init__(
        self,
        application_name: str,
        langchain_callback_metadata: Dict[str, Any] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize callback handler."""
        self.application_name = application_name

        self.new_relic_monitor = monitor.initialization(
            application_name=application_name,
            **kwargs,
        )
        self.langchain_callback_metadata = langchain_callback_metadata
        self.tool_invocation_counter = 0

    def get_and_update_tool_invocation_counter(self):
        self.tool_invocation_counter += 1
        return self.tool_invocation_counter

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any,
    ) -> Any:
        """Run when Chat Model starts running."""
        invocation_params = kwargs.get("invocation_params", {})
        tags = {
            "model": invocation_params.get("model"),
            "model_name": invocation_params.get("model_name"),
            "temperature": invocation_params.get("temperature"),
            "request_timeout": invocation_params.get("request_timeout"),
            "max_tokens": invocation_params.get("max_tokens"),
            "stream": invocation_params.get("stream"),
            "n": invocation_params.get("n"),
            "temperature": invocation_params.get("temperature"),
        }

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        tags = {}

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""
        tags = {"error": str(error)}

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Run when chain starts running."""
        trace = newrelic.agent.current_trace()
        node = trace.create_node()
        tags = {
            "input": inputs.get("input"),
            "run_id": str(kwargs.get("run_id")),
            "start_tags": str(kwargs.get("tags")),
            "start_metadata": str(kwargs.get("metadata")),
        }

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        tags = {
            "outputs": outputs.get("output"),
            "run_id": str(kwargs.get("run_id")),
            "end_tags": str(kwargs.get("tags")),
        }

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""
        tags = {error: str(error)}

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        tags = {
            "tool_name": serialized.get("name"),
            "tool_description": serialized.get("description"),
            "tool_input": input_str,
        }

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        tags = {
            "tool_output": output,
            "tool_invocation_counter": self.get_and_update_tool_invocation_counter(),
        }

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when tool errors."""
        tags = {
            "error": error,
        }

    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
