from abc import ABC, abstractmethod
from typing import Dict, Any, NamedTuple
import os
from anthropic import Anthropic, AnthropicError 
import logging
from pathlib import Path
import json
# from utils import load_json

def read_file(file_path: Path) -> str | None:
    """Reads a file and returns its content."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

def load_json(file_path: Path) -> dict | None:
    """Loads a JSON file."""
    content = read_file(file_path)
    if content:
        return json.loads(content)
    return None

class LLMResponse(NamedTuple):
    text: str
    model_identifier: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    metadata: Dict[str, Any] = {}

class BaseLLMClient(ABC):
    @abstractmethod
    def query(self, prompt: str, params: Dict[str, Any]) -> LLMResponse:
        pass

class AnthropicClaudeClient(BaseLLMClient):

    """
    LLM Client for interacting with Anthropic's Claude models.
    """
    def __init__(self, api_key: str | None = None, model_identifier: str = "claude-opus-4-5-20251101"):
        self.model_identifier = model_identifier
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided or found in ANTHROPIC_API_KEY environment variable.")
        
        self.client = Anthropic(api_key=self.api_key)

    def _load_llm_config(self, task_type: str, llm_config_path: Path = Path("llm_configs")) -> dict:
        if not llm_config_path.exists():
            raise ValueError(f"LLM config file not found for task type: {task_type}")
        llm_config_path = llm_config_path / f"{task_type}.json"
        return load_json(llm_config_path)

    def query(self, prompt: str, task_type: str) -> LLMResponse:
        params = self._load_llm_config(task_type)
        logging.info(f"AnthropicClaudeClient querying model {self.model_identifier} with params: {params}")
        max_tokens = params.get("max_tokens", params.get("max_new_tokens", 2048))
        temperature = params.get("temperature", 0.5)
        top_p = params.get("top_p", None)
        top_k = params.get("top_k", None)
        kwargs = {}
        if top_p:
            kwargs["top_p"] = top_p
        if top_k:
            kwargs["top_k"] = top_k

        messages = [{"role": "user", "content": prompt}]
        system_prompt = params.get("system", None)

        try:
            response = self.client.messages.create(
                model=self.model_identifier,
                max_tokens=max_tokens,
                messages=messages,
                temperature=temperature,
                system=system_prompt,
                stream=True,
                **kwargs
            )
            response_text_parts = []
            prompt_tokens = completion_tokens = None
            for event in response:
                t = event.type
                if t == "message_start":
                    # available at start
                    prompt_tokens = getattr(event.message.usage, "input_tokens", None)
                elif t == "content_block_start":
                    # initial text chunk (sometimes empty)
                    blk = getattr(event, "content_block", None)
                    if getattr(blk, "text", None):
                        response_text_parts.append(blk.text)
                elif t == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if getattr(delta, "text", None):
                        response_text_parts.append(delta.text)
                elif t == "message_delta":
                    usage = getattr(event, "usage", None)
                    if usage:
                        completion_tokens = getattr(usage, "output_tokens", None)

            response_text = "".join(response_text_parts)

            return LLMResponse(
                text=response_text,
                model_identifier=self.model_identifier,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                metadata={"streaming": True}
            )
            # response_text = ""
            # if response.content and isinstance(response.content, list):
            #     for block in response.content:
            #         if block.type == "text":
            #             response_text += block.text
            
            # prompt_tokens = response.usage.input_tokens if response.usage else None
            # completion_tokens = response.usage.output_tokens if response.usage else None
            # logging.info(f"prompt tokens: {prompt_tokens}, completion tokens: {completion_tokens}")
            
            # return LLMResponse(
            #     text=response_text,
            #     model_identifier=self.model_identifier,
            #     prompt_tokens=prompt_tokens,
            #     completion_tokens=completion_tokens,
            #     metadata={
            #         "anthropic_model_used": response.model,
            #         "stop_reason": response.stop_reason,
            #         "params_sent": {"max_tokens": max_tokens, "temperature": temperature, "system_prompt_length": len(system_prompt or "")}
            #     }
            # )
        except AnthropicError as e:
            return LLMResponse(
                text=f"Anthropic API Error: {str(e)}",
                model_identifier=self.model_identifier,
                metadata={"error": True, "details": str(e)}
            )
        except Exception as e:
            return LLMResponse(
                text=f"Unexpected Error: {str(e)}",
                model_identifier=self.model_identifier,
                metadata={"error": True, "details": str(e)}
            )
