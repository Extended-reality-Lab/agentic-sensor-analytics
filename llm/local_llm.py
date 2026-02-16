"""
Local LLM implementation using Ollama with native structured outputs.
Handles connection to Ollama server, streaming responses, and retry logic.
"""

import time
from typing import Optional
from pathlib import Path
import ollama
import os

from .interface import (
    LLMInterface,
    TaskSpecification,
    LLMError,
    LLMGenerationError,
    LLMParseError
)
from .prompts import SystemContext, PromptTemplates
from .parser import TaskSpecificationParser
from .config import LLMConfig, load_config


class OllamaLLM(LLMInterface):
    """
    LLM implementation using Ollama for local model inference.
    Uses Ollama's native structured output support with Pydantic schemas.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        config: Optional[LLMConfig] = None
    ):
        """
        Initialize Ollama client.
        
        Args:
            model_name: Name of the Ollama model to use (overrides config)
            base_url: Base URL for Ollama server (overrides config)
            temperature: Sampling temperature (overrides config)
            timeout: Request timeout in seconds (overrides config)
            max_retries: Number of retries on failure (overrides config)
            config: LLMConfig object (if None, loads from default locations)
        """
        # Load config if not provided
        if config is None:
            config = load_config()
        
        self.config = config
        
        # Use provided values or fall back to config
        self.model_name = model_name or config.llm.model_name

        self.base_url = (
            base_url or
            os.getenv('OLLAMA_BASE_URL') or
            config.llm.base_url
        )

        self.temperature = temperature if temperature is not None else config.llm.temperature
        self.timeout = timeout or config.llm.timeout
        self.max_retries = max_retries or config.llm.max_retries
        self.min_confidence = config.llm.min_confidence
        
        # Initialize Ollama client
        self.client = ollama.Client(host=self.base_url)
    
    @classmethod
    def from_config(cls, config_path: Optional[str | Path] = None) -> 'OllamaLLM':
        """
        Create OllamaLLM instance from configuration file.
        
        Args:
            config_path: Path to YAML configuration file (optional)
            
        Returns:
            Configured OllamaLLM instance
            
        Example:
            llm = OllamaLLM.from_config('llm_config.yaml')
        """
        config = load_config(config_path)
        return cls(config=config)
    
    def _generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        stream: bool = False
    ) -> str:
        """
        Generate completion from Ollama with retry logic.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            stream: Whether to stream the response
            
        Returns:
            Generated text
            
        Raises:
            LLMGenerationError: If generation fails after retries
        """
        # Use streaming setting from config if not explicitly set
        if not stream:
            stream = self.config.performance.enable_streaming
        
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    options={
                        "temperature": self.temperature,
                        "num_predict": self.config.llm.max_tokens
                    },
                    stream=stream
                )
                
                if stream:
                    # Collect streamed response
                    full_response = ""
                    for chunk in response:
                        if 'message' in chunk and 'content' in chunk['message']:
                            full_response += chunk['message']['content']
                    return full_response
                else:
                    return response['message']['content']
                    
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                else:
                    raise LLMGenerationError(
                        f"Failed to generate response after {self.max_retries} attempts: {e}"
                    )
        
        raise LLMGenerationError("Unexpected error in generation")
    
    def extract_intent(
        self,
        user_query: str,
        system_context: SystemContext,
        selected_node: str = None 
    ) -> TaskSpecification:
        """
        Extract structured task specification from natural language query.
        Uses Ollama's native structured output support with Pydantic schema.
        
        Args:
            user_query: The user's natural language question
            system_context: Available sensors, locations, and time ranges
            
        Returns:
            TaskSpecification object with extracted parameters
            
        Raises:
            LLMParseError: If the query cannot be parsed
            LLMGenerationError: If the LLM fails to generate valid output
        """
        # Generate prompt
        prompt = PromptTemplates.get_intent_extraction_prompt(
            user_query=user_query,
            available_sensors=system_context.available_sensors,
            available_locations=system_context.available_locations,
            time_range=system_context.time_range,
            selected_node=selected_node
        )
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Use Ollama's native structured JSON output 
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    format=TaskSpecification.model_json_schema(),
                    options={
                        "temperature": self.temperature,
                        "num_predict": self.config.llm.max_tokens
                    }
                )
                
                # Parse and validate response using Pydantic
                task_spec = TaskSpecification.model_validate_json(
                    response['message']['content']
                )
                
                # Validate against system context
                errors = TaskSpecificationParser.validate_against_context(
                    task_spec=task_spec,
                    available_sensors=system_context.available_sensors,
                    available_locations=system_context.available_locations,
                    time_range=system_context.time_range
                )
                
                if errors:
                    if attempt < self.max_retries - 1:
                        # Retry with error feedback
                        error_msg = "Previous extraction had errors:\n" + "\n".join(f"- {e}" for e in errors)
                        messages.append({
                            "role": "assistant",
                            "content": response['message']['content']
                        })
                        messages.append({
                            "role": "user",
                            "content": f"{error_msg}\n\nPlease correct the extraction."
                        })
                        continue
                    else:
                        # Generate user-friendly error explanation
                        error_explanation = self.explain_error(user_query, errors)
                        raise LLMParseError(error_explanation)
                
                # Success!
                return task_spec
                
            except LLMParseError:
                raise
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                else:
                    raise LLMGenerationError(f"Failed to extract intent: {e}")
        
        raise LLMGenerationError("Failed to extract intent after all retries")
    
    def explain_results(
        self,
        original_query: str,
        task_spec: TaskSpecification,
        results: list[dict],
        stream: bool = False
    ):
        """
        Convert analytics results into natural language explanation.
        
        Args:
            original_query: The user's original question
            task_spec: The structured task that was executed
            results: List of analytics results with metadata
            stream: Whether to stream the response (returns generator if True)
            
        Returns:
            Natural language explanation string if stream=False,
            or generator yielding chunks if stream=True
            
        Raises:
            LLMGenerationError: If explanation generation fails
        """
        # Convert task_spec to dict for prompt
        task_spec_dict = task_spec.model_dump()
        
        # Generate prompt
        prompt = PromptTemplates.get_result_explanation_prompt(
            original_query=original_query,
            task_spec=task_spec_dict,
            results=results
        )
        
        # Build messages
        messages = [{"role": "user", "content": prompt}]
        
        if stream:
            # Return a generator that yields chunks
            return self._stream_chat(messages)
        else:
            # Get complete explanation
            try:
                explanation = self._generate(prompt=prompt, stream=False)
                return explanation.strip()
            except LLMGenerationError as e:
                raise LLMGenerationError(f"Failed to generate explanation: {e}")
    
    def _stream_chat(self, messages: list[dict]):
        """
        Internal generator method for streaming chat responses.
        
        Args:
            messages: List of message dictionaries
            
        Yields:
            String chunks from the streaming response
            
        Raises:
            LLMGenerationError: If streaming fails
        """
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.config.llm.max_tokens
                },
                stream=True
            )
            
            for chunk in response:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
                    
        except Exception as e:
            raise LLMGenerationError(f"Failed to stream response: {e}")
    
    def explain_error(
        self,
        user_query: str,
        errors: list[str]
    ) -> str:
        """
        Generate user-friendly explanation of validation errors.
        
        Args:
            user_query: The user's query
            errors: List of validation error messages
            
        Returns:
            User-friendly error explanation
        """
        prompt = PromptTemplates.get_error_explanation_prompt(
            user_query=user_query,
            errors=errors
        )
        
        try:
            explanation = self._generate(prompt=prompt, stream=False)
            return explanation.strip()
        except LLMGenerationError:
            # Fallback to simple error formatting if LLM fails
            return "I encountered some issues with your query:\n" + "\n".join(
                f"- {error}" for error in errors
            )
    
    def is_available(self) -> bool:
        """
        Check if the LLM is available and ready to use.
        
        Returns:
            True if LLM is loaded and ready, False otherwise
        """
        try:
            self.client.list()
            return True
        except Exception:
            return False
    
    def get_model_info(self) -> dict:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        try:
            models = self.client.list()
            for model in models.get('models', []):
                if model['name'] == self.model_name:
                    return model
            return {"error": "Model not found"}
        except Exception as e:
            return {"error": str(e)}