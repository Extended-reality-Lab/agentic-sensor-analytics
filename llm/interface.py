"""
Abstract interface for LLM components.
Defines the contract that all LLM implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
from enum import Enum
from .prompts import SystemContext


class IntentType(str, Enum):
    """Supported query intent types."""
    QUERY = "query"
    COMPARISON = "comparison"
    AGGREGATION = "aggregation"
    THRESHOLD_SCAN = "threshold_scan"


class Operation(str, Enum):
    """Supported analytics operations."""
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    SUM = "sum"
    STD = "std"
    COUNT = "count"
    SUMMARY = "summary"


class AggregationLevel(str, Enum):
    """Supported temporal aggregation levels."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"


class TaskSpecification(BaseModel):
    """
    Structured representation of a user's analytics task.
    This is the output of LLM intent extraction.
    """
    intent_type: IntentType = Field(
        description="Type of query: simple query, comparison, or aggregation"
    )
    sensor_type: str = Field(
        description="Type of sensor data to query (e.g., temperature, humidity)"
    )
    location: str | list[str] = Field(
        description="Single location string or list of locations for comparison"
    )
    start_time: datetime = Field(
        description="Start of time range for query"
    )
    end_time: datetime = Field(
        description="End of time range for query"
    )
    operation: Operation = Field(
        description="Analytics operation to perform"
    )
    aggregation_level: Optional[AggregationLevel] = Field(
        default=None,
        description="Temporal aggregation level (if applicable)"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="LLM's confidence in the extraction (0.0-1.0)"
    )

    # Threshold scan fields
    threshold_value: Optional[float] = Field(
        default=None,
        description="Numeric threshold to test each reading against (threshold_scan only)"
    )
    threshold_operator: Optional[str] = Field(
        default=None,
        description="Comparison operator: '>' | '>=' | '<' | '<=' (threshold_scan only)"
    )
    result_threshold: Optional[float] = Field(
        default=None,
        description="Secondary filter: keep locations where percent_time satisfies this value (threshold_scan only)"
    )

    @field_validator('end_time')
    @classmethod
    def end_after_start(cls, v: datetime, info) -> datetime:
        """Validate that end_time is after start_time."""
        if 'start_time' in info.data and v <= info.data['start_time']:
            raise ValueError("end_time must be after start_time")
        return v

    @model_validator(mode='after')
    def validate_location_matches_intent(self) -> 'TaskSpecification':
        """Validate location matches intent type."""
        if self.intent_type == IntentType.COMPARISON and isinstance(self.location, str):
            raise ValueError("Comparison queries require multiple locations")
        if self.intent_type == IntentType.QUERY and isinstance(self.location, list):
            raise ValueError("Simple queries require single location")
        return self

    def get_locations_list(self) -> list[str]:
        """Get locations as a list regardless of whether it's string or list."""
        return [self.location] if isinstance(self.location, str) else self.location

class LLMInterface(ABC):
    """
    Abstract base class for LLM components.
    All LLM implementations must inherit from this class.
    """

    @abstractmethod
    def extract_intent(
        self,
        user_query: str,
        system_context: SystemContext
) -> TaskSpecification:
        """
        Extract structured task specification from natural language query.
        
        Args:
            user_query: The user's natural language question
            system_context: Available sensors, locations, and time ranges
            
        Returns:
            TaskSpecification object with extracted parameters
            
        Raises:
            ValueError: If the query cannot be parsed
            LLMError: If the LLM fails to generate valid output
        """
        pass

    @abstractmethod
    def explain_results(
        self,
        original_query: str,
        task_spec: TaskSpecification,
        results: list[dict]
    ) -> str:
        """
        Convert analytics results into natural language explanation.
        
        Args:
            original_query: The user's original question
            task_spec: The structured task that was executed
            results: List of analytics results with metadata
            
        Returns:
            Natural language explanation of the results
            
        Raises:
            LLMError: If explanation generation fails
        """
        pass


    @abstractmethod
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
        pass

    def is_available(self) -> bool:
        """
        Check if the LLM is available and ready to use.
        
        Returns:
            True if LLM is loaded and ready, False otherwise
        """
        return True


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class LLMParseError(LLMError):
    """Raised when LLM output cannot be parsed into expected format."""
    pass


class LLMGenerationError(LLMError):
    """Raised when LLM fails to generate output."""
    pass