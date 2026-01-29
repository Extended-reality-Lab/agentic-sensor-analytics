"""
Prompt templates for LLM intent extraction and result explanation.
"""

from typing import List, Dict
from datetime import datetime, timedelta, timezone


class PromptTemplates:
    """Centralized prompt templates for the LLM component."""
        
    @staticmethod
    def get_intent_extraction_prompt(
        user_query: str,
        available_sensors: List[str],
        available_locations: List[str],
        time_range: tuple[datetime, datetime]
    ) -> str:
        """
        Generate prompt for extracting structured task specification from natural language.
        
        Args:
            user_query: The user's natural language query
            available_sensors: List of valid sensor types in the system
            available_locations: List of valid locations
            time_range: Tuple of (earliest_datetime, latest_datetime) for available data
            
        Returns:
            Formatted prompt string
        """
        current_date = datetime.now((timezone.utc))
        
        prompt = f"""You are a task extraction assistant for a smart building analytics system. 
    Your job is to convert natural language queries into structured JSON task specifications.

    Data available from: {time_range[0].strftime('%Y-%m-%d')} to {time_range[1].strftime('%Y-%m-%d')}
    Current date: {current_date.strftime('%Y-%m-%d')}

    AVAILABLE LOCATIONS:
    {', '.join(available_locations)}

    AVAILABLE SENSOR TYPES:
    {', '.join(available_sensors)}

    USER QUERY:
    {user_query}

    INSTRUCTIONS:
    Extract the following information and return ONLY valid JSON (no markdown, no explanations):

    {{
    "intent_type": "<query|comparison|aggregation>",
    "sensor_type": "<temperature|humidity|moisture|strain>",
    "location": "<single location string OR list of locations for comparison>",
    "start_time": "<ISO 8601 datetime>",
    "end_time": "<ISO 8601 datetime>",
    "operation": "<mean|max|min|sum|std|count|summary>",
    "aggregation_level": "<hourly|daily|weekly|null>",
    "confidence": <0.0-1.0 confidence score>
    }}

    INTENT TYPE SELECTION - Follow these rules IN ORDER:

    **STEP 1: Check for COMPARISON**
    If the query mentions TWO OR MORE locations OR uses comparison words:
    - Keywords: "compare", "vs", "versus", "between", "difference between", "which"
    - Examples: 
      * "Compare Node 14 and Node 15" → comparison
      * "Temperature between Node 11 vs Node 15" → comparison
      * "Which is warmer, Node 1 or Node 2" → comparison
    - Set: intent_type = "comparison"
    - Set: location = ["Location1", "Location2"]  // MUST be a list with 2+ items
    - Set: aggregation_level = null

    **STEP 2: Check for AGGREGATION**
    If the query asks for time-grouped/broken-down data (NOT just a range):
    - Keywords: "each day", "daily", "hourly", "weekly", "by day", "by hour", "per day", "breakdown", "day by day", "every day"
    - Examples:
      * "Temperature EACH DAY last week" → aggregation (daily)
      * "Show HOURLY averages" → aggregation (hourly)
      * "Daily breakdown of humidity" → aggregation (daily)
      * "What was temperature each day" → aggregation (daily)
    - Set: intent_type = "aggregation"
    - Set: location = "Single Location"  // MUST be a string
    - Set: aggregation_level = "hourly" | "daily" | "weekly"

    **STEP 3: Check for STATISTICAL SUMMARY**
    If the query explicitly asks for statistics, summary, or distribution:
    - Keywords: "statistical summary", "statistics", "stats", "summary", "distribution"
    - Examples:
      * "Give me a statistical summary" → triggers StatisticalSummaryTool
      * "Show statistics for temperature" → triggers StatisticalSummaryTool
    - Set: intent_type = "query"
    - Set: operation = "summary"  // This routes to StatisticalSummaryTool
    - Set: location = "Single Location"  // MUST be a string
    - Set: aggregation_level = null

    **STEP 4: DEFAULT to QUERY**
    Simple single-value statistic over a time period:
    - Examples:
      * "Average temperature last week" → query (returns ONE number)
      * "Max humidity yesterday" → query (returns ONE number)
      * "Temperature in Node 15 last week" → query (returns ONE number)
    - Set: intent_type = "query"
    - Set: location = "Single Location"  // MUST be a string
    - Set: aggregation_level = null

    CRITICAL VALIDATION:
    - comparison → location MUST be a list with 2+ items, aggregation_level = null
    - aggregation → location MUST be a string, aggregation_level MUST be "hourly"|"daily"|"weekly"
    - query → location MUST be a string, aggregation_level = null

    DATE PARSING RULES:
    - "yesterday" = previous day from current date (e.g., {(current_date - timedelta(days=1)).strftime('%Y-%m-%d')})
    - "last week" = past 7 days from current date (e.g., {(current_date - timedelta(days=7)).strftime('%Y-%m-%d')} to {current_date.strftime('%Y-%m-%d')})
    - "past month" = past 30 days from current date (e.g., {(current_date - timedelta(days=30)).strftime('%Y-%m-%d')} to {current_date.strftime('%Y-%m-%d')})
    - Always use {current_date.year} as the year for recent dates unless explicitly stated otherwise
    - All times should be in ISO 8601 format with timezone (use 'T00:00:00+00:00' for start of day, 'T23:59:59+00:00' for end of day)

    Return ONLY the JSON object.
    """
        return prompt
    
    @staticmethod
    def get_result_explanation_prompt(
        original_query: str,
        task_spec: Dict,
        results: List[Dict]
    ) -> str:
        """
        Generate prompt for explaining analytics results in natural language.
        
        Args:
            original_query: The user's original question
            task_spec: The structured task specification
            results: List of analytics results with metadata
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are explaining analytics results from a smart building system.

ORIGINAL USER QUERY:
{original_query}

TASK SPECIFICATION:
{task_spec}

ANALYTICS RESULTS:
{results}

INSTRUCTIONS:
Provide a clear, concise natural language explanation of the results.

GUIDELINES:
1. Directly answer the user's question
2. Include the numerical result with units
3. Add relevant context (sample size, time range)
4. Mention any data quality notes if present
5. Keep response to 2-3 sentences
6. Do NOT hallucinate or add information not in the results
7. Use natural, conversational language
8. CRITICAL: Use the EXACT location from TASK SPECIFICATION

Now explain the results above:
"""
        return prompt

    @staticmethod
    def get_error_explanation_prompt(
        user_query: str,
        errors: List[str]
    ) -> str:
        """
        Generate prompt for explaining validation errors to users.
        
        Args:
            user_query: The user's query
            errors: List of validation error messages
            
        Returns:
            Formatted prompt for error explanation
        """
        prompt = f"""Explain validation errors to the user in a helpful way.

USER QUERY:
{user_query}

VALIDATION ERRORS:
{errors}

Generate a user-friendly explanation that:
1. Explains what went wrong clearly
2. Suggests how to fix the query
3. Remains polite and helpful
4. Uses simple language (no technical jargon)

Return only the explanation, 2-3 sentences maximum.
"""
        return prompt


class SystemContext:
    """Container for system context passed to prompts."""
    
    def __init__(
        self,
        available_sensors: List[str],
        available_locations: List[str],
        time_range: tuple[datetime, datetime]
    ):
        self.available_sensors = available_sensors
        self.available_locations = available_locations
        self.time_range = time_range
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "available_sensors": self.available_sensors,
            "available_locations": self.available_locations,
            "time_range": [
                self.time_range[0].isoformat(),
                self.time_range[1].isoformat()
            ]
        }