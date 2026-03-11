"""
Prompt templates for LLM intent extraction and result explanation.
UPDATED: Enhanced detection of "summary" keyword for statistical summaries.
UPDATED: Added selected_node context support for 3D viewer integration.
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
        time_range: tuple[datetime, datetime],
        selected_node: str = None
    ) -> str:
        """
        Generate prompt for extracting structured task specification from natural language.
        
        Args:
            user_query: The user's natural language query
            available_sensors: List of valid sensor types in the system
            available_locations: List of valid locations
            time_range: Tuple of (earliest_datetime, latest_datetime) for available data
            selected_node: Currently selected node in 3D viewer (optional)
            
        Returns:
            Formatted prompt string
        """
        current_date = datetime.now(timezone.utc)
        
        # Add selected node context if available
        selected_node_context = ""
        if selected_node:
            selected_node_context = f"""
SELECTED NODE CONTEXT:
The user has currently selected: {selected_node}

IMPORTANT LOCATION RULES:
- If the user's query does NOT explicitly mention a location, use the selected node: {selected_node}
- If the user explicitly mentions location(s), use those instead (ignore selected node)
- Examples with {selected_node} selected:
  * "average temperature yesterday" → location should be "{selected_node}"
  * "show me humidity last week" → location should be "{selected_node}"
  * "compare Node 1 and Node 5" → location should be ["Node 1", "Node 5"] (explicit, ignore selected)
  * "temperature in Node 10" → location should be "Node 10" (explicit, ignore selected)
"""
        
        prompt = f"""You are a task extraction assistant for a smart building analytics system.
Your job is to convert natural language queries into structured JSON task specifications.

Data available from: {time_range[0].strftime('%Y-%m-%d')} to {time_range[1].strftime('%Y-%m-%d')}
Current date: {current_date.strftime('%Y-%m-%d')}

AVAILABLE LOCATIONS:
{', '.join(available_locations)}

AVAILABLE SENSOR TYPES:
{', '.join(available_sensors)}
{selected_node_context}
USER QUERY:
{user_query}

OUTPUT FORMAT — return ONLY this JSON, no markdown, no explanation:

{{
  "intent_type": "<query|comparison|aggregation|threshold>",
  "sensor_type": "<temperature|humidity|moisture|strain>",
  "location": "<single location string OR list of strings for comparison>",
  "start_time": "<ISO 8601 datetime with timezone>",
  "end_time": "<ISO 8601 datetime with timezone>",
  "operation": "<mean|max|min|sum|std|count|summary>",
  "aggregation_level": "<hourly|daily|weekly|null>",
  "confidence": <0.0-1.0>,
  "threshold_value": <number or null>,
  "threshold_operator": "<>|>=|<|<=|null>",
  "result_threshold": <number or null>
}}

════════════════════════════════════════
CLASSIFICATION — work through A→E in order. Use the FIRST match.
════════════════════════════════════════

A) THRESHOLD — matches when ALL of:
   • Query asks WHICH or WHERE (not asking about one named node)
   • Contains a sensor threshold: "exceeded X", "above X", "below X", "over X°"
   → intent_type = "threshold"
   → location = "{available_locations[0] if available_locations else 'Node 1'}"  ← MUST be exactly ONE valid location string, NOT a list, NOT multiple nodes
   → operation = "mean"  ← placeholder
   → aggregation_level = null
   → threshold_value = the numeric sensor value (e.g. 25.0)
   → threshold_operator: "exceeded/above/over/more than" → ">"  |  "below/under/less than" → "<"
   → result_threshold = secondary % cutoff if stated (e.g. "more than 50% of the time" → 50.0), else 0.0
   EXAMPLES:
     "Which nodes exceeded 25°C more than 50% of the time?" → threshold, threshold_value=25.0, threshold_operator=">", result_threshold=50.0
     "Where did humidity go above 60% for over 30% of last week?" → threshold, threshold_value=60.0, threshold_operator=">", result_threshold=30.0

B) SUMMARY — matches when ANY of these words appear in the query (REGARDLESS of other words):
   "summary", "summarize", "statistics", "stats", "distribution", "overview"
   → intent_type = "query"
   → operation = "summary"
   → aggregation_level = null
   → location = the named node (single string)
   !! THIS RULE IS ABSOLUTE. If "summary", "statistics", "stats", "overview", or "distribution"
      appear anywhere in the query, set operation="summary" and intent_type="query". No exceptions. !!
   EXAMPLES:
     "Provide a summary of moisture in Node 15 last month" → query, operation=summary
     "Give me statistics for Node 4" → query, operation=summary
     "What are the stats for humidity in Node 8 last week?" → query, operation=summary
     "Overview of temperature in Node 1" → query, operation=summary

C) AGGREGATION — matches when query explicitly requests time-grouped output AND rule B did not match:
   Required signal words (at least one must be present):
     "each day", "each hour", "each week", "daily breakdown",
     "by day", "by hour", "per day", "hourly", "day by day", "every day"
   → intent_type = "aggregation"
   → aggregation_level:
       "each day" / "daily" / "by day" / "per day" / "day by day" / "every day" → "daily"
       "each hour" / "hourly" / "by hour" → "hourly"
       "each week" / "weekly" / "by week" → "weekly"
   → location = the named node (single string)
   EXAMPLES:
     "What was the temperature each day in Node 1 last week?" → aggregation, daily
     "Show me hourly humidity in Node 4 yesterday" → aggregation, hourly
     "Daily breakdown of moisture in Node 15" → aggregation, daily
     "Temperature in Node 8 every day last month" → aggregation, daily
   NOT aggregation (no time-grouping signal words → use rule E instead):
     "What was the highest moisture level over the past year?" → query, max
     "Maximum temperature last month in Node 4" → query, max
     "What was the temperature in Node 1 last week?" → query, mean

D) COMPARISON — matches when query names TWO OR MORE locations, or uses "compare/vs/versus":
   → intent_type = "comparison"
   → location = list of location strings, e.g. ["Node 15", "Node 25"]
   → aggregation_level = null
   EXAMPLES:
     "Compare humidity between Node 15 and Node 25" → comparison, ["Node 15","Node 25"]
     "Temperature in Node 1 vs Node 8 last week" → comparison, ["Node 1","Node 8"]

E) QUERY (default) — everything else:
   → intent_type = "query"
   → aggregation_level = null
   → operation: "average/avg" → "mean" | "maximum/max/highest" → "max" | "minimum/min/lowest" → "min" | default → "mean"
   EXAMPLES:
     "What was the average temperature in Node 15 yesterday?" → query, mean
     "Max humidity in Node 4 last week" → query, max
     "What was the temperature in Node 1 last month?" → query, mean

════════════════════════════════════════
DATE RULES
════════════════════════════════════════
- "yesterday"   → {(current_date - timedelta(days=1)).strftime('%Y-%m-%d')}T00:00:00+00:00 to {(current_date - timedelta(days=1)).strftime('%Y-%m-%d')}T23:59:59+00:00
- "last week"   → {(current_date - timedelta(days=7)).strftime('%Y-%m-%d')}T00:00:00+00:00 to {current_date.strftime('%Y-%m-%d')}T23:59:59+00:00
- "last month"  → {(current_date - timedelta(days=30)).strftime('%Y-%m-%d')}T00:00:00+00:00 to {current_date.strftime('%Y-%m-%d')}T23:59:59+00:00
- "past year"   → {(current_date - timedelta(days=365)).strftime('%Y-%m-%d')}T00:00:00+00:00 to {current_date.strftime('%Y-%m-%d')}T23:59:59+00:00
- Named month (e.g. "June 2025") → first day T00:00:00+00:00 to last day T23:59:59+00:00
- Always include timezone offset (+00:00)

════════════════════════════════════════
FIELD RULES
════════════════════════════════════════
- threshold : threshold_value and threshold_operator REQUIRED; result_threshold = 0.0 if not stated
- comparison     : location MUST be a JSON array with 2+ strings
- aggregation    : location MUST be a single string; aggregation_level MUST be "hourly"|"daily"|"weekly"
- query/summary  : location MUST be a single string; aggregation_level = null
- Non-threshold: threshold_value = null, threshold_operator = null, result_threshold = null

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

NODE:
{task_spec["location"]}

INSTRUCTIONS:
Provide a clear, concise natural language explanation of the results.

GUIDELINES:
1. Directly answer the user's question
2. Include the numerical result with units
3. Add relevant context (time range from actual_time_range). Do NOT describe the
   number of days or periods as a "sample size" for aggregation results — the
   date range already conveys this.
   IMPORTANT: Use "actual_time_range" from the results for ANY date/time references.
   Do NOT use start_time/end_time from the task specification — the actual data may
   cover a shorter period than what was requested (e.g. sensor data may have stopped early).
4. Mention any data quality notes if present
5. Keep response to 2-3 sentences
6. Do NOT hallucinate or add information not in the results
7. Use natural, conversational language

RESULT TYPE RULES — vary your response based on what "value" contains:
- Scalar (single number): state the value and units. For max/min operations, if
  "extreme_timestamp" is present at the TOP LEVEL of the result (not inside metadata),
  use it to state when the extreme occurred. Do NOT use actual_time_range end date as
  a substitute. If extreme_timestamp is null or absent, do NOT invent or guess a date.
- If "formatted_aggregation" is present in the results: copy it VERBATIM into your
  response, one entry per line. Do NOT summarize, average, or reformat it.
- List of {{timestamp, value}} objects (aggregation): present EACH period on its own line
  as "- [date]: [value] [unit]". Do NOT average them into a single number.
- Dict (summary statistics): report mean, min, max, std, and median.
- List of {{location, percent_time}} objects (threshold scan): list each qualifying
  location and its percent_time value.

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