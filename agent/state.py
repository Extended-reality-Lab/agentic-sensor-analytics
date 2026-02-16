"""
Agent state definitions for LangGraph workflow.
Defines the state structure that flows through the agent graph.
"""

from typing import TypedDict, Optional, List, Dict, Any
from datetime import datetime
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from llm import TaskSpecification


class AgentState(TypedDict, total=False):
    """
    State that flows through the agent graph.
    Each node can read from and write to this state.
    
    Attributes:
        user_query: Original natural language query from user
        task_spec: Extracted TaskSpecification from LLM
        validation_errors: List of validation error messages
        data: Retrieved sensor data as DataFrame
        analytics_result: Result from analytics tool execution
        explanation: Natural language explanation of results
        explanation_stream: Generator for streaming explanation
        execution_trace: List of execution steps for debugging
        error_explanation: User-friendly error message
        success: Whether the workflow completed successfully
    """
    # Input
    user_query: str
    
    # LLM extraction
    task_spec: Optional[TaskSpecification]
    
    # Validation
    validation_errors: List[str]
    
    # Data retrieval
    data: Optional[pd.DataFrame]
    
    # Analytics execution
    analytics_result: Optional[Dict[str, Any]]
    
    # Result generation
    explanation: Optional[str]
    explanation_stream: Optional[Any]
    error_explanation: Optional[str]
    use_streaming: bool
    
    # Workflow tracking
    execution_trace: List[Dict[str, Any]]
    success: bool
    
    # Timestamps for performance tracking
    start_time: Optional[datetime]
    end_time: Optional[datetime]

    selected_node: Optional[str]


class ExecutionTrace(TypedDict):
    """
    Single entry in the execution trace.
    Records what happened at each step of the workflow.
    """
    step: str
    timestamp: datetime
    status: str  # 'started', 'completed', 'failed'
    details: Dict[str, Any]
    duration_ms: Optional[float]


def create_initial_state(user_query: str, use_streaming: bool = True) -> AgentState:
    """
    Create initial agent state from user query.
    
    Args:
        user_query: Natural language query from user
        
    Returns:
        AgentState with initial values
    """
    return AgentState(
        user_query=user_query,
        task_spec=None,
        validation_errors=[],
        data=None,
        analytics_result=None,
        explanation=None,
        explanation_stream=None,
        use_streaming=use_streaming,
        error_explanation=None,
        execution_trace=[],
        success=False,
        start_time=datetime.now(),
        end_time=None
    )


def add_trace_entry(
    state: AgentState,
    step: str,
    status: str,
    details: Dict[str, Any],
    duration_ms: Optional[float] = None
) -> None:
    """
    Add an entry to the execution trace.
    
    Args:
        state: Current agent state
        step: Name of the step
        status: Status of the step ('started', 'completed', 'failed')
        details: Additional details about the step
        duration_ms: Optional duration in milliseconds
    """
    trace_entry: ExecutionTrace = {
        'step': step,
        'timestamp': datetime.now(),
        'status': status,
        'details': details,
        'duration_ms': duration_ms
    }
    
    if 'execution_trace' not in state:
        state['execution_trace'] = []
    
    state['execution_trace'].append(trace_entry)


def get_execution_summary(state: AgentState) -> Dict[str, Any]:
    """Generate summary of execution from trace."""
    if not state.get('execution_trace'):
        return {}
    
    unique_steps = set()
    completed_steps = set()
    failed_steps = set()
    total_duration = 0.0
    
    for entry in state['execution_trace']:
        step_name = entry['step']
        unique_steps.add(step_name)
        
        if entry.get('duration_ms'):
            total_duration += entry['duration_ms']
        
        if entry['status'] == 'completed':
            completed_steps.add(step_name)
        elif entry['status'] == 'failed':
            failed_steps.add(step_name)
    
    return {
        'total_steps': len(unique_steps),
        'steps_completed': len(completed_steps),
        'steps_failed': len(failed_steps),
        'total_duration_ms': total_duration,
        'success': state.get('success', False)
    }