"""
Agent module for agentic workflow orchestration.

This module implements the agentic layer of the smart building analytics system
using LangGraph for stateful workflow management.

Key Components:
    - AgentState: TypedDict defining the state structure
    - AgentNodes: Collection of node functions that transform state
    - AgentExecutor: High-level interface for executing queries
    - create_agent_graph: Graph construction function

Usage:
    # Simple usage with default config
    from agent import AgentExecutor
    
    executor = AgentExecutor.from_config()
    result = executor.execute("What was the average temperature in Node 15 yesterday?")
    
    if result['success']:
        print(result['explanation'])
    else:
        print(result['error_explanation'])
    
    # View execution trace
    print(executor.visualize_trace(result))

Architecture:
    The agent implements a state-based workflow with the following nodes:
    
    1. interpret_query: Extract TaskSpecification from natural language
    2. validate_task: Validate parameters against available data
    3. retrieve_data: Fetch sensor data from repository
    4. execute_analytics: Run analytics tools on data
    5. generate_explanation: Create natural language explanation
    6. handle_error: Generate user-friendly error messages
    
    The workflow includes conditional routing to handle errors gracefully
    at each step, ensuring robust execution.
"""

from .state import (
    AgentState,
    ExecutionTrace,
    create_initial_state,
    add_trace_entry,
    get_execution_summary
)

from .nodes import AgentNodes

from .graph import (
    create_agent_graph,
    AgentExecutor,
    should_continue,
    should_continue_after_retrieval,
    should_continue_after_analytics
)


__all__ = [
    # State management
    'AgentState',
    'ExecutionTrace',
    'create_initial_state',
    'add_trace_entry',
    'get_execution_summary',
    
    # Nodes
    'AgentNodes',
    
    # Graph and execution
    'create_agent_graph',
    'AgentExecutor',
    'should_continue',
    'should_continue_after_retrieval',
    'should_continue_after_analytics',
]


# Version info
__version__ = '0.1.0'
__author__ = 'Sean Clayton'