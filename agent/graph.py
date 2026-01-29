"""
Agent graph construction using LangGraph.
Defines the workflow graph with conditional routing.
"""

from typing import Literal
from langgraph.graph import StateGraph, END
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from llm import OllamaLLM
from data import SensorDataRepository, LLMDataBridge

from .state import AgentState
from .nodes import AgentNodes


def should_continue(state: AgentState) -> Literal["retrieve_data", "handle_error"]:
    """
    Conditional routing after validation.
    Routes to data retrieval if no errors, otherwise to error handling.
    
    Args:
        state: Current agent state
        
    Returns:
        Next node name: 'retrieve_data' or 'handle_error'
    """
    validation_errors = state.get('validation_errors', [])
    
    if validation_errors:
        return "handle_error"
    else:
        return "retrieve_data"


def should_continue_after_retrieval(state: AgentState) -> Literal["execute_analytics", "handle_error"]:
    """
    Conditional routing after data retrieval.
    Routes to analytics if data retrieved, otherwise to error handling.
    
    Args:
        state: Current agent state
        
    Returns:
        Next node name: 'execute_analytics' or 'handle_error'
    """
    validation_errors = state.get('validation_errors', [])
    
    # Check if data retrieval failed
    if validation_errors:
        return "handle_error"
    
    # Check if we have data
    if state.get('data') is None:
        return "handle_error"
    
    return "execute_analytics"


def should_continue_after_analytics(state: AgentState) -> Literal["generate_explanation", "handle_error"]:
    """
    Conditional routing after analytics execution.
    Routes to explanation if successful, otherwise to error handling.
    
    Args:
        state: Current agent state
        
    Returns:
        Next node name: 'generate_explanation' or 'handle_error'
    """
    validation_errors = state.get('validation_errors', [])
    
    # Check if analytics failed
    if validation_errors:
        return "handle_error"
    
    # Check if we have analytics result
    if state.get('analytics_result') is None:
        return "handle_error"
    
    return "generate_explanation"


def create_agent_graph(
    llm: OllamaLLM,
    repository: SensorDataRepository,
    bridge: LLMDataBridge
) -> StateGraph:
    """
    Create and compile the agent workflow graph.
    
    Args:
        llm: LLM interface for intent extraction and explanation
        repository: Data repository for validation
        bridge: Bridge between LLM and data layer
        
    Returns:
        Compiled StateGraph ready for execution
        
    Graph structure:
        START → interpret_query → validate_task → [conditional routing]
        
        If validation passes:
            → retrieve_data → execute_analytics → generate_explanation → END
        
        If validation fails or errors occur:
            → handle_error → END
    """
    # Initialize nodes
    nodes = AgentNodes(llm, repository, bridge)
    
    # Create graph with AgentState
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("interpret_query", nodes.interpret_query)
    workflow.add_node("validate_task", nodes.validate_task)
    workflow.add_node("retrieve_data", nodes.retrieve_data)
    workflow.add_node("execute_analytics", nodes.execute_analytics)
    workflow.add_node("generate_explanation", nodes.generate_explanation)
    workflow.add_node("handle_error", nodes.handle_error)
    
    # Set entry point
    workflow.set_entry_point("interpret_query")
    
    # Add edges
    # interpret_query always goes to validate_task
    workflow.add_edge("interpret_query", "validate_task")
    
    # validate_task has conditional routing
    workflow.add_conditional_edges(
        "validate_task",
        should_continue,
        {
            "retrieve_data": "retrieve_data",
            "handle_error": "handle_error"
        }
    )
    
    # retrieve_data has conditional routing
    workflow.add_conditional_edges(
        "retrieve_data",
        should_continue_after_retrieval,
        {
            "execute_analytics": "execute_analytics",
            "handle_error": "handle_error"
        }
    )
    
    # execute_analytics has conditional routing
    workflow.add_conditional_edges(
        "execute_analytics",
        should_continue_after_analytics,
        {
            "generate_explanation": "generate_explanation",
            "handle_error": "handle_error"
        }
    )
    
    # generate_explanation and handle_error both go to END
    workflow.add_edge("generate_explanation", END)
    workflow.add_edge("handle_error", END)
    
    # Compile the graph
    return workflow.compile()


class AgentExecutor:
    """
    High-level executor for the agent workflow.
    Provides a simple interface for running queries.
    """
    
    def __init__(
        self,
        llm: OllamaLLM,
        repository: SensorDataRepository,
        bridge: LLMDataBridge
    ):
        """
        Initialize agent executor.
        
        Args:
            llm: LLM interface
            repository: Data repository
            bridge: Data bridge
        """
        self.llm = llm
        self.repository = repository
        self.bridge = bridge
        self.graph = create_agent_graph(llm, repository, bridge)
    
    def execute(self, user_query: str) -> AgentState:
        """
        Execute a user query through the agent workflow.
        
        Args:
            user_query: Natural language query from user
            
        Returns:
            Final agent state with results or errors
            
        Example:
            executor = AgentExecutor.from_config()
            result = executor.execute("What was the average temperature in Node 15 yesterday?")
            
            if result['success']:
                print(result['explanation'])
            else:
                print(result['error_explanation'])
        """
        from .state import create_initial_state
        
        # Create initial state
        initial_state = create_initial_state(user_query)
        
        # Ensure repository is connected
        if not self.repository.api_client.authenticated:
            self.repository.connect()
        
        # Execute graph
        final_state = self.graph.invoke(initial_state)
        
        return final_state
    
    @classmethod
    def from_config(cls, llm_config_path=None, data_config_path=None) -> 'AgentExecutor':
        """
        Create agent executor from configuration files.
        
        Args:
            llm_config_path: Path to LLM configuration file (optional)
            data_config_path: Path to data configuration file (optional)
            
        Returns:
            Configured AgentExecutor instance
            
        Example:
            executor = AgentExecutor.from_config()
            result = executor.execute("Show me the temperature data")
        """
        llm = OllamaLLM.from_config(llm_config_path)
        repository = SensorDataRepository.from_config(data_config_path)
        bridge = LLMDataBridge(repository)
        
        return cls(llm, repository, bridge)
    
    def get_execution_trace(self, state: AgentState) -> list:
        """
        Get formatted execution trace from state.
        
        Args:
            state: Agent state with execution trace
            
        Returns:
            List of trace entries
        """
        return state.get('execution_trace', [])
    
    def visualize_trace(self, state: AgentState) -> str:
        """
        Create a text visualization of the execution trace.
        
        Args:
            state: Agent state with execution trace
            
        Returns:
            Formatted string showing execution flow
        """
        trace = state.get('execution_trace', [])
        
        if not trace:
            return "No execution trace available"
        
        lines = ["Execution Trace:", "=" * 60]
        
        for i, entry in enumerate(trace, 1):
            step = entry['step']
            status = entry['status']
            duration = entry.get('duration_ms', 0)
            
            status_symbol = "✓" if status == "completed" else "✗" if status == "failed" else "→"
            
            duration_str = f"{duration:.2f}ms" if duration is not None else "N/A"
            lines.append(f"{i}. {status_symbol} {step} ({status}) - {duration_str}")
            
            # Add details for failed steps
            if status == "failed" and 'error' in entry.get('details', {}):
                lines.append(f"   Error: {entry['details']['error']}")
        
        lines.append("=" * 60)
        
        from .state import get_execution_summary
        summary = get_execution_summary(state)
        
        lines.append(f"Total Duration: {summary.get('total_duration_ms', 0):.2f}ms")
        lines.append(f"Steps Completed: {summary.get('steps_completed', 0)}")
        lines.append(f"Steps Failed: {summary.get('steps_failed', 0)}")
        lines.append(f"Success: {summary.get('success', False)}")
        
        return "\n".join(lines)