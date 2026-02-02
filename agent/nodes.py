"""
Agent graph nodes for LangGraph workflow.
Each node is a function that transforms the agent state.
"""

import time
from typing import Dict, Any
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from llm import OllamaLLM, SystemContext, LLMError
from data import SensorDataRepository, LLMDataBridge, RepositoryError
from analytics import get_registry

from .state import AgentState, add_trace_entry


class AgentNodes:
    """
    Collection of node functions for the agent graph.
    Each node transforms the agent state.
    """
    
    def __init__(
        self,
        llm: OllamaLLM,
        repository: SensorDataRepository,
        bridge: LLMDataBridge
    ):
        """
        Initialize agent nodes with required components.
        
        Args:
            llm: LLM interface for intent extraction and explanation
            repository: Data repository for validation
            bridge: Bridge between LLM and data layer
        """
        self.llm = llm
        self.repository = repository
        self.bridge = bridge
        self.registry = get_registry()
    
    def interpret_query(self, state: AgentState) -> AgentState:
        """
        Node 1: Extract structured task specification from user query.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with task_spec
        """
        start_time = time.time()
        add_trace_entry(state, 'interpret_query', 'started', {
            'user_query': state['user_query']
        })
        
        try:
            # Get system context for validation
            context_dict = self.bridge.get_system_context()
            context = SystemContext(**context_dict)
            
            # Extract intent using LLM
            task_spec = self.llm.extract_intent(state['user_query'], context)
            
            # Update state
            state['task_spec'] = task_spec
            
            duration_ms = (time.time() - start_time) * 1000
            add_trace_entry(state, 'interpret_query', 'completed', {
                'intent_type': task_spec.intent_type,
                'sensor_type': task_spec.sensor_type,
                'location': task_spec.location,
                'operation': task_spec.operation,
                'confidence': task_spec.confidence
            }, duration_ms)
            
        except LLMError as e:
            duration_ms = (time.time() - start_time) * 1000
            add_trace_entry(state, 'interpret_query', 'failed', {
                'error': str(e)
            }, duration_ms)
            
            state['validation_errors'] = [f"Failed to understand query: {e}"]
            state['success'] = False
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            add_trace_entry(state, 'interpret_query', 'failed', {
                'error': str(e),
                'error_type': type(e).__name__
            }, duration_ms)
            
            state['validation_errors'] = [f"Unexpected error during query interpretation: {e}"]
            state['success'] = False
        
        return state
    
    def validate_task(self, state: AgentState) -> AgentState:
        """
        Node 2: Validate task specification against available data.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with validation_errors
        """
        start_time = time.time()
        add_trace_entry(state, 'validate_task', 'started', {
            'task_spec': state['task_spec'].model_dump() if state.get('task_spec') else None
        })
        
        try:
            task_spec = state.get('task_spec')
            
            if not task_spec:
                state['validation_errors'] = ["No task specification to validate"]
                add_trace_entry(state, 'validate_task', 'failed', {
                    'error': 'Missing task specification'
                })
                return state
            
            # Validate using bridge
            errors = self.bridge.validate_task(task_spec)
            
            state['validation_errors'] = errors
            
            duration_ms = (time.time() - start_time) * 1000
            
            if errors:
                add_trace_entry(state, 'validate_task', 'failed', {
                    'errors': errors,
                    'num_errors': len(errors)
                }, duration_ms)
            else:
                add_trace_entry(state, 'validate_task', 'completed', {
                    'message': 'Task specification is valid'
                }, duration_ms)
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            add_trace_entry(state, 'validate_task', 'failed', {
                'error': str(e),
                'error_type': type(e).__name__
            }, duration_ms)
            
            state['validation_errors'] = [f"Validation error: {e}"]
        
        return state
    
    def retrieve_data(self, state: AgentState) -> AgentState:
        """
        Node 3: Retrieve sensor data based on task specification.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with data DataFrame
        """
        start_time = time.time()
        add_trace_entry(state, 'retrieve_data', 'started', {
            'task_spec': state['task_spec'].model_dump() if state.get('task_spec') else None
        })
        
        try:
            task_spec = state['task_spec']
            
            if not task_spec:
                raise ValueError("No task specification available")
            
            # Execute task to get data
            data = self.bridge.execute_task(task_spec)
            
            # Check if DataFrame is empty
            if data.empty:
                error_msg = f"No data found for {task_spec.sensor_type} in {task_spec.location}"
                state['validation_errors'] = [error_msg]
                
                duration_ms = (time.time() - start_time) * 1000
                add_trace_entry(state, 'retrieve_data', 'failed', {
                    'error': error_msg,
                    'rows_retrieved': 0
                }, duration_ms)
            else:
                state['data'] = data
                
                duration_ms = (time.time() - start_time) * 1000
                add_trace_entry(state, 'retrieve_data', 'completed', {
                    'rows_retrieved': len(data),
                    'columns': list(data.columns),
                    'time_range': f"{data['timestamp'].min()} to {data['timestamp'].max()}" if 'timestamp' in data.columns else None
                }, duration_ms)
        
        except RepositoryError as e:
            duration_ms = (time.time() - start_time) * 1000
            add_trace_entry(state, 'retrieve_data', 'failed', {
                'error': str(e)
            }, duration_ms)
            
            state['validation_errors'] = [f"Data retrieval failed: {e}"]
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            add_trace_entry(state, 'retrieve_data', 'failed', {
                'error': str(e),
                'error_type': type(e).__name__
            }, duration_ms)
            
            state['validation_errors'] = [f"Unexpected error during data retrieval: {e}"]
        
        return state
    
    def execute_analytics(self, state: AgentState) -> AgentState:
        """
        Node 4: Execute analytics tool on retrieved data.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with analytics_result
        """
        start_time = time.time()
        add_trace_entry(state, 'execute_analytics', 'started', {
            'operation': state['task_spec'].operation if state.get('task_spec') else None
        })
        
        try:
            task_spec = state['task_spec']
            data = state.get('data')
            
            if data is None:
                raise ValueError("No data available for analytics")
            
            if task_spec is None:
                raise ValueError("No task specification available")
            
            # Get appropriate tool from registry
            if task_spec.intent_type.value == 'aggregation':
                tool = self.registry.get_tool('temporal_aggregation')
            elif task_spec.intent_type.value == 'comparison':
                tool = self.registry.get_tool('spatial_comparison')
            elif task_spec.operation.value == 'summary':
                tool = self.registry.get_tool('statistical_summary')
            else:
                tool = self.registry.get_tool('temporal_mean')
            
            if tool is None:
                raise ValueError(f"No tool found for operation: {task_spec.operation}")
            
            # Build kwargs for tool execution
            kwargs = {'operation': task_spec.operation.value}
            
            # Add aggregation level if present
            if task_spec.aggregation_level:
                kwargs['aggregation_level'] = task_spec.aggregation_level
            
            # Execute tool
            result = tool.execute(data, **kwargs)
            
            # Check if execution was successful
            if not result.success:
                error_msg = result.error_message or "Analytics execution failed"
                state['validation_errors'] = [error_msg]
                
                duration_ms = (time.time() - start_time) * 1000
                add_trace_entry(state, 'execute_analytics', 'failed', {
                    'error': error_msg,
                    'tool': tool.name
                }, duration_ms)
            else:
                state['analytics_result'] = result.model_dump()
                
                duration_ms = (time.time() - start_time) * 1000
                add_trace_entry(state, 'execute_analytics', 'completed', {
                    'tool': tool.name,
                    'operation': task_spec.operation.value,
                    'result_value': result.value,
                    'result_unit': result.unit,
                    'tool_execution_time_ms': result.execution_time_ms
                }, duration_ms)
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            add_trace_entry(state, 'execute_analytics', 'failed', {
                'error': str(e),
                'error_type': type(e).__name__
            }, duration_ms)
            
            state['validation_errors'] = [f"Analytics execution error: {e}"]
        
        return state
    
    def generate_explanation(self, state: AgentState) -> AgentState:
        """
        Node 5: Generate natural language explanation of results.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with explanation
        """
        start_time = time.time()
        add_trace_entry(state, 'generate_explanation', 'started', {})
        
        try:
            task_spec = state['task_spec']
            analytics_result = state.get('analytics_result')
            
            if not analytics_result:
                raise ValueError("No analytics result to explain")
            
            # Create a cleaned summary for the LLM (remove verbose data like time_series)
            result_summary = {
                'value': analytics_result.get('value'),
                'unit': analytics_result.get('unit'),
                'success': analytics_result.get('success'),
                'execution_time_ms': analytics_result.get('execution_time_ms'),
                'metadata': {}
            }
            
            # Add only essential metadata (not the full time_series/aggregated_data)
            original_metadata = analytics_result.get('metadata', {})
            essential_keys = ['operation', 'sample_size', 'std_dev', 'min', 'max', 
                            'num_periods', 'overall_aggregate', 'aggregation_level',
                            'num_locations', 'count', 'mean', 'median', 'skewness', 'kurtosis']
            
            for key in essential_keys:
                if key in original_metadata:
                    result_summary['metadata'][key] = original_metadata[key]
            
            # Generate explanation using LLM with cleaned summary
            explanation = self.llm.explain_results(
                state['user_query'],
                task_spec,
                [result_summary]
            )
            
            state['explanation'] = explanation
            state['success'] = True
            state['end_time'] = datetime.now()
            
            duration_ms = (time.time() - start_time) * 1000
            add_trace_entry(state, 'generate_explanation', 'completed', {
                'explanation_length': len(explanation)
            }, duration_ms)
        
        except LLMError as e:
            duration_ms = (time.time() - start_time) * 1000
            add_trace_entry(state, 'generate_explanation', 'failed', {
                'error': str(e)
            }, duration_ms)
            
            # Fallback to simple explanation
            analytics_result = state.get('analytics_result', {})
            state['explanation'] = (
                f"Result: {analytics_result.get('value')} "
                f"{analytics_result.get('unit', '')}"
            )
            state['success'] = True
            state['end_time'] = datetime.now()
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            add_trace_entry(state, 'generate_explanation', 'failed', {
                'error': str(e),
                'error_type': type(e).__name__
            }, duration_ms)
            
            # Still mark as success if we have analytics result
            if state.get('analytics_result'):
                analytics_result = state['analytics_result']
                state['explanation'] = (
                    f"Result: {analytics_result.get('value')} "
                    f"{analytics_result.get('unit', '')}"
                )
                state['success'] = True
            else:
                state['success'] = False
            
            state['end_time'] = datetime.now()
        
        return state
    
    def handle_error(self, state: AgentState) -> AgentState:
        """
        Node 6: Generate user-friendly error explanation.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with error_explanation
        """
        start_time = time.time()
        add_trace_entry(state, 'handle_error', 'started', {
            'num_errors': len(state.get('validation_errors', []))
        })
        
        try:
            errors = state.get('validation_errors', [])
            
            if not errors:
                errors = ["An unknown error occurred"]
            
            # Generate user-friendly error explanation using LLM
            try:
                error_explanation = self.llm.explain_error(
                    state['user_query'],
                    errors
                )
            except LLMError:
                # Fallback to simple formatting
                error_explanation = (
                    "I encountered some issues with your query:\n" +
                    "\n".join(f"• {error}" for error in errors)
                )
            
            state['error_explanation'] = error_explanation
            state['success'] = False
            state['end_time'] = datetime.now()
            
            duration_ms = (time.time() - start_time) * 1000
            add_trace_entry(state, 'handle_error', 'completed', {
                'errors_handled': len(errors)
            }, duration_ms)
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            add_trace_entry(state, 'handle_error', 'failed', {
                'error': str(e),
                'error_type': type(e).__name__
            }, duration_ms)
            
            # Absolute fallback
            state['error_explanation'] = "An error occurred processing your query."
            state['success'] = False
            state['end_time'] = datetime.now()
        
        return state