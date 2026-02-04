"""
Streamlit application for Smart Building Analytics.
Provides a chat interface for natural language queries with visualizations.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
from typing import Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import AgentExecutor, get_execution_summary
from llm import SystemContext


# Page configuration
st.set_page_config(
    page_title="Smart Building Analytics",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_sensors_by_node():
    """
    Get mapping of which sensors are available for each node.
    
    Returns:
        Dictionary mapping location names to list of sensor types
    """
    if st.session_state.executor is None:
        return {}
    
    try:
        # Use the repository's built-in method
        repository = st.session_state.executor.bridge.repository
        return repository.get_sensors_by_node()

    except Exception as e:
        print(f"Error getting sensors by node: {e}")
        return {}

def initialize_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'execution_traces' not in st.session_state:
        st.session_state.execution_traces = []
    
    if 'executor' not in st.session_state:
        st.session_state.executor = None
    
    if 'show_trace' not in st.session_state:
        st.session_state.show_trace = False
    
    if 'system_context' not in st.session_state:
        st.session_state.system_context = None


def initialize_executor():
    """Initialize the agent executor."""
    if st.session_state.executor is None:
        with st.spinner("Initializing analytics engine..."):
            try:
                st.session_state.executor = AgentExecutor.from_config()
                
                # Get system context for sidebar
                st.session_state.system_context = st.session_state.executor.bridge.get_system_context()
                
                st.success("✓ Analytics engine initialized")
            except Exception as e:
                st.error(f"Failed to initialize analytics engine: {e}")
                st.stop()


def display_sidebar():
    """Display sidebar with system information and settings."""
    with st.sidebar:
        st.title("🏢 Smart Building Analytics")
        st.markdown("---")
        
        # System status
        st.subheader("System Status")
        if st.session_state.executor:
            st.success("✓ Online")
            
            # Display LLM info
            model_name = st.session_state.executor.llm.model_name
            st.info(f"**Model:** {model_name}")
        else:
            st.warning("⚠ Initializing...")
        
        st.markdown("---")
        
        # System context information
        if st.session_state.system_context:
            st.subheader("Available Data")
            
            context = st.session_state.system_context
            
            # Available sensors
            with st.expander("📊 Sensor Types", expanded=False):
                sensors = context.get('available_sensors', [])
                for sensor in sensors:
                    sensor = sensor.capitalize()
                    st.text(f"• {sensor}")
            
            # Available nodes
            def natural_sort_key(text):
                return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', text)]

            node_sensors = get_sensors_by_node()

            # Locations that exist and have available
            locations = [loc for loc in context.get('available_locations', []) if node_sensors.get(loc)]
            locations = sorted(locations, key=natural_sort_key)
            
            with st.expander("📡 Nodes", expanded=False):
                for location in locations:
                    sensors = node_sensors.get(location)
                    with st.expander(f"🔹 {location}", expanded=False):
                        sensors = node_sensors.get(location, [])
                        for sensor in sensors:
                            st.text(f"  • {sensor.capitalize()}")
            
            # Time range 
            with st.expander("📅 Time Range", expanded=False):
                time_range = context.get('time_range', None)
                if time_range:
                    if isinstance(time_range, tuple) and len(time_range) == 2:
                        start_dt, end_dt = time_range
                        st.text(f"From: {start_dt.strftime('%Y-%m-%d')}")
                        st.text(f"To: {end_dt.strftime('%Y-%m-%d')}")
                    elif isinstance(time_range, dict):
                        st.text(f"From: {time_range.get('start', 'N/A')}")
                        st.text(f"To: {time_range.get('end', 'N/A')}")
                    elif hasattr(time_range, 'start_time') and hasattr(time_range, 'end_time'):
                        st.text(f"From: {time_range.start_time.strftime('%Y-%m-%d')}")
                        st.text(f"To: {time_range.end_time.strftime('%Y-%m-%d')}")
                    else:
                        st.text(f"Time range: {time_range}")
        
        st.markdown("---")
        
        # Settings
        st.subheader("Settings")
        st.session_state.show_trace = st.checkbox(
            "Show execution trace",
            value=st.session_state.show_trace,
            help="Display detailed execution steps for each query"
        )
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation", width='stretch'):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.session_state.execution_traces = []
            st.rerun()
        
        st.markdown("---")
        
        # Example queries
        st.subheader("Example Queries")
        examples = [
            "What was the temperature each day in Node 1 last week?",
            "Compare humidity between Node 15 and Node 25 over the past week.",
            "What was the average strain for Node 35 in June 2025?",
            "What was the average humidity for Node 4 last week?",
            "Provide a summary of moisture levels in Node 15 for the past month.",
        ]
        
        for example in examples:
            if st.button(f"💡 {example[:40]}...", key=example, width='stretch'):
                # Add example to chat input
                st.session_state.example_query = example


def create_visualization(analytics_result: dict, task_spec) -> Optional[go.Figure]:
    """
    Create appropriate visualization based on analytics result.
    
    Args:
        analytics_result: Result from analytics execution
        task_spec: TaskSpecification object
        
    Returns:
        Plotly figure or None
    """
    try:
        metadata = analytics_result.get('metadata', {})
        operation = metadata.get('operation', '')
        
        # Time series visualization
        if 'time_series' in metadata:
            df = pd.DataFrame(metadata['time_series'])
            
            fig = px.line(
                df,
                x='timestamp',
                y='value',
                title=f"{task_spec.sensor_type.title()} Over Time - {task_spec.location}",
                labels={'value': f'{task_spec.sensor_type.title()} ({analytics_result.get("unit", "")})',
                       'timestamp': 'Time'}
            )
            
            fig.update_layout(
                hovermode='x unified',
                template='plotly_white',
                height=400
            )
            
            return fig
        
        # Aggregated time series (daily, hourly, etc.)
        elif 'aggregated_data' in metadata:
            df = pd.DataFrame(metadata['aggregated_data'])
            
            fig = px.bar(
                df,
                x='period',
                y='value',
                title=f"{task_spec.aggregation_level.title()} Average {task_spec.sensor_type.title()} - {task_spec.location}",
                labels={'value': f'{task_spec.sensor_type.title()} ({analytics_result.get("unit", "")})',
                       'period': 'Period'}
            )
            
            fig.update_layout(
                template='plotly_white',
                height=400
            )
            
            return fig
        
        # Spatial comparison
        elif 'comparison_data' in metadata:
            comp_data = metadata['comparison_data']
            
            locations = list(comp_data.keys())
            values = [comp_data[loc]['mean'] for loc in locations]
            
            fig = px.bar(
                x=locations,
                y=values,
                title=f"{task_spec.sensor_type.title()} Comparison",
                labels={'x': 'Location', 'y': f'{task_spec.sensor_type.title()} ({analytics_result.get("unit", "")})'}
            )
            
            fig.update_layout(
                template='plotly_white',
                height=400
            )
            
            return fig
        
        # Statistical summary visualization
        elif operation == 'summary':
            stats = metadata.get('statistics', {})
            
            # Create box plot representation
            fig = go.Figure()
            
            fig.add_trace(go.Box(
                y=[stats.get('min', 0), 
                   stats.get('25%', 0), 
                   stats.get('mean', 0), 
                   stats.get('75%', 0), 
                   stats.get('max', 0)],
                name=task_spec.sensor_type.title(),
                boxmean='sd'
            ))
            
            fig.update_layout(
                title=f"Statistical Summary - {task_spec.sensor_type.title()}",
                yaxis_title=f'{task_spec.sensor_type.title()} ({analytics_result.get("unit", "")})',
                template='plotly_white',
                height=400
            )
            
            return fig
        
    except Exception as e:
        st.warning(f"Could not create visualization: {e}")
        return None
    
    return None


def display_execution_trace(state: dict):
    """Display execution trace in an expander."""
    if not st.session_state.show_trace:
        return
    
    trace = state.get('execution_trace', [])
    
    if not trace:
        return
    
    with st.expander("🔍 Execution Trace", expanded=False):
        summary = get_execution_summary(state)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Steps", summary.get('total_steps', 0))
        with col2:
            st.metric("Completed", summary.get('steps_completed', 0))
        with col3:
            st.metric("Failed", summary.get('steps_failed', 0))
        with col4:
            total_duration = summary.get('total_duration_ms', 0)
            if total_duration is None:
                total_duration = 0
            st.metric("Duration", f"{total_duration:.2f}ms")
        
        st.markdown("---")
        
        # Detailed trace
        for i, entry in enumerate(trace, 1):
            step = entry['step']
            status = entry['status']
            duration = entry.get('duration_ms', 0)

            if duration is None:
                duration = 0
            
            # Status icon
            if status == "completed":
                icon = "✅"
                color = "green"
            elif status == "failed":
                icon = "❌"
                color = "red"
            else:
                icon = "⏳"
                color = "orange"
            
            # Display step
            st.markdown(f"**{i}. {icon} {step}** ({status}) - {duration:.2f}ms")
            
            # Display details if available
            details = entry.get('details', {})
            if details:
                with st.container():
                    for key, value in details.items():
                        if key != 'error':  # Handle errors separately
                            st.text(f"  {key}: {value}")
                    
                    # Show error if present
                    if 'error' in details:
                        st.error(f"  Error: {details['error']}")


def display_message(message: dict, message_index: int):
    """Display a chat message with appropriate formatting."""
    role = message['role']
    content = message['content']
    
    with st.chat_message(role):
        st.markdown(content)
        
        # Display visualization if present
        if 'visualization' in message:
            st.plotly_chart(message['visualization'], width='stretch', key=f"viz_{message_index}")
        
        # Display execution trace if present
        if 'state' in message:
            display_execution_trace(message['state'])


def process_query(query: str):
    """Process user query and update chat."""
    # Add user message
    st.session_state.messages.append({
        'role': 'user',
        'content': query
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    # Process with agent
    with st.chat_message("assistant"):
        with st.spinner("Processing query..."):
            try:
                # Execute query
                result = st.session_state.executor.execute(query)
                
                # Store in history
                st.session_state.conversation_history.append(result)
                st.session_state.execution_traces.append(result.get('execution_trace', []))
                
                # Determine response content
                if result.get('success'):
                    response_content = result.get('explanation', 'Query processed successfully.')
                    
                    # Create visualization if analytics result available
                    visualization = None
                    if result.get('analytics_result') and result.get('task_spec'):
                        visualization = create_visualization(
                            result['analytics_result'],
                            result['task_spec']
                        )
                    
                    # Display explanation
                    st.markdown(response_content)
                    
                    # Display visualization
                    if visualization:
                        st.plotly_chart(visualization, width='stretch', key=f"viz_current")
                    
                    # Add to message history
                    message = {
                        'role': 'assistant',
                        'content': response_content,
                        'state': result
                    }
                    if visualization:
                        message['visualization'] = visualization
                    
                    st.session_state.messages.append(message)
                    
                    # Display trace
                    display_execution_trace(result)
                
                else:
                    # Error occurred
                    error_content = result.get('error_explanation', 'An error occurred processing your query.')
                    st.error(error_content)
                    
                    # Add to message history
                    st.session_state.messages.append({
                        'role': 'assistant',
                        'content': error_content,
                        'state': result
                    })
                    
                    # Display trace
                    display_execution_trace(result)
                
            except Exception as e:
                error_msg = f"An unexpected error occurred: {str(e)}"
                st.error(error_msg)
                
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': error_msg
                })


def main():
    """Main application function."""
    # Initialize
    initialize_session_state()
    initialize_executor()
    display_sidebar()
    
    # Main content area
    st.title("Smart Building Analytics")
    st.markdown("Ask questions about Peavy Hall sensor data in natural language.")
    st.markdown("---")
    
    # Display conversation history
    for idx, message in enumerate(st.session_state.messages):
        display_message(message, idx)
    
    # Chat input
    query = st.chat_input("Ask about Peavy Hall sensor data...")
    
    # Handle example query from sidebar
    if 'example_query' in st.session_state:
        query = st.session_state.example_query
        del st.session_state.example_query
    
    # Process query
    if query:
        process_query(query)
        st.rerun()


if __name__ == "__main__":
    main()