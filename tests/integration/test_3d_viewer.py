import streamlit as st
import json
import sys
import os
from pathlib import Path

# Add component to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from streamlit_3d_viewer import building_3d_viewer

st.set_page_config(layout="wide", page_title="3D Building Viewer Test")

st.title("🏢 Peavy Hall 3D Sensor Visualization")

# Load node positions
with open('config/node_positions.json', 'r') as f:
    node_positions = json.load(f)

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    
    show_only_floor = st.selectbox(
        "Filter by Floor",
        options=["All Floors", "Floor 1", "Floor 2", "Floor 3"]
    )
    
    if st.button("Reset View"):
        st.session_state.active_node = None
        st.rerun()

# Filter nodes by floor if selected
filtered_nodes = node_positions.copy()
if show_only_floor != "All Floors":
    floor_num = int(show_only_floor.split()[-1])
    filtered_nodes = {
        k: v for k, v in node_positions.items() 
        if v['floor'] == floor_num
    }

# Display 3D viewer
st.subheader("Interactive Building View")
st.info("💡 Click on any sensor node to select it")

selected_node = building_3d_viewer(
    node_positions=filtered_nodes,
    active_node=st.session_state.get('active_node'),
    height=600,
    key="building_viewer"
)

# Handle node selection
if selected_node:
    st.session_state.active_node = selected_node
    st.rerun()

# Display selected node info
if st.session_state.get('active_node'):
    node_id = st.session_state.active_node
    node_data = node_positions.get(node_id, {})
    
    st.success(f"✅ Selected: **{node_id}**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Room", node_data.get('room', 'Unknown'))
    with col2:
        st.metric("Floor", node_data.get('floor', 'Unknown'))
    with col3:
        st.metric("Sensors", len(node_data.get('sensor_types', [])))
    
    st.write("**Available Sensors:**", ", ".join(node_data.get('sensor_types', [])))
    
    # Auto-populate query
    if st.button("Query This Node"):
        st.session_state.example_query = f"Show me temperature data for {node_id} yesterday"
        st.info(f"Query populated: {st.session_state.example_query}")