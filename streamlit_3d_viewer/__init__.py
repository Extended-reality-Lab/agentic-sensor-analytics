import streamlit.components.v1 as components
import os

_RELEASE = False

if not _RELEASE:
    _component_func = components.declare_component(
        "building_3d_viewer",
        url="http://localhost:3000",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component(
        "building_3d_viewer",
        path=build_dir
    )

def building_3d_viewer(
    node_positions,
    model_url=None,
    active_node=None,
    height=600,
    key=None
):
    """
    Display interactive 3D building visualization.
    
    Parameters
    ----------
    node_positions : dict
        Dictionary mapping node IDs to {x, y, z, floor, room, sensor_types}
    model_url : str, optional
        Path to .glb building model (if None, uses procedural geometry)
    active_node : str, optional
        Currently selected node ID to highlight
    height : int
        Component height in pixels
    key : str, optional
        Streamlit component key
        
    Returns
    -------
    str or None
        Selected node ID when user clicks, None otherwise
    """
    component_value = _component_func(
        nodePositions=node_positions,
        modelUrl=model_url,
        activeNode=active_node,
        height=height,
        key=key,
        default=None
    )
    return component_value