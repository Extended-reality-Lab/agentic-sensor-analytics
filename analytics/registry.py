"""
Registry for managing analytics tools.
"""

from typing import Optional, Dict, List
from .base import AnalyticsTool
from .tools import (
    TemporalMeanTool,
    TemporalAggregationTool,
    SpatialComparisonTool,
    StatisticalSummaryTool,
    ThresholdScanTool,
    ResultFilterTool,
)


class ToolRegistry:
    """Central registry for analytics tools."""
    
    def __init__(self):
        # Create single ToolRegistry class
        self._tools: Dict[str, AnalyticsTool] = {}
        self._initialize_tools()
    
    def _initialize_tools(self):
        """Register default tools."""
        self.register(TemporalMeanTool())
        self.register(TemporalAggregationTool())
        self.register(SpatialComparisonTool())
        self.register(StatisticalSummaryTool())
        self.register(ThresholdScanTool())
        self.register(ResultFilterTool())
    
    def register(self, tool: AnalyticsTool) -> None:
        """
        Implement register(tool) method storing tools in dictionary by name.
        
        Args:
            tool: AnalyticsTool instance to register
        """
        self._tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[AnalyticsTool]:
        """
        Implement get_tool(name) to get tool.
        
        Args:
            name: Tool name
            
        Returns:
            AnalyticsTool instance or None
        """
        return self._tools.get(name)
    
    def list_tools(self) -> List[Dict]:
        """
        Implement list_tools() returning tool metadata.
        
        Returns:
            List of tool metadata
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
            for tool in self._tools.values()
        ]
    
    def get_tool_by_operation(self, operation: str) -> Optional[AnalyticsTool]:
        """
        Map operations to tool names: {"mean": "temporal_mean", "max": "temporal_max"}
        
        Args:
            operation: Operation type (mean, max, min, etc.)
            
        Returns:
            AnalyticsTool instance or None
        """
        operation_map = {
            "mean": "temporal_mean",
            "max": "temporal_mean",  # Uses temporal_mean tool with max operation
            "min": "temporal_mean",  # Uses temporal_mean tool with min operation
            "aggregation": "temporal_aggregation",
            "comparison": "spatial_comparison",
            "summary": "statistical_summary"
        }
        
        tool_name = operation_map.get(operation)
        if tool_name:
            return self.get_tool(tool_name)
        return None


# Global registry instance
_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """Get global registry instance."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry