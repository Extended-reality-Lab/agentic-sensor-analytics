"""
Export utilities for saving analytics results and visualizations.
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import plotly.graph_objects as go


class ResultExporter:
    """
    Handles exporting analytics results to various formats.
    """
    
    def __init__(self, output_dir: Optional[str | Path] = None):
        """
        Initialize exporter.
        
        Args:
            output_dir: Directory for exports (default: ./exports)
        """
        self.output_dir = Path(output_dir) if output_dir else Path('./exports')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_to_json(
        self,
        state: Dict[str, Any],
        filename: Optional[str] = None
    ) -> Path:
        """
        Export complete agent state to JSON.
        
        Args:
            state: Agent state dictionary
            filename: Optional filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"analytics_result_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Prepare state for JSON serialization
        serializable_state = self._make_serializable(state)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_state, f, indent=2)
        
        return filepath
    
    def export_to_csv(
        self,
        data: pd.DataFrame,
        filename: Optional[str] = None
    ) -> Path:
        """
        Export DataFrame to CSV.
        
        Args:
            data: DataFrame to export
            filename: Optional filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"sensor_data_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        data.to_csv(filepath, index=False)
        
        return filepath
    
    def export_visualization(
        self,
        fig: go.Figure,
        filename: Optional[str] = None,
        format: str = 'html'
    ) -> Path:
        """
        Export visualization to file.
        
        Args:
            fig: Plotly figure
            filename: Optional filename (auto-generated if None)
            format: Export format ('html', 'png', 'svg', 'pdf')
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"visualization_{timestamp}.{format}"
        
        filepath = self.output_dir / filename
        
        if format == 'html':
            fig.write_html(str(filepath))
        elif format == 'png':
            fig.write_image(str(filepath), format='png')
        elif format == 'svg':
            fig.write_image(str(filepath), format='svg')
        elif format == 'pdf':
            fig.write_image(str(filepath), format='pdf')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return filepath
    
    def export_execution_trace(
        self,
        trace: list,
        filename: Optional[str] = None
    ) -> Path:
        """
        Export execution trace to JSON.
        
        Args:
            trace: Execution trace list
            filename: Optional filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"execution_trace_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        serializable_trace = self._make_serializable(trace)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_trace, f, indent=2)
        
        return filepath
    
    def export_conversation(
        self,
        messages: list,
        filename: Optional[str] = None
    ) -> Path:
        """
        Export conversation history to JSON.
        
        Args:
            messages: List of message dictionaries
            filename: Optional filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"conversation_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Remove non-serializable content
        serializable_messages = []
        for msg in messages:
            clean_msg = {
                'role': msg.get('role'),
                'content': msg.get('content'),
                'timestamp': datetime.now().isoformat()
            }
            serializable_messages.append(clean_msg)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_messages, f, indent=2)
        
        return filepath
    
    def create_report(
        self,
        state: Dict[str, Any],
        include_visualization: bool = True
    ) -> Path:
        """
        Create comprehensive HTML report of analytics results.
        
        Args:
            state: Agent state dictionary
            include_visualization: Whether to include visualizations
            
        Returns:
            Path to report file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"analytics_report_{timestamp}.html"
        filepath = self.output_dir / filename
        
        # Build HTML report
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>Smart Building Analytics Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 40px; }",
            "h1 { color: #1f77b4; }",
            "h2 { color: #2ca02c; }",
            ".section { margin: 20px 0; padding: 20px; background: #f9f9f9; border-radius: 5px; }",
            ".metadata { color: #666; font-size: 0.9em; }",
            "table { border-collapse: collapse; width: 100%; margin: 10px 0; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #1f77b4; color: white; }",
            ".success { color: #2ca02c; }",
            ".error { color: #d62728; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>Smart Building Analytics Report</h1>",
            f"<p class='metadata'>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        ]
        
        # Query section
        html_parts.append("<div class='section'>")
        html_parts.append("<h2>Query</h2>")
        html_parts.append(f"<p>{state.get('user_query', 'N/A')}</p>")
        html_parts.append("</div>")
        
        # Task specification section
        if state.get('task_spec'):
            task_spec = state['task_spec']
            html_parts.append("<div class='section'>")
            html_parts.append("<h2>Task Specification</h2>")
            html_parts.append("<table>")
            html_parts.append(f"<tr><th>Intent Type</th><td>{task_spec.intent_type}</td></tr>")
            html_parts.append(f"<tr><th>Sensor Type</th><td>{task_spec.sensor_type}</td></tr>")
            html_parts.append(f"<tr><th>Location</th><td>{task_spec.location}</td></tr>")
            html_parts.append(f"<tr><th>Operation</th><td>{task_spec.operation}</td></tr>")
            html_parts.append(f"<tr><th>Time Range</th><td>{task_spec.start_time} to {task_spec.end_time}</td></tr>")
            html_parts.append("</table>")
            html_parts.append("</div>")
        
        # Results section
        success = state.get('success', False)
        status_class = "success" if success else "error"
        
        html_parts.append("<div class='section'>")
        html_parts.append(f"<h2 class='{status_class}'>Results</h2>")
        
        if success:
            explanation = state.get('explanation', 'No explanation available')
            html_parts.append(f"<p>{explanation}</p>")
            
            # Analytics results
            if state.get('analytics_result'):
                analytics = state['analytics_result']
                html_parts.append("<h3>Analytics Details</h3>")
                html_parts.append("<table>")
                html_parts.append(f"<tr><th>Value</th><td>{analytics.get('value', 'N/A')}</td></tr>")
                html_parts.append(f"<tr><th>Unit</th><td>{analytics.get('unit', 'N/A')}</td></tr>")
                
                # Metadata
                metadata = analytics.get('metadata', {})
                for key, value in metadata.items():
                    if not isinstance(value, (dict, list)):
                        html_parts.append(f"<tr><th>{key}</th><td>{value}</td></tr>")
                
                html_parts.append("</table>")
        else:
            error = state.get('error_explanation', 'Unknown error')
            html_parts.append(f"<p class='error'>{error}</p>")
        
        html_parts.append("</div>")
        
        # Execution trace section
        if state.get('execution_trace'):
            html_parts.append("<div class='section'>")
            html_parts.append("<h2>Execution Trace</h2>")
            html_parts.append("<table>")
            html_parts.append("<tr><th>Step</th><th>Status</th><th>Duration (ms)</th></tr>")
            
            for entry in state['execution_trace']:
                step = entry.get('step', 'Unknown')
                status = entry.get('status', 'Unknown')
                duration = entry.get('duration_ms', 0)
                html_parts.append(f"<tr><td>{step}</td><td>{status}</td><td>{duration:.2f}</td></tr>")
            
            html_parts.append("</table>")
            html_parts.append("</div>")
        
        html_parts.extend(["</body>", "</html>"])
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(html_parts))
        
        return filepath
    
    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert object to JSON-serializable format.
        
        Args:
            obj: Object to convert
            
        Returns:
            Serializable version
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj