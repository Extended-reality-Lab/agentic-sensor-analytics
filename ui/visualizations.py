"""
Visualization utilities for creating charts from analytics results.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Dict, Any, List
from datetime import datetime


class VisualizationBuilder:
    """
    Builder class for creating visualizations from analytics results.
    Supports time series, comparisons, distributions, and statistical summaries.
    """
    
    def __init__(self, template: str = "plotly_white", height: int = 400):
        """
        Initialize visualization builder.
        
        Args:
            template: Plotly template to use
            height: Default height for charts
        """
        self.template = template
        self.height = height
    
    def create_time_series(
        self,
        data: List[Dict[str, Any]],
        sensor_type: str,
        location: str,
        unit: str
    ) -> go.Figure:
        """
        Create time series line chart.
        
        Args:
            data: List of {timestamp, value} dictionaries
            sensor_type: Type of sensor data
            location: Location name
            unit: Unit of measurement
            
        Returns:
            Plotly figure
        """
        df = pd.DataFrame(data)
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = px.line(
            df,
            x='timestamp',
            y='value',
            title=f"{sensor_type.title()} Over Time - {location}",
            labels={
                'value': f'{sensor_type.title()} ({unit})',
                'timestamp': 'Time'
            }
        )
        
        fig.update_layout(
            hovermode='x unified',
            template=self.template,
            height=self.height,
            xaxis_title="Time",
            yaxis_title=f"{sensor_type.title()} ({unit})"
        )
        
        fig.update_traces(
            line=dict(width=2),
            hovertemplate='<b>%{x}</b><br>%{y:.2f} ' + unit
        )
        
        return fig
    
    def create_aggregated_series(
        self,
        data: List[Dict[str, Any]],
        sensor_type: str,
        location: str,
        unit: str,
        aggregation_level: str
    ) -> go.Figure:
        """
        Create aggregated time series (daily, hourly, weekly averages).
        
        Args:
            data: List of {period, value} dictionaries
            sensor_type: Type of sensor data
            location: Location name
            unit: Unit of measurement
            aggregation_level: Level of aggregation (hourly, daily, weekly)
            
        Returns:
            Plotly figure
        """
        df = pd.DataFrame(data)
        
        fig = px.bar(
            df,
            x='period',
            y='value',
            title=f"{aggregation_level.title()} Average {sensor_type.title()} - {location}",
            labels={
                'value': f'{sensor_type.title()} ({unit})',
                'period': 'Period'
            }
        )
        
        fig.update_layout(
            template=self.template,
            height=self.height,
            xaxis_title="Period",
            yaxis_title=f"{sensor_type.title()} ({unit})",
            showlegend=False
        )
        
        fig.update_traces(
            hovertemplate='<b>%{x}</b><br>%{y:.2f} ' + unit
        )
        
        return fig
    
    def create_comparison_chart(
        self,
        comparison_data: Dict[str, Dict[str, float]],
        sensor_type: str,
        unit: str,
        metric: str = 'mean'
    ) -> go.Figure:
        """
        Create comparison bar chart across locations.
        
        Args:
            comparison_data: Dict mapping location to {mean, min, max, std, etc.}
            sensor_type: Type of sensor data
            unit: Unit of measurement
            metric: Which metric to compare (mean, min, max)
            
        Returns:
            Plotly figure
        """
        locations = list(comparison_data.keys())
        values = [comparison_data[loc].get(metric, 0) for loc in locations]
        
        fig = px.bar(
            x=locations,
            y=values,
            title=f"{sensor_type.title()} Comparison ({metric.title()})",
            labels={
                'x': 'Location',
                'y': f'{sensor_type.title()} ({unit})'
            }
        )
        
        fig.update_layout(
            template=self.template,
            height=self.height,
            xaxis_title="Location",
            yaxis_title=f"{sensor_type.title()} ({unit})",
            showlegend=False
        )
        
        fig.update_traces(
            hovertemplate='<b>%{x}</b><br>%{y:.2f} ' + unit
        )
        
        return fig
    
    def create_multi_location_time_series(
        self,
        comparison_data: Dict[str, List[Dict[str, Any]]],
        sensor_type: str,
        unit: str
    ) -> go.Figure:
        """
        Create multi-line time series comparing multiple locations.
        
        Args:
            comparison_data: Dict mapping location to list of {timestamp, value}
            sensor_type: Type of sensor data
            unit: Unit of measurement
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        for location, data in comparison_data.items():
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['value'],
                mode='lines',
                name=location,
                hovertemplate=f'<b>{location}</b><br>%{{x}}<br>%{{y:.2f}} {unit}'
            ))
        
        fig.update_layout(
            title=f"{sensor_type.title()} Comparison Over Time",
            xaxis_title="Time",
            yaxis_title=f"{sensor_type.title()} ({unit})",
            template=self.template,
            height=self.height,
            hovermode='x unified'
        )
        
        return fig
    
    def create_statistical_summary(
        self,
        statistics: Dict[str, float],
        sensor_type: str,
        location: str,
        unit: str
    ) -> go.Figure:
        """
        Create box plot style visualization of statistical summary.
        
        Args:
            statistics: Dict with min, max, mean, median, std, etc.
            sensor_type: Type of sensor data
            location: Location name
            unit: Unit of measurement
            
        Returns:
            Plotly figure
        """
        # Create box plot
        fig = go.Figure()
        
        fig.add_trace(go.Box(
            y=[
                statistics.get('min', 0),
                statistics.get('25%', statistics.get('min', 0)),
                statistics.get('median', statistics.get('mean', 0)),
                statistics.get('75%', statistics.get('max', 0)),
                statistics.get('max', 0)
            ],
            name=location,
            boxmean='sd',
            hovertemplate=(
                f'<b>{location}</b><br>' +
                'Min: %{y:.2f} ' + unit + '<br>' +
                'Max: %{y:.2f} ' + unit
            )
        ))
        
        fig.update_layout(
            title=f"Statistical Summary - {sensor_type.title()} at {location}",
            yaxis_title=f"{sensor_type.title()} ({unit})",
            template=self.template,
            height=self.height,
            showlegend=False
        )
        
        return fig
    
    def create_distribution_histogram(
        self,
        data: List[float],
        sensor_type: str,
        location: str,
        unit: str,
        bins: int = 30
    ) -> go.Figure:
        """
        Create histogram showing data distribution.
        
        Args:
            data: List of values
            sensor_type: Type of sensor data
            location: Location name
            unit: Unit of measurement
            bins: Number of histogram bins
            
        Returns:
            Plotly figure
        """
        fig = px.histogram(
            x=data,
            nbins=bins,
            title=f"{sensor_type.title()} Distribution - {location}",
            labels={'x': f'{sensor_type.title()} ({unit})', 'y': 'Frequency'}
        )
        
        fig.update_layout(
            template=self.template,
            height=self.height,
            xaxis_title=f"{sensor_type.title()} ({unit})",
            yaxis_title="Frequency",
            showlegend=False
        )
        
        return fig
    
    def create_heatmap(
        self,
        data: pd.DataFrame,
        sensor_type: str,
        unit: str
    ) -> go.Figure:
        """
        Create heatmap for temporal patterns (hour of day vs day of week).
        
        Args:
            data: DataFrame with timestamp and value columns
            sensor_type: Type of sensor data
            unit: Unit of measurement
            
        Returns:
            Plotly figure
        """
        # Ensure timestamp is datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Extract hour and day of week
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.day_name()
        
        # Pivot for heatmap
        heatmap_data = data.pivot_table(
            values='value',
            index='day_of_week',
            columns='hour',
            aggfunc='mean'
        )
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data = heatmap_data.reindex([d for d in day_order if d in heatmap_data.index])
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='Blues',
            hovertemplate='Day: %{y}<br>Hour: %{x}<br>Average: %{z:.2f} ' + unit
        ))
        
        fig.update_layout(
            title=f"{sensor_type.title()} Patterns by Day and Hour",
            xaxis_title="Hour of Day",
            yaxis_title="Day of Week",
            template=self.template,
            height=self.height
        )
        
        return fig


def create_visualization_from_result(
    analytics_result: Dict[str, Any],
    task_spec,
    builder: Optional[VisualizationBuilder] = None
) -> Optional[go.Figure]:
    """
    Create appropriate visualization based on analytics result and task specification.
    
    Args:
        analytics_result: Result dictionary from analytics execution
        task_spec: TaskSpecification object
        builder: Optional VisualizationBuilder instance
        
    Returns:
        Plotly figure or None if visualization not applicable
    """
    if builder is None:
        builder = VisualizationBuilder()
    
    metadata = analytics_result.get('metadata', {})
    unit = analytics_result.get('unit', 'units')
    
    # Time series
    if 'time_series' in metadata:
        return builder.create_time_series(
            data=metadata['time_series'],
            sensor_type=task_spec.sensor_type,
            location=task_spec.location if isinstance(task_spec.location, str) else task_spec.location[0],
            unit=unit
        )
    
    # Aggregated time series
    elif 'aggregated_data' in metadata:
        return builder.create_aggregated_series(
            data=metadata['aggregated_data'],
            sensor_type=task_spec.sensor_type,
            location=task_spec.location if isinstance(task_spec.location, str) else task_spec.location[0],
            unit=unit,
            aggregation_level=task_spec.aggregation_level or 'daily'
        )
    
    # Spatial comparison
    elif 'comparison_data' in metadata:
        return builder.create_comparison_chart(
            comparison_data=metadata['comparison_data'],
            sensor_type=task_spec.sensor_type,
            unit=unit
        )
    
    # Multi-location time series
    elif 'multi_location_series' in metadata:
        return builder.create_multi_location_time_series(
            comparison_data=metadata['multi_location_series'],
            sensor_type=task_spec.sensor_type,
            unit=unit
        )
    
    # Statistical summary
    elif 'statistics' in metadata:
        location = task_spec.location if isinstance(task_spec.location, str) else task_spec.location[0]
        return builder.create_statistical_summary(
            statistics=metadata['statistics'],
            sensor_type=task_spec.sensor_type,
            location=location,
            unit=unit
        )
    
    # Distribution
    elif 'distribution' in metadata:
        location = task_spec.location if isinstance(task_spec.location, str) else task_spec.location[0]
        return builder.create_distribution_histogram(
            data=metadata['distribution'],
            sensor_type=task_spec.sensor_type,
            location=location,
            unit=unit
        )
    
    return None