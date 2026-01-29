"""
Core analytics tool implementations - FIXED FOR VISUALIZATIONS.
"""

import time
import pandas as pd
from scipy import stats

from .base import AnalyticsTool, AnalyticsResult


class TemporalMeanTool(AnalyticsTool):
    """Calculate mean value over time range."""
    
    def __init__(self):
        super().__init__()
        self.name = "temporal_mean"
        self.description = "Calculate mean, min, or max value"
        self.parameters = ["operation"]

    def execute(self, data: pd.DataFrame, **kwargs) -> AnalyticsResult:
        start_time = time.time()
        
        # Validate DataFrame has required columns: timestamp, value, unit
        required_cols = {'timestamp', 'value', 'unit'}
        if not required_cols.issubset(data.columns):
            return AnalyticsResult(
                value=None,
                unit=None,
                metadata={},
                success=False,
                error_message=f"Missing required columns: {required_cols - set(data.columns)}",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Get operation from kwargs (default to 'mean')
        operation = kwargs.get('operation', 'mean')
        
        # Calculate based on operation
        if operation == 'mean':
            result_value = data['value'].mean()
        elif operation == 'min':
            result_value = data['value'].min()
        elif operation == 'max':
            result_value = data['value'].max()
        else:
            return AnalyticsResult(
                value=None,
                unit=None,
                metadata={},
                success=False,
                error_message=f"Invalid operation: {operation}",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Prepare time series data for visualization
        time_series = [
            {
                'timestamp': row['timestamp'].isoformat(),
                'value': float(row['value'])
            }
            for _, row in data.iterrows()
        ]
        
        # Compute metadata: std dev, min, max, sample size
        metadata = {
            "operation": operation,
            "std_dev": float(data['value'].std()),
            "min": float(data['value'].min()),
            "max": float(data['value'].max()),
            "sample_size": len(data),
            "time_series": time_series
        }
        
        # Return AnalyticsResult with statistics
        return AnalyticsResult(
            value=float(result_value),
            unit=data['unit'].iloc[0],
            metadata=metadata,
            success=True,
            execution_time_ms=(time.time() - start_time) * 1000
        )


class TemporalAggregationTool(AnalyticsTool):
    """Aggregate data by time periods."""
    
    def __init__(self):
        super().__init__()
        self.name = "temporal_aggregation"
        self.description = "Aggregate data by hourly/daily/weekly periods"
        self.parameters = ["aggregation_level", "operation"]
    
    def execute(self, data: pd.DataFrame, **kwargs) -> AnalyticsResult:
        start_time = time.time()
        
        # Validate required columns
        required_cols = {'timestamp', 'value', 'unit'}
        if not required_cols.issubset(data.columns):
            return AnalyticsResult(
                value=None,
                unit=None,
                metadata={},
                success=False,
                error_message=f"Missing required columns: {required_cols - set(data.columns)}",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Accept aggregation_level: "hourly", "daily", or "weekly"
        aggregation_level = kwargs.get('aggregation_level')
        operation = kwargs.get('operation', 'mean')
        
        freq_map = {
            'hourly': 'h',
            'daily': 'D',
            'weekly': 'W'
        }
        
        if aggregation_level not in freq_map:
            return AnalyticsResult(
                value=None,
                unit=None,
                metadata={},
                success=False,
                error_message=f"Invalid aggregation_level: {aggregation_level}",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Use pandas resample() method with appropriate frequency
        data_indexed = data.set_index('timestamp').sort_index()
        
        # Apply aggregation function to each time group
        aggregated = data_indexed['value'].resample(freq_map[aggregation_level]).agg(operation)
        
        # Return DataFrame with aggregated values
        result_data = [
            {"timestamp": ts.isoformat(), "value": float(val)}
            for ts, val in aggregated.items() if pd.notna(val)
        ]
        
        # Prepare aggregated_data for visualization (with period labels)
        aggregated_data = []
        for ts, val in aggregated.items():
            if pd.notna(val):
                if aggregation_level == 'hourly':
                    period_label = ts.strftime('%Y-%m-%d %H:%M')
                elif aggregation_level == 'daily':
                    period_label = ts.strftime('%Y-%m-%d')
                else:  # weekly
                    period_label = f"Week of {ts.strftime('%Y-%m-%d')}"
                
                aggregated_data.append({
                    "period": period_label,
                    "value": float(val)
                })
        
        # Calculate overall aggregate across entire period
        if operation == 'mean':
            overall_aggregate = data['value'].mean()
        elif operation == 'min':
            overall_aggregate = data['value'].min()
        elif operation == 'max':
            overall_aggregate = data['value'].max()
        elif operation == 'sum':
            overall_aggregate = data['value'].sum()
        else:
            overall_aggregate = data['value'].agg(operation)
        
        return AnalyticsResult(
            value=result_data,
            unit=data['unit'].iloc[0],
            metadata={
                "aggregation_level": aggregation_level,
                "operation": operation,
                "num_periods": len(result_data),
                "overall_aggregate": float(overall_aggregate),
                "aggregated_data": aggregated_data
            },
            success=True,
            execution_time_ms=(time.time() - start_time) * 1000
        )


class SpatialComparisonTool(AnalyticsTool):
    """Compare statistics across multiple locations."""
    
    def __init__(self):
        super().__init__()
        self.name = "spatial_comparison"
        self.description = "Compare values across multiple locations"
        self.parameters = ["operation"]
    
    def execute(self, data: pd.DataFrame, **kwargs) -> AnalyticsResult:
        start_time = time.time()
        
        # Validate required columns
        required_cols = {'timestamp', 'value', 'unit', 'location'}
        if not required_cols.issubset(data.columns):
            return AnalyticsResult(
                value=None,
                unit=None,
                metadata={},
                success=False,
                error_message=f"Missing required columns: {required_cols - set(data.columns)}",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        operation = kwargs.get('operation', 'mean')
        
        # Compute statistics for each location
        comparison_results = []
        comparison_data = {}
        
        for location in data['location'].unique():
            location_data = data[data['location'] == location]
            
            if operation == 'mean':
                value = location_data['value'].mean()
            elif operation == 'min':
                value = location_data['value'].min()
            elif operation == 'max':
                value = location_data['value'].max()
            else:
                return AnalyticsResult(
                    value=None,
                    unit=None,
                    metadata={},
                    success=False,
                    error_message=f"Invalid operation: {operation}",
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            comparison_results.append({
                "location": location,
                "value": float(value)
            })
            
            # Add to comparison_data for visualization
            comparison_data[location] = {
                "mean": float(location_data['value'].mean()),
                "min": float(location_data['value'].min()),
                "max": float(location_data['value'].max()),
                "std": float(location_data['value'].std())
            }
        
        # Calculate relative differences and rankings
        comparison_results.sort(key=lambda x: x['value'], reverse=True)
        for i, result in enumerate(comparison_results, 1):
            result['rank'] = i
        
        highest_value = comparison_results[0]['value']
        for result in comparison_results:
            if highest_value != 0:
                result['percent_of_highest'] = (result['value'] / highest_value) * 100
        
        # Return structured comparison results
        return AnalyticsResult(
            value=comparison_results,
            unit=data['unit'].iloc[0],
            metadata={
                "operation": operation,
                "num_locations": len(comparison_results),
                "comparison_data": comparison_data
            },
            success=True,
            execution_time_ms=(time.time() - start_time) * 1000
        )


class StatisticalSummaryTool(AnalyticsTool):
    """Generate comprehensive statistical summary."""
    
    def __init__(self):
        super().__init__()
        self.name = "statistical_summary"
        self.description = "Generate comprehensive statistical summary"
        self.parameters = []
    
    def execute(self, data: pd.DataFrame, **kwargs) -> AnalyticsResult:
        start_time = time.time()
        
        # Validate required columns
        required_cols = {'timestamp', 'value', 'unit'}
        if not required_cols.issubset(data.columns):
            return AnalyticsResult(
                value=None,
                unit=None,
                metadata={},
                success=False,
                error_message=f"Missing required columns: {required_cols - set(data.columns)}",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Use pandas .describe() for basic statistics
        desc = data['value'].describe()
        
        # Calculate additional metrics using scipy.stats (skewness, quartiles)
        summary = {
            "count": int(desc['count']),
            "mean": float(desc['mean']),
            "std": float(desc['std']),
            "min": float(desc['min']),
            "q1": float(desc['25%']),
            "median": float(desc['50%']),
            "q3": float(desc['75%']),
            "max": float(desc['max']),
            "skewness": float(stats.skew(data['value'].dropna())),
            "kurtosis": float(stats.kurtosis(data['value'].dropna()))
        }
        
        # Rename keys for app.py compatibility (it expects '25%', '75%', etc.)
        statistics_for_viz = {
            "min": summary["min"],
            "25%": summary["q1"],
            "mean": summary["mean"],
            "median": summary["median"],
            "75%": summary["q3"],
            "max": summary["max"]
        }
        
        # Package all statistics into result metadata
        return AnalyticsResult(
            value=summary,
            unit=data['unit'].iloc[0],
            metadata={
                "operation": "summary",
                "statistics": statistics_for_viz
            },
            success=True,
            execution_time_ms=(time.time() - start_time) * 1000
        )