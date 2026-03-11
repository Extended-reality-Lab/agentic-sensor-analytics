"""
Bridge between LLM module and Data module.
Translates TaskSpecification from LLM into data queries.
"""

import pandas as pd
from typing import List, Dict
from datetime import datetime

# Import from LLM module (adjust path as needed)
from llm import TaskSpecification, IntentType

# Import from data module
from .repository import SensorDataRepository, RepositoryError
from .models import DataQuery, SystemState


class LLMDataBridge:
    """
    Bridges LLM TaskSpecification to Data Repository queries.
    Translates high-level intent into specific data requests.
    """
    
    def __init__(self, repository: SensorDataRepository):
        """
        Initialize bridge with data repository.
        
        Args:
            repository: SensorDataRepository instance
        """
        self.repository = repository
    
    def execute_task(self, task_spec: TaskSpecification) -> pd.DataFrame:
        """
        Execute TaskSpecification and return data.
        
        Args:
            task_spec: TaskSpecification from LLM
            
        Returns:
            DataFrame with sensor readings
            
        Raises:
            RepositoryError: If data retrieval fails
        """
        # Route based on intent type
        if task_spec.intent_type == IntentType.QUERY:
            return self._execute_query(task_spec)
        
        elif task_spec.intent_type == IntentType.COMPARISON:
            return self._execute_comparison(task_spec)
        
        elif task_spec.intent_type == IntentType.AGGREGATION:
            return self._execute_aggregation(task_spec)

        elif task_spec.intent_type == IntentType.THRESHOLD_SCAN:
            return self._execute_scan(task_spec)
        
        else:
            raise ValueError(f"Unknown intent type: {task_spec.intent_type}")
    
    def _execute_query(self, task_spec: TaskSpecification) -> pd.DataFrame:
        """
        Execute simple query for single location.
        
        Args:
            task_spec: TaskSpecification with intent_type=QUERY
            
        Returns:
            DataFrame with sensor readings
        """
        # task_spec.location is a string for QUERY intent
        location = task_spec.location if isinstance(task_spec.location, str) else task_spec.location[0]
        
        df = self.repository.get_readings(
            sensor_type=task_spec.sensor_type,
            location=location,
            start_time=task_spec.start_time,
            end_time=task_spec.end_time
        )
        
        return df
    
    def _execute_comparison(self, task_spec: TaskSpecification) -> pd.DataFrame:
        """
        Execute comparison query across multiple locations.
        
        Args:
            task_spec: TaskSpecification with intent_type=COMPARISON
            
        Returns:
            DataFrame with readings from all locations
        """
        # task_spec.location is a list for COMPARISON intent
        locations = task_spec.location if isinstance(task_spec.location, list) else [task_spec.location]
        
        df = self.repository.get_readings_multiple_locations(
            sensor_type=task_spec.sensor_type,
            locations=locations,
            start_time=task_spec.start_time,
            end_time=task_spec.end_time
        )
        
        return df
    
    def _execute_aggregation(self, task_spec: TaskSpecification) -> pd.DataFrame:
        """
        Execute aggregation query (returns raw data for analytics tools to aggregate).
        
        Args:
            task_spec: TaskSpecification with intent_type=AGGREGATION
            
        Returns:
            DataFrame with sensor readings
        """
        # For aggregation, get raw data and let analytics tools handle grouping
        location = task_spec.location if isinstance(task_spec.location, str) else task_spec.location[0]
        
        df = self.repository.get_readings(
            sensor_type=task_spec.sensor_type,
            location=location,
            start_time=task_spec.start_time,
            end_time=task_spec.end_time
        )
        
        return df

    def _execute_scan(self, task_spec: TaskSpecification) -> pd.DataFrame:
        """
        Execute threshold scan across all available locations.
        Fetches data for every location that carries the requested sensor type.

        Args:
            task_spec: TaskSpecification with intent_type=THRESHOLD_SCAN

        Returns:
            DataFrame with readings from all locations (has 'location' column)
        """
        sensors_by_node = self.repository.get_sensors_by_node()
        locations = [
            loc for loc, sensors in sensors_by_node.items()
            if task_spec.sensor_type in sensors
        ]

        df = self.repository.get_readings_multiple_locations(
            sensor_type=task_spec.sensor_type,
            locations=locations,
            start_time=task_spec.start_time,
            end_time=task_spec.end_time
        )

        return df
    
    def get_system_context(self) -> Dict:
        """
        Get system context for LLM (available sensors, locations, time range).
        This is used by the LLM to validate user queries.
        
        Returns:
            Dictionary with system metadata compatible with LLM SystemContext
        """
        self.repository.connect()
        
        available_sensors = self.repository.get_available_sensors()
        available_locations = self.repository.get_available_locations()
        time_range = self.repository.get_time_range()
        
        return {
            'available_sensors': available_sensors,
            'available_locations': available_locations,
            'time_range': time_range
        }
    
    def validate_task(self, task_spec: TaskSpecification) -> List[str]:
        """
        Validate TaskSpecification against available data.
        
        Args:
            task_spec: TaskSpecification to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Get locations as list
        locations = task_spec.get_locations_list()
        
        # Validate each location
        for location in locations:
            location_errors = self.repository.validate_parameters(
                sensor_type=task_spec.sensor_type,
                location=location,
                time_range=(task_spec.start_time, task_spec.end_time)
            )
            errors.extend(location_errors)
        
        return errors