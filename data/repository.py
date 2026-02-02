"""
SensorDataRepository: Unified interface for accessing sensor data.
Translates LLM TaskSpecification into API queries and returns formatted data.
"""

import pandas as pd
from typing import List, Optional, Dict
from datetime import datetime, timezone
from pathlib import Path

from .api_client import SMTAPIClient, SMTAPIError
from .models import (
    SensorReading,
    SensorMetadata,
    NodeMetadata,
    DataQuery,
    DataQueryResult,
    SystemState,
    TimeRange
)
from .config import DataConfig, load_config


class RepositoryError(Exception):
    """Base exception for repository errors."""
    pass


class SensorDataRepository:
    """
    Central repository for sensor data access.
    Provides unified interface for querying data and system metadata.
    """
    
    def __init__(
        self,
        api_client: Optional[SMTAPIClient] = None,
        config: Optional[DataConfig] = None
    ):
        """
        Initialize repository.
        
        Args:
            api_client: SMT API client (if None, creates from config)
            config: Data configuration (if None, loads from file)
        """
        if config is None:
            config = load_config()
        
        self.config = config
        
        if api_client is None:
            api_client = SMTAPIClient.from_config(config)
        
        self.api_client = api_client
        
        # Cache for metadata
        self._nodes_cache: Optional[List[NodeMetadata]] = None
        self._sensors_cache: Optional[List[SensorMetadata]] = None
        self._system_state_cache: Optional[SystemState] = None
    
    @classmethod
    def from_config(cls, config_path: Optional[str | Path] = None) -> 'SensorDataRepository':
        """
        Create repository from configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configured SensorDataRepository instance
        """
        config = load_config(config_path)
        return cls(config=config)
    
    def connect(self) -> None:
        """Establish connection to API."""
        if not self.api_client.authenticated:
            self.api_client.login()
    
    def disconnect(self) -> None:
        """Close connection to API."""
        if self.api_client.authenticated:
            self.api_client.logout()
    
    def get_readings(
        self,
        sensor_type: str,
        location: str,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """
        Retrieve sensor readings as pandas DataFrame.
        
        Args:
            sensor_type: Type of sensor (temperature, humidity, etc.)
            location: Location name
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            DataFrame with columns: timestamp, value, unit, location
            
        Raises:
            RepositoryError: If data retrieval fails
        """
        self.connect()
        
        try:
            # Find matching sensor
            sensor = self._find_sensor(sensor_type, location)
            
            if sensor is None:
                raise RepositoryError(
                    f"No {sensor_type} sensor found for location: {location}"
                )
            
            # Get readings from API
            readings = self.api_client.get_sensor_data(
                sensor_id=sensor.sensor_id,
                start_date=start_time,
                end_date=end_time
            )
            
            # Update unit in readings
            for reading in readings:
                reading.unit = sensor.unit
            
            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    'timestamp': r.timestamp,
                    'value': r.value,
                    'unit': r.unit,
                    'location': location,
                    'quality_flag': r.quality_flag
                }
                for r in readings
            ])
            
            return df
            
        except SMTAPIError as e:
            raise RepositoryError(f"API error: {e}")
        except Exception as e:
            raise RepositoryError(f"Error retrieving readings: {e}")
    
    def get_readings_multiple_locations(
        self,
        sensor_type: str,
        locations: List[str],
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """
        Retrieve sensor readings from multiple locations.
        
        Args:
            sensor_type: Type of sensor
            locations: List of location names
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            DataFrame with readings from all locations
        """
        dfs = []
        
        for location in locations:
            try:
                df = self.get_readings(sensor_type, location, start_time, end_time)
                dfs.append(df)
            except RepositoryError as e:
                print(f"Warning: Could not get data for {location}: {e}")
                continue
        
        if not dfs:
            raise RepositoryError(f"No data found for any location: {locations}")
        
        return pd.concat(dfs, ignore_index=True)
    
    def get_available_sensors(self) -> List[str]:
        """Get list of available sensor types."""
        self.connect()
        
        sensors = self._get_all_sensors()
        
        sensor_types = set()
        for sensor in sensors:
            sensor_type = self._normalize_sensor_type(sensor.sensor_type)
            if sensor_type:
                sensor_types.add(sensor_type)
        
        return sorted(list(sensor_types))
    
    def get_available_locations(self) -> List[str]:
        """Get list of available locations."""
        self.connect()
        
        nodes = self._get_all_nodes()
        
        # Convert to human-readable format
        locations = {
            self._get_human_readable_location(node.name) 
            for node in nodes 
            if node.location
        }
        
        return sorted(list(locations))
    
    def get_time_range(self) -> tuple[datetime, datetime]:
        """
        Get available time range for data.
        
        Returns:
            Tuple of (earliest_datetime, latest_datetime)
            
        Note:
            This is a simplified implementation. In production,
            you might query the API for actual min/max timestamps.
        """
        # For now, return a reasonable range
        # TODO: Query API for actual min/max timestamps
        earliest = datetime(2019, 5, 1, tzinfo=timezone.utc)
        latest = datetime.now(timezone.utc)
        
        return (earliest, latest)
    
    def validate_parameters(
        self,
        sensor_type: str,
        location: str,
        time_range: tuple[datetime, datetime]
    ) -> List[str]:
        """
        Validate query parameters against available data.
        
        Args:
            sensor_type: Sensor type to validate
            location: Location to validate
            time_range: Time range tuple (start, end)
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check sensor type
        available_sensors = self.get_available_sensors()
        if sensor_type not in available_sensors:
            errors.append(
                f"Unknown sensor type '{sensor_type}'. "
                f"Available: {', '.join(available_sensors)}"
            )
        
        # Check location
        available_locations = self.get_available_locations()
        if location not in available_locations:
            errors.append(
                f"Unknown location '{location}'. "
                f"Available: {', '.join(available_locations[:5])}..."
            )
        
        # Check time range
        min_date, max_date = self.get_time_range()
        if time_range[0] < min_date:
            errors.append(
                f"Start time ({time_range[0].date()}) is before available data "
                f"(starts {min_date.date()})"
            )
        if time_range[1] > max_date:
            errors.append(
                f"End time ({time_range[1].date()}) is after available data "
                f"(ends {max_date.date()})"
            )
        
        return errors
    
    def get_system_state(self) -> SystemState:
        """
        Get current system state with metadata.
        
        Returns:
            SystemState object
        """
        self.connect()
        
        nodes = self._get_all_nodes()
        sensors = self._get_all_sensors()
        time_range = self.get_time_range()
        
        available_sensors = self.get_available_sensors()
        available_locations = self.get_available_locations()
        
        return SystemState(
            available_sensors=available_sensors,
            available_locations=available_locations,
            time_range=TimeRange(start_time=time_range[0], end_time=time_range[1]),
            total_sensors=len(sensors),
            total_nodes=len(nodes),
            last_updated=datetime.now(timezone.utc)
        )
    
    def _get_all_nodes(self) -> List[NodeMetadata]:
        """Get all nodes (with caching)."""
        if self._nodes_cache is None:
            self._nodes_cache = self.api_client.list_nodes(self.config.api.job_id)
        return self._nodes_cache
    
    def _get_all_sensors(self) -> List[SensorMetadata]:
        """Get all sensors from all nodes (with caching)."""
        if self._sensors_cache is None:
            nodes = self._get_all_nodes()
            sensors = []
            
            for node in nodes:
                try:
                    node_sensors = self.api_client.list_sensors(node.node_id)
                    # Update location from node
                    for sensor in node_sensors:
                        sensor.location = node.location
                    sensors.extend(node_sensors)
                except SMTAPIError as e:
                    print(f"Warning: Could not get sensors for node {node.node_id}: {e}")
                    continue
            
            self._sensors_cache = sensors
        
        return self._sensors_cache
    
    def _find_sensor(
        self,
        sensor_type: str,
        location: str
    ) -> Optional[SensorMetadata]:
        """Find sensor matching type and location."""
        sensors = self._get_all_sensors()
        
        for sensor in sensors:
            # Check if sensor type matches
            normalized_type = self._normalize_sensor_type(sensor.sensor_type)
            if normalized_type != sensor_type:
                continue
            
            # Convert sensor location to human-readable format
            sensor_location_readable = self._get_human_readable_location(sensor.location)
            
            # Check exact match (case-insensitive)
            if sensor_location_readable.lower() == location.lower():
                return sensor
            
            # Also check exact match against raw location
            if sensor.location.lower() == location.lower():
                return sensor
        
        return None
    
    def _get_human_readable_location(self, node_name: str) -> str:
        """Convert node name to human-readable format."""
        # Extract node number from format like "15_9279"
        if '_' in node_name:
            node_num = node_name.split('_')[0]
            return f"Node {node_num}"
        return node_name
    
    def _normalize_sensor_type(self, sensor_type_name: str) -> str:
        """
        Normalize sensor type name to standard categories.
        
        Args:
            sensor_type_name: Raw sensor type name from API
            
        Returns:
            Normalized sensor type (temperature, humidity, co2, etc.)
        """
        name_lower = sensor_type_name.lower()
        
        # Check against configured keywords
        if any(kw in name_lower for kw in self.config.sensor_mapping.temperature_keywords):
            return 'temperature'
        elif any(kw in name_lower for kw in self.config.sensor_mapping.humidity_keywords):
            return 'humidity'
        elif any(kw in name_lower for kw in self.config.sensor_mapping.co2_keywords):
            return 'co2'
        elif any(kw in name_lower for kw in self.config.sensor_mapping.moisture_keywords):
            return 'moisture'
        elif 'equation' in name_lower or 'quadratic' in name_lower or 'strain' in name_lower:
            return 'strain'
        elif name_lower == 'unknown':
            return None
        else:
            return None
    def clear_cache(self) -> None:
        """Clear all cached metadata."""
        self._nodes_cache = None
        self._sensors_cache = None
        self._system_state_cache = None
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False