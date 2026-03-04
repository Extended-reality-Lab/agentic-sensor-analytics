"""
Data module for sensor data access and management.

This module provides a unified interface for accessing sensor data from the
SMT Analytics API. It handles authentication, data retrieval, validation,
and provides integration with the LLM and analytics modules.

Architecture:
- API Client: Handles SMT Analytics API communication
- Repository: Provides high-level data access interface
- Models: Pydantic models for type safety and validation
- Validation: Data quality checks and flagging
- Config: YAML-based configuration management
"""

from .models import (
    SensorReading,
    SensorMetadata,
    NodeMetadata,
    DataQuery,
    DataQueryResult,
    SystemState,
    TimeRange,
    SensorType
)

from .api_client import (
    SMTAPIClient,
    SMTAPIError,
    SMTAuthenticationError
)

from .repository import (
    SensorDataRepository,
    RepositoryError
)

from .unified_repository import UnifiedRepository

from .config import (
    DataConfig,
    APISettings,
    CacheSettings,
    ValidationSettings,
    load_config
)

from .llm_bridge import (
    LLMDataBridge
)

__all__ = [
    # Main repository
    'SensorDataRepository',
    'UnifiedRepository',
    
    # API Client
    'SMTAPIClient',
    
    # Models
    'SensorReading',
    'SensorMetadata',
    'NodeMetadata',
    'DataQuery',
    'DataQueryResult',
    'SystemState',
    'TimeRange',
    'SensorType',
    
    # Configuration
    'DataConfig',
    'APISettings',
    'CacheSettings',
    'ValidationSettings',
    'load_config',
    
    # Errors
    'SMTAPIError',
    'SMTAuthenticationError',
    'RepositoryError',

    # Data Bridge
    'LLMDataBridge',
]

__version__ = '0.1.0'
__author__ = 'Sean Clayton'
__description__ = 'Data access layer for agentic smart building analytics'