"""
Test integration between LLM and Data modules.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from llm import OllamaLLM, SystemContext
from data import SensorDataRepository
from datetime import datetime, timezone, timedelta


def test_end_to_end_query():
    """Test complete flow from query to data retrieval."""
    
    # Use absolute paths relative to project root
    data_config_path = project_root / 'config' / 'data_config.yaml'
    llm_config_path = project_root / 'config' / 'llm_config.yaml'
    
    # Initialize components
    repo = SensorDataRepository.from_config(data_config_path)
    llm = OllamaLLM.from_config(llm_config_path)
    
    # Connect and get system context
    repo.connect()
    
    context = SystemContext(
        available_sensors=repo.get_available_sensors(),
        available_locations=repo.get_available_locations(),
        time_range=repo.get_time_range()
    )
    
    # Test query
    user_query = "What was the average temperature in Node 15 last week?"
    
    # Extract intent
    task_spec = llm.extract_intent(user_query, context)
    
    print(f"Intent Type: {task_spec.intent_type}")
    print(f"Sensor Type: {task_spec.sensor_type}")
    print(f"Location: {task_spec.location}")
    print(f"Time Range: {task_spec.start_time} to {task_spec.end_time}")
    print(f"Operation: {task_spec.operation}")
    
    # Get data
    df = repo.get_readings(
        sensor_type=task_spec.sensor_type,
        location=task_spec.location,
        start_time=task_spec.start_time,
        end_time=task_spec.end_time
    )
    
    print(f"\nRetrieved {len(df)} readings")
    print(f"Average: {df['value'].mean():.2f}°C\n")

    print("Data frame:\n")
    print(df)
    
    repo.disconnect()


if __name__ == "__main__":
    test_end_to_end_query()