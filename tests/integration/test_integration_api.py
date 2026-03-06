"""
Quick test script to verify API connection and basic functionality.
Run from project root: python tests/integration/test_api_connection.py
"""

import sys
from pathlib import Path

# CRITICAL: Add project root to path FIRST
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("Testing SMT Analytics API Connection...")
print("=" * 60)

# Now import after path is set
from data.api_client import SMTAPIClient
from data.config import load_config
from datetime import datetime, timedelta, timezone

# Load configuration from config/data_config.yaml
config_path = project_root / "config" / "data_config.yaml"
config = load_config(config_path)

# Initialize client with credentials from config
client = SMTAPIClient(
    username=config.api.username,
    password=config.api.password,
    base_url=config.api.base_url
)

try:
    # Test 1: Login
    print("\n1. Testing login...")
    if client.login():
        print("   ✓ Login successful!")
    else:
        print("   ✗ Login failed!")
        sys.exit(1)
    
    # Test 2: List nodes
    print(f"\n2. Testing node listing...")
    nodes = client.list_nodes(job_id=config.api.job_id)
    print(f"   ✓ Successfully retrieved {len(nodes)} nodes")
    
    # Test 3: List sensors from first node
    print(f"\n3. Testing sensor listing...")
    if nodes:
        first_node = nodes[-1]
        sensors = client.list_sensors(first_node.node_id)
        print(f"   ✓ Successfully retrieved {len(sensors)} sensors from node {first_node.name}")
    else:
        print("   ⚠ No nodes found to test sensors")
    
    # Test 4: Get sensor data
    print(f"\n4. Testing data retrieval...")
    if nodes and sensors:
        test_sensor = sensors[0] if sensors else None
        
        if test_sensor:
            readings = client.get_sensor_data(
                sensor_id=test_sensor.sensor_id,
                start_date=datetime.now(timezone.utc) - timedelta(days=180),
                end_date=datetime.now(timezone.utc)
            )
            print(f"   ✓ Successfully retrieved {len(readings)} readings from sensor {test_sensor.name}")
        else:
            print("   ⚠ No sensors found for data test")
    else:
        print("   ⚠ Skipping data retrieval test")
    
    # Test 5: Logout
    print("\n5. Testing logout...")
    if client.logout():
        print("   ✓ Logout successful!")
    
    print("\n" + "=" * 60)
    print("✓ All API tests passed!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)