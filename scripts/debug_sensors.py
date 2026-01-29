"""
Debug script to see ALL sensors - no filtering, no ignoring.
Shows every single sensor in the system.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import SensorDataRepository

def main():
    print("Connecting to sensor data repository...")
    repo = SensorDataRepository.from_config()
    repo.connect()
    
    print("\nFetching all sensors...")
    sensors = repo._get_all_sensors()
    
    print(f"\nFound {len(sensors)} total sensors")
    print("\n" + "="*120)
    
    # Statistics by raw sensor type
    sensor_types = {}
    for sensor in sensors:
        raw_type = sensor.sensor_type
        if raw_type not in sensor_types:
            sensor_types[raw_type] = []
        sensor_types[raw_type].append(sensor)
    
    print(f"\nSensor Types Breakdown ({len(sensor_types)} unique types):")
    print("="*120)
    
    for raw_type, sensors_of_type in sorted(sensor_types.items(), key=lambda x: -len(x[1])):
        normalized = repo._normalize_sensor_type(raw_type)
        count = len(sensors_of_type)
        
        if normalized:
            print(f"✓ '{raw_type}' → Normalized to '{normalized}' ({count} sensors)")
        else:
            print(f"  '{raw_type}' → Not normalized ({count} sensors)")
    
    print("\n" + "="*120)
    print("\nALL SENSORS - COMPLETE LIST:")
    print("="*120)
    print(f"Showing all {len(sensors)} sensors\n")
    
    # Group by node for better organization
    sensors_by_node = {}
    for sensor in sensors:
        node_id = sensor.node_id
        if node_id not in sensors_by_node:
            sensors_by_node[node_id] = []
        sensors_by_node[node_id].append(sensor)
    
    print(f"Found {len(sensors_by_node)} nodes total\n")
    
    for node_id in sorted(sensors_by_node.keys()):
        node_sensors = sensors_by_node[node_id]
        location = node_sensors[0].location if node_sensors else "Unknown"
        
        print(f"\n{'─'*120}")
        print(f"NODE {node_id} - Location: {location} - {len(node_sensors)} sensors")
        print(f"{'─'*120}")
        
        for sensor in sorted(node_sensors, key=lambda s: s.sensor_id):
            normalized = repo._normalize_sensor_type(sensor.sensor_type)
            normalized_str = f"{normalized}" if normalized else "not_normalized"
            
            # Format channel as string
            channel_str = str(sensor.input_channel) if sensor.input_channel is not None else "N/A"
            
            print(f"  Sensor {sensor.sensor_id:5d} | "
                  f"Type: {sensor.sensor_type:45s} | "
                  f"Normalized: {normalized_str:15s} | "
                  f"Ch: {channel_str:3s} | "
                  f"Unit: {sensor.unit:6s} | "
                  f"Name: {sensor.name}")
    
    print("\n" + "="*120)
    print("\nSTATISTICS:")
    print("="*120)
    
    # Count by normalized type
    normalized_counts = {}
    not_normalized = 0
    
    for sensor in sensors:
        normalized = repo._normalize_sensor_type(sensor.sensor_type)
        if normalized:
            if normalized not in normalized_counts:
                normalized_counts[normalized] = 0
            normalized_counts[normalized] += 1
        else:
            not_normalized += 1
    
    print(f"\nTotal Sensors: {len(sensors)}")
    print(f"\nNormalized sensor types:")
    for sensor_type, count in sorted(normalized_counts.items()):
        print(f"  • {sensor_type}: {count} sensors")
    
    print(f"\nNot normalized: {not_normalized} sensors")
    
    print("\n" + "="*120)
    
    # Save to detailed text file
    output_file = Path(__file__).parent / "all_sensors_detailed_report.txt"
    print(f"\nSaving detailed report to: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*120 + "\n")
        f.write("COMPLETE SENSOR REPORT - ALL SENSORS\n")
        f.write("="*120 + "\n\n")
        f.write(f"Total Sensors: {len(sensors)}\n")
        f.write(f"Total Nodes: {len(sensors_by_node)}\n\n")
        
        f.write("SENSOR TYPE BREAKDOWN:\n")
        f.write("-"*120 + "\n")
        for raw_type, sensors_of_type in sorted(sensor_types.items(), key=lambda x: -len(x[1])):
            normalized = repo._normalize_sensor_type(raw_type)
            count = len(sensors_of_type)
            f.write(f"'{raw_type}' -> {normalized if normalized else 'not_normalized'}: {count} sensors\n")
        
        f.write("\n" + "="*120 + "\n")
        f.write("ALL SENSORS BY NODE:\n")
        f.write("="*120 + "\n\n")
        
        for node_id in sorted(sensors_by_node.keys()):
            node_sensors = sensors_by_node[node_id]
            location = node_sensors[0].location if node_sensors else "Unknown"
            
            f.write(f"\n{'─'*120}\n")
            f.write(f"NODE {node_id} - {location} - {len(node_sensors)} sensors\n")
            f.write(f"{'─'*120}\n")
            
            for sensor in sorted(node_sensors, key=lambda s: s.sensor_id):
                normalized = repo._normalize_sensor_type(sensor.sensor_type)
                channel_str = str(sensor.input_channel) if sensor.input_channel is not None else "N/A"
                
                f.write(f"ID:{sensor.sensor_id:5d} | "
                       f"Type:{sensor.sensor_type:45s} | "
                       f"Norm:{normalized if normalized else 'not_normalized':15s} | "
                       f"Ch:{channel_str:3s} | "
                       f"Unit:{sensor.unit:6s} | "
                       f"Name:{sensor.name}\n")
        
        f.write("\n" + "="*120 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*120 + "\n")
    
    print(f"✓ Detailed report saved!")
    print("\n" + "="*120)
    
    repo.disconnect()
    
    print("\nDone! All sensors have been listed above and saved to the report file.")


if __name__ == "__main__":
    main()