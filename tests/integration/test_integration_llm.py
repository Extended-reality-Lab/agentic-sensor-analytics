"""
Simple test script for the LLM module.
Run this to verify Ollama connection and intent extraction.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm import OllamaLLM, SystemContext, LLMError, LLMConfig


def test_ollama_connection():
    """Test basic Ollama connection."""
    print("Testing Ollama connection...")
    
    try:
        # Try to load from config file
        try:
            llm = OllamaLLM.from_config()
            print(f"✓ Loaded configuration from file")
        except FileNotFoundError:
            # Fall back to defaults
            print("⚠ No config file found, using defaults")
            llm = OllamaLLM(
                model_name="llama3.1:8b",
                temperature=0.1
            )
        
        print("✓ Successfully connected to Ollama")
        print(f"✓ Model loaded: {llm.model_name}")
        print(f"✓ Base URL: {llm.base_url}")
        print(f"✓ Temperature: {llm.temperature}")
        print(f"✓ Max retries: {llm.max_retries}")
        
        # Check if model is available
        if llm.is_available():
            print("✓ LLM is available and ready")
        else:
            print("✗ LLM is not available")
            return False
            
        # Get model info
        model_info = llm.get_model_info()
        if "error" not in model_info:
            print(f"✓ Model info retrieved: {model_info.get('name', 'Unknown')}")
        
        return True
        
    except LLMError as e:
        print(f"✗ Connection failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_intent_extraction():
    """Test intent extraction with example queries."""
    print("\nTesting intent extraction...")
    
    try:
        # Load LLM with config
        try:
            llm = OllamaLLM.from_config()
        except FileNotFoundError:
            llm = OllamaLLM(model_name="llama3.1:8b", temperature=0.1)
        
        # Create mock system context
        context = SystemContext(
            available_sensors=["temperature", "humidity", "co2", "energy", "occupancy"],
            available_locations=["Node12", "Node13", "Node14", "Node15", "Node16"],
            time_range=(
                datetime(2025, 1, 1, tzinfo=timezone.utc),
                datetime.now(timezone.utc) + timedelta(days=1)
            )
        )
        
        # Test queries
        test_queries = [
            "What was the average temperature in Node 15 yesterday?",
            "Compare humidity levels between Node 14 and Node 15 last week",
            "Show me daily average humidity for Node 15 over the past month"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Test Query {i} ---")
            print(f"Query: {query}")
            
            try:
                task_spec = llm.extract_intent(query, context)
                
                print(f"✓ Intent extraction successful")
                print(f"  Intent Type: {task_spec.intent_type}")
                print(f"  Sensor: {task_spec.sensor_type}")
                print(f"  Location: {task_spec.location}")
                print(f"  Operation: {task_spec.operation}")
                print(f"  Time Range: {task_spec.start_time.date()} to {task_spec.end_time.date()}")
                print(f"  Confidence: {task_spec.confidence:.2f}")
                
            except Exception as e:
                print(f"✗ Extraction failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_result_explanation():
    """Test result explanation generation."""
    print("\n\nTesting result explanation...")
    
    try:
        # Load LLM with config
        try:
            llm = OllamaLLM.from_config()
        except FileNotFoundError:
            llm = OllamaLLM(model_name="llama3.1:8b", temperature=0.1)
        
        # Create mock system context
        context = SystemContext(
            available_sensors=["temperature"],
            available_locations=["Node15"],
            time_range=(
                datetime(2025, 1, 1, tzinfo=timezone.utc),
                datetime.now(timezone.utc) + timedelta(days=1)
            )
        )
        
        # Mock query and extraction
        query = "What was the average temperature in Node15 yesterday?"
        task_spec = llm.extract_intent(query, context)
        
        # Mock analytics results
        mock_results = [
            {
                "value": 22.4,
                "unit": "°C",
                "operation": "mean",
                "sample_size": 1440,
                "std_dev": 1.2,
                "min": 19.5,
                "max": 25.1
            }
        ]
        
        # Generate explanation
        explanation = llm.explain_results(query, task_spec, mock_results)

        assert len(explanation) > 0, "Explanation should not be empty"
        assert "temperature" in explanation.lower() or "22.4" in explanation
        
        print("✓ Explanation generated successfully:")
        print(f"  {explanation}")
        
        return True
        
    except Exception as e:
        print(f"✗ Explanation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_explanation():
    """Test error explanation generation."""
    print("\n\nTesting error explanation...")
    
    try:
        # Load LLM with config
        try:
            llm = OllamaLLM.from_config()
        except FileNotFoundError:
            llm = OllamaLLM(model_name="llama3.1:8b", temperature=0.1)
        
        query = "What was the temperature in Node100 yesterday?"
        errors = [
            "Unknown location 'Node100'. Available locations include: Node13, Node14, Node15...",
        ]
        
        explanation = llm.explain_error(query, errors)
        
        print("✓ Error explanation generated:")
        print(f"  {explanation}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error explanation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """Test configuration loading."""
    print("\n\nTesting configuration loading...")
    
    try:
        # Test loading from file
        config_path = Path(__file__).parent.parent.parent / "config" / "llm_config.yaml"
        
        if config_path.exists():
            config = LLMConfig.from_yaml(config_path)
            print(f"✓ Loaded config from: {config_path}")
            print(f"  Model: {config.llm.model_name}")
            print(f"  Base URL: {config.llm.base_url}")
            print(f"  Temperature: {config.llm.temperature}")
            print(f"  Streaming: {config.performance.enable_streaming}")
        else:
            print(f"⚠ Config file not found at: {config_path}")
            print(f"  Using default configuration")
            config = LLMConfig()
            print(f"  Model: {config.llm.model_name}")
        
        return True
        
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("LLM Module Test Suite")
    print("=" * 60)
    
    # Run configuration test first
    test_config_loading()
    
    # Run connection test
    connection_ok = test_ollama_connection()
    
    if connection_ok:
        test_intent_extraction()
        test_result_explanation()
        test_error_explanation()
    else:
        print("\n⚠ Skipping remaining tests due to connection failure")
        print("\nMake sure Ollama is running:")
        print("  1. Install: curl -fsSL https://ollama.com/install.sh | sh")
        print("  2. Pull model: ollama pull llama3.1:8b")
        print("  3. Verify: ollama list")
    
    print("\n" + "=" * 60)
    print("Test suite completed")
    print("=" * 60)