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
    """Test intent extraction with all query types."""
    print("\nTesting intent extraction...")
    
    try:
        # Load LLM with config
        try:
            llm = OllamaLLM.from_config()
        except FileNotFoundError:
            llm = OllamaLLM(model_name="llama3.1:8b", temperature=0.1)
        
        # Create mock system context
        context = SystemContext(
            available_sensors=["temperature", "humidity", "moisture", "strain"],
            available_locations=["Node 1", "Node 4", "Node 8", "Node 15", "Node 25", "Node 35"],
            time_range=(
                datetime(2025, 1, 1, tzinfo=timezone.utc),
                datetime.now(timezone.utc) + timedelta(days=1)
            )
        )

        # One test query per intent type
        test_queries = [
            # (description, query, expected_intent, expected_operation)
            ("QUERY/mean",          "What was the average temperature in Node 15 yesterday?",              "query",          "mean"),
            ("QUERY/max",           "What was the highest humidity in Node 4 last month?",                 "query",          "max"),
            ("QUERY/min",           "What was the lowest moisture level in Node 1 last week?",             "query",          "min"),
            ("QUERY/summary",       "Provide a summary of moisture levels in Node 15 for the past month.", "query",          "summary"),
            ("AGGREGATION/daily",   "What was the temperature each day in Node 1 last week?",              "aggregation",    "mean"),
            ("AGGREGATION/hourly",  "Show me hourly humidity in Node 4 yesterday.",                        "aggregation",    "mean"),
            ("COMPARISON",          "Compare humidity between Node 15 and Node 25 over the past week.",    "comparison",     "mean"),
            ("THRESHOLD",      "Which nodes exceeded 20°C more than 50% of the time last month?",    "threshold", "mean"),
        ]
        
        passed = 0
        failed = 0

        for desc, query, expected_intent, expected_op in test_queries:
            print(f"\n--- {desc} ---")
            print(f"Query: {query}")
            
            try:
                task_spec = llm.extract_intent(query, context)

                intent_ok = task_spec.intent_type.value == expected_intent
                op_ok     = task_spec.operation.value  == expected_op

                status = "✓" if (intent_ok and op_ok) else "✗"
                if intent_ok and op_ok:
                    passed += 1
                else:
                    failed += 1

                print(f"  {status} intent_type : {task_spec.intent_type.value!r}  (expected {expected_intent!r})")
                print(f"  {status} operation   : {task_spec.operation.value!r}  (expected {expected_op!r})")
                print(f"    sensor_type       : {task_spec.sensor_type}")
                print(f"    location          : {task_spec.location}")
                print(f"    time_range        : {task_spec.start_time.date()} to {task_spec.end_time.date()}")
                print(f"    aggregation_level : {task_spec.aggregation_level}")
                print(f"    threshold_value   : {task_spec.threshold_value}")
                print(f"    result_threshold  : {task_spec.result_threshold}")
                print(f"    confidence        : {task_spec.confidence:.2f}")
                
            except Exception as e:
                failed += 1
                print(f"  ✗ Extraction failed: {e}")

        print(f"\n{'='*40}")
        print(f"Intent extraction: {passed}/{passed+failed} passed")
        print(f"{'='*40}")
        return failed == 0
        
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
            available_locations=["Node 15"],
            time_range=(
                datetime(2025, 1, 1, tzinfo=timezone.utc),
                datetime.now(timezone.utc) + timedelta(days=1)
            )
        )
        
        # Mock query and extraction
        query = "What was the average temperature in Node 15 yesterday?"
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
            "Unknown location 'Node100'. Available locations include: Node 1, Node 4, Node 15...",
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