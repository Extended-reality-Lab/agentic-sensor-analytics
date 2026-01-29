"""
Integration tests for the agent module.
Tests the complete agentic workflow using LangGraph.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent import AgentExecutor, create_initial_state, get_execution_summary
from llm import OllamaLLM, SystemContext
from data import SensorDataRepository, LLMDataBridge


def test_agent_executor_creation():
    """Test creating agent executor from config."""
    print("\nTesting AgentExecutor creation...")
    
    try:
        executor = AgentExecutor.from_config()
        
        assert executor.llm is not None
        assert executor.repository is not None
        assert executor.bridge is not None
        assert executor.graph is not None
        
        print("✓ AgentExecutor created successfully")
        print(f"  LLM: {executor.llm.model_name}")
        print(f"  Repository: {type(executor.repository).__name__}")
        
        return True
    
    except Exception as e:
        print(f"✗ Failed to create executor: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_query_workflow():
    """Test complete workflow with a simple query."""
    print("\nTesting simple query workflow...")
    
    try:
        executor = AgentExecutor.from_config()
        
        # Execute a simple query
        query = "What was the average temperature in Node 15 yesterday?"
        print(f"Query: {query}")
        
        result = executor.execute(query)
        
        # Check basic state structure
        assert 'user_query' in result
        assert 'task_spec' in result
        assert 'execution_trace' in result
        assert 'success' in result
        
        print(f"\n✓ Workflow completed")
        print(f"  Success: {result['success']}")
        
        # Print execution trace
        print("\n" + executor.visualize_trace(result))
        
        # Check result
        if result['success']:
            assert 'explanation' in result
            print(f"\n✓ Explanation generated:")
            print(f"  {result['explanation']}...")
        else:
            assert 'error_explanation' in result
            print(f"\n✓ Error explanation generated:")
            print(f"  {result['error_explanation']}")
        
        return True
    
    except Exception as e:
        print(f"✗ Workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comparison_query_workflow():
    """Test workflow with comparison query."""
    print("\nTesting comparison query workflow...")
    
    try:
        executor = AgentExecutor.from_config()
        
        query = "Compare humidity levels between Node 14 and Node 15 last week"
        print(f"Query: {query}")
        
        result = executor.execute(query)
        
        print(f"\n✓ Workflow completed")
        print(f"  Success: {result['success']}")
        
        # Print trace
        trace = executor.get_execution_trace(result)
        print(f"  Steps executed: {len(trace)}")
        
        # Check task specification
        if result.get('task_spec'):
            task_spec = result['task_spec']
            print(f"\n✓ Task specification:")
            print(f"  Intent: {task_spec.intent_type}")
            print(f"  Sensor: {task_spec.sensor_type}")
            print(f"  Locations: {task_spec.location}")
            print(f"  Operation: {task_spec.operation}")
        
        if result['success'] and result.get('explanation'):
            print(f"\n✓ Explanation:")
            print(f"  {result['explanation']}")
        
        return True
    
    except Exception as e:
        print(f"✗ Comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_aggregation_query_workflow():
    """Test workflow with temporal aggregation query."""
    print("\nTesting aggregation query workflow...")
    
    try:
        executor = AgentExecutor.from_config()
        
        query = "Show me daily average temperature for Node 15 last week"
        print(f"Query: {query}")
        
        result = executor.execute(query)
        
        print(f"\n✓ Workflow completed")
        print(f"  Success: {result['success']}")
        
        # Check aggregation level was extracted
        if result.get('task_spec'):
            task_spec = result['task_spec']
            print(f"\n✓ Task specification:")
            print(f"  Intent: {task_spec.intent_type}")
            print(f"  Aggregation level: {task_spec.aggregation_level}")
        
        # Check analytics result
        if result.get('analytics_result'):
            analytics = result['analytics_result']
            print(f"\n✓ Analytics result:")
            print(f"  Operation: {analytics.get('metadata', {}).get('operation')}")
            print(f"  Periods: {analytics.get('metadata', {}).get('num_periods')}")
        
        return True
    
    except Exception as e:
        print(f"✗ Aggregation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validation_error_handling():
    """Test workflow with invalid query that should fail validation."""
    print("\nTesting validation error handling...")
    
    try:
        executor = AgentExecutor.from_config()
        
        # Query with invalid location
        query = "What was the temperature in Node 100 yesterday?"
        print(f"Query: {query}")
        
        result = executor.execute(query)
        
        print(f"\n✓ Workflow completed")
        print(f"  Success: {result['success']}")
        
        # Should have failed validation
        assert result['success'] == False, "Expected validation to fail"
        assert result.get('validation_errors'), "Expected validation errors"
        assert result.get('error_explanation'), "Expected error explanation"
        
        print(f"\n✓ Validation errors detected:")
        for error in result.get('validation_errors', []):
            print(f"  - {error}")
        
        print(f"\n✓ Error explanation:")
        print(f"  {result['error_explanation']}")
        
        # Check that handle_error node was called
        trace = executor.get_execution_trace(result)
        error_steps = [t for t in trace if t['step'] == 'handle_error']
        assert len(error_steps) > 0, "Expected handle_error to be called"
        print(f"\n✓ handle_error node was executed")
        
        return True
    
    except AssertionError as e:
        print(f"✗ Assertion failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_execution_trace():
    """Test that execution trace is properly recorded."""
    print("\nTesting execution trace recording...")
    
    try:
        executor = AgentExecutor.from_config()
        
        query = "What was the average temperature in Node 15 yesterday?"
        result = executor.execute(query)
        
        # Check trace exists
        trace = result.get('execution_trace', [])
        assert len(trace) > 0, "Expected execution trace"
        
        print(f"\n✓ Execution trace recorded: {len(trace)} steps")
        
        # Check trace structure
        for entry in trace:
            assert 'step' in entry
            assert 'timestamp' in entry
            assert 'status' in entry
            assert 'details' in entry
            
            print(f"  Step: {entry['step']}, Status: {entry['status']}")
        
        # Get execution summary
        summary = get_execution_summary(result)
        
        print(f"\n✓ Execution summary:")
        print(f"  Total steps: {summary['total_steps']}")
        print(f"  Completed: {summary['steps_completed']}")
        print(f"  Failed: {summary['steps_failed']}")
        print(f"  Total duration: {summary['total_duration_ms']:.2f}ms")
        print(f"  Success: {summary['success']}")
        
        return True
    
    except Exception as e:
        print(f"✗ Trace test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_state_structure():
    """Test agent state structure and transitions."""
    print("\nTesting agent state structure...")
    
    try:
        # Test initial state creation
        query = "Test query"
        state = create_initial_state(query)
        
        assert state['user_query'] == query
        assert state['task_spec'] is None
        assert state['validation_errors'] == []
        assert state['data'] is None
        assert state['execution_trace'] == []
        assert state['success'] == False
        assert state['start_time'] is not None
        
        print("✓ Initial state structure is correct")
        
        # Test that state can flow through workflow
        executor = AgentExecutor.from_config()
        result = executor.execute("What was the average temperature in Node 15 yesterday?")
        
        # Check final state has all expected fields
        assert 'user_query' in result
        assert 'task_spec' in result
        assert 'execution_trace' in result
        assert 'success' in result
        assert 'end_time' in result
        
        print("✓ Final state structure is correct")
        
        # Check that either explanation or error_explanation is present
        assert result.get('explanation') or result.get('error_explanation'), \
            "Expected either explanation or error_explanation"
        
        print("✓ State transitions completed successfully")
        
        return True
    
    except Exception as e:
        print(f"✗ State structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_conditional_routing():
    """Test that conditional routing works correctly."""
    print("\nTesting conditional routing...")
    
    try:
        executor = AgentExecutor.from_config()
        
        # Test 1: Valid query should route through all success nodes
        valid_query = "What was the average temperature in Node 15 yesterday?"
        result = executor.execute(valid_query)
        
        trace_steps = [t['step'] for t in result.get('execution_trace', [])]
        
        print(f"\n✓ Valid query route:")
        print(f"  Steps: {' → '.join(trace_steps)}")
        
        # Should have gone through: interpret → validate → retrieve → execute → generate
        if result['success']:
            assert 'interpret_query' in trace_steps
            assert 'validate_task' in trace_steps
            assert 'retrieve_data' in trace_steps
            assert 'execute_analytics' in trace_steps
            assert 'generate_explanation' in trace_steps
            print("✓ Routed through all success nodes")
        
        # Test 2: Invalid query should route to error handling
        invalid_query = "What was the temperature in InvalidNode yesterday?"
        result = executor.execute(invalid_query)
        
        trace_steps = [t['step'] for t in result.get('execution_trace', [])]
        
        print(f"\n✓ Invalid query route:")
        print(f"  Steps: {' → '.join(trace_steps)}")
        
        # Should have gone to handle_error
        assert 'handle_error' in trace_steps
        assert result['success'] == False
        print("✓ Routed to error handler")
        
        return True
    
    except Exception as e:
        print(f"✗ Routing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Agent Module Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Executor Creation", test_agent_executor_creation),
        ("Simple Query Workflow", test_simple_query_workflow),
        ("Comparison Query Workflow", test_comparison_query_workflow),
        ("Aggregation Query Workflow", test_aggregation_query_workflow),
        ("Validation Error Handling", test_validation_error_handling),
        ("Execution Trace", test_execution_trace),
        ("State Structure", test_state_structure),
        ("Conditional Routing", test_conditional_routing),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        print(f"Running: {test_name}")
        print('=' * 60)
        
        try:
            if test_func():
                passed += 1
                print(f"\n✓ {test_name} PASSED")
            else:
                failed += 1
                print(f"\n✗ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"\n✗ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Total: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 60)
    
    sys.exit(0 if failed == 0 else 1)