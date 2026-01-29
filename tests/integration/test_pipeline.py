import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm import OllamaLLM, SystemContext
from data import SensorDataRepository, LLMDataBridge
from analytics import get_registry

# Initialize components
llm = OllamaLLM.from_config()
repository = SensorDataRepository.from_config()
bridge = LLMDataBridge(repository)
registry = get_registry()

# Get system context
context_dict = bridge.get_system_context()
context = SystemContext(**context_dict)

# Extract intent
query = "What was the average temperature in Node 15 last week?"
task_spec = llm.extract_intent(query, context)

# Get data
data = bridge.execute_task(task_spec)

# Execute analytics
tool = registry.get_tool_by_operation(task_spec.operation.value)
result = tool.execute(data, operation=task_spec.operation.value)

# Generate explanation
explanation = llm.explain_results(query, task_spec, [result.model_dump()])

print(explanation)