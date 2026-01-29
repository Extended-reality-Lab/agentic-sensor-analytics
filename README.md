# Agentic Sensor Analytics

An agentic AI system for natural-language analytics of smart building sensor data.

<div style="text-align: center;">
  <img src="docs/assets/ui-example.png" alt="UI" width="650"/>
</div>


## Overview

This project enables users to query building sensor data using natural language questions. The system combines a local LLM for intent extraction with deterministic analytics tools to provide accurate, verifiable results.

## Architecture

The system uses a **five-layer agentic architecture**:

1. **User Query** → Natural language input
2. **Intent Interpretation** → LLM extracts structured task specification
3. **Agent Planning** → Validates parameters and creates execution plan
4. **Tool Execution** → Deterministic analytics compute results
5. **Result Explanation** → LLM converts results to natural language

This separation ensures reliable computations while maintaining natural language interaction.

## Key Features

- **Natural Language Interface**: Ask questions in plain English
- **Deterministic Analytics**: All calculations are verifiable and reproducible
- **Local LLM**: Uses Ollama for privacy-preserving inference
- **Real Sensor Data**: Connects to SMT Analytics API for Peavy Hall building data
- **Full Transparency**: Complete execution traces for every query

## Quick Start

### Prerequisites
- Python 3.10+
- Ollama installed and running
- Access to SMT Analytics API

### Installation
```bash
# Clone repository
git clone <repository-url>
cd agentic-sensor-analytics

# Install dependencies
pip install -r requirements.txt

# Pull LLM model
ollama pull llama3.1:8b
```

### Configuration
Create .env file from .env.example with SMT API credentials.

## Author
Sean Clayton  