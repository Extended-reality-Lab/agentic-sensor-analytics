#!/bin/sh
echo "Waiting for Ollama service to be ready..."
sleep 10

while ! ollama list >/dev/null 2>&1; do
  echo "Waiting for Ollama API..."
  sleep 3
done

echo "Ollama is ready!"

MODEL=$(awk '/model_name:/ {print $2}' /llm_config.yaml | tr -d '\r\n')
echo "Config file requests model: $MODEL"

if ollama pull "$MODEL"; then
  echo "Model $MODEL successfully pulled!"
else
  echo "Failed to pull model $MODEL"
  exit 1
fi