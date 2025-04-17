# config.py
import os

# Configuration for Ollama connection
# Default values can be overridden by environment variables
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:instruct") # Or "mistral:instruct", "phi3:instruct" etc.
OLLAMA_REQUEST_TIMEOUT = int(os.getenv("OLLAMA_REQUEST_TIMEOUT", 120)) # Timeout in seconds

# You can add other configurations here if needed
print("--- config.py loaded ---")
print(f"OLLAMA_BASE_URL inside config.py: {OLLAMA_BASE_URL}")