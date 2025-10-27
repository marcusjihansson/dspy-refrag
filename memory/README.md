# Memory-Enhanced REFRAG Extension

This directory contains an extension to the core dspy_refrag system, demonstrating how to add memory capabilities for enhanced reasoning, optimization, and quick fact retrieval. The extension is designed as an advanced example for users who want to build conversational AI with persistent memory.

## Overview

The memory-enhanced REFRAG builds on the core dspy_refrag (located in `src/dspy_refrag/`) by adding Weaviate-based memory storage. It includes:

- **Conversation Memory**: Store and retrieve past interactions for context-aware responses.
- **Quick Facts Caching**: Fast retrieval of common answers to reduce latency.
- **Optimization Memory**: Collect high-quality Q&A pairs for training optimizers like MIPRO.
- **Reasoning Memory**: Maintain reasoning paths for iterative refinement.

This extension is **not part of the core dspy_refrag** but serves as a modular add-on, allowing users to enhance the base REFRAG with memory without modifying the core implementation.

## Integration with Core dspy_refrag

The extension imports and extends core classes like `REFRAGModule`, `SimpleRetriever`, and `Sensor` from `src/dspy_refrag/`. To use:

1. Ensure core dspy_refrag is available (e.g., via `sys.path.append('src')`).
2. Import core components in your scripts.
3. Initialize memory components (e.g., WeaviateMemory) and pass them to MemoryEnhancedREFRAG.

### Key Components

- **MemoryEnhancedREFRAG** (in `memory_refrag/memory_enhanced_refrag.py`): The main extension class that inherits from core's REFRAGModule and adds memory features.
- **MemoryStore** (in `memory_solution_in_refrag/memory_store.py`): General interface and simple implementations (in-memory, local JSON).
- **QuickFactsRetrieval**, **OptimizationMemory**, **ReasoningWithMemory**: Specialized memory modules for different use cases.

## Setup

1. **Install Dependencies**:
   - Core dspy_refrag requirements (e.g., dspy, numpy).
   - Weaviate: `pip install weaviate-client`.
   - Optional: Ollama for embeddings (if using make_ollama_embedder).

2. **Run Weaviate**:
   - For the memory extension: `docker-compose -f memory/docker-compose.yml up`.
   - This sets up Weaviate with text2vec-transformers for memory vectorization.
   - Or connect to an existing instance.

3. **Configure Paths**:
   - In your script, add `import sys; sys.path.append('src')` to import core modules.

## Usage Example

```python
import sys
sys.path.append('src')  # For core dspy_refrag imports

from dspy_refrag.refrag import REFRAGModule
from dspy_refrag.retriever import SimpleRetriever
from dspy_refrag.sensor import Sensor
from memory.memory_refrag.memory_enhanced_refrag import MemoryEnhancedREFRAG
from memory.memory_solution_in_refrag.memory_store import InMemoryStore

# Initialize core components
retriever = SimpleRetriever()
sensor = Sensor()
memory = InMemoryStore()

# Create memory-enhanced REFRAG
enhanced_refrag = MemoryEnhancedREFRAG(
    memory=memory,
    retriever=retriever,
    session_id="user_session_1"
)

# Use it
result = enhanced_refrag("What is the main topic?")
print(result.answer)
```

## Advanced Features

- **Iterative Refinement**: The extension refines queries based on memory context.
- **Caching**: Quick facts are cached for low-latency responses.
- **Optimization**: Store examples for training (e.g., MIPRO).

## Notes

- This is an extension example; modify as needed for your use case.
- For production, ensure Weaviate is properly configured and scaled.
- Fallback: If memory is disabled, it behaves like core REFRAG.

For more details, see the README in `memory_solution_in_refrag/`.