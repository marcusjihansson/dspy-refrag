# Memory Solution Components for dspy_refrag Extension

This directory contains the core memory components used in the memory-enhanced REFRAG extension. These modules provide persistent storage and retrieval capabilities using Weaviate, integrated with the core dspy_refrag system.

## Components Overview

### MemoryStore (memory_store.py)
- **Purpose**: Central memory storage class for interactions, using Weaviate as the vector database.
- **Features**:
  - Store Q&A interactions with metadata (e.g., reasoning, context, scores).
  - Retrieve similar past interactions for context.
  - Support for different memory types (e.g., reasoning, optimization, quick_facts).
- **Usage**: Initialize with a Weaviate client and collection name. Handles schema creation automatically.

### QuickFactsRetrieval (low_latency_memory.py)
- **Purpose**: Low-latency Q&A for quick retrieval of common answers.
- **Features**:
  - Caches high-confidence answers in memory.
  - Falls back to document retrieval if not cached.
  - Uses similarity thresholds for direct reuse.
- **Integration**: Extends dspy.Module and works with a MemoryStore (in-memory or local JSON).

### OptimizationMemory (optimization_memory.py)
- **Purpose**: Store high-quality Q&A pairs for training optimizers (e.g., MIPRO).
- **Features**:
  - Collects examples with scores and ground truth.
  - Retrieves filtered examples for optimization.
- **Integration**: Queries Weaviate for optimization-specific data.

### ReasoningWithMemory (reasoning_memory.py)
- **Purpose**: Maintain reasoning paths for iterative refinement.
- **Features**:
  - Stores and retrieves reasoning chains.
  - Provides context from past reasoning for new queries.
- **Integration**: Uses a MemoryStore for storage and retrieval.

## Integration with Core dspy_refrag

These components are designed to extend core dspy_refrag (from `src/dspy_refrag/`) without modifying it. They use core's data structures (e.g., Passage) where possible.

- **Imports**: Components import from core (e.g., `from src.dspy_refrag.common import Passage`).
- **Compatibility**: Memory storage aligns with core's retriever and sensor interfaces.
- **Extension Point**: Used in `memory_refrag/memory_enhanced_refrag.py` to enhance REFRAGModule.

## Setup and Dependencies

1. **Weaviate**: Required for vector storage. Run via Docker or connect to a hosted instance.
2. **Python Packages**:
   - (optional) Any vector DB client if you implement a vectorized MemoryStore backend.
   - `dspy`: For DSPy modules.
   - `numpy`: For vector operations.
3. **Configuration**:
   - Set Weaviate URL (default: http://localhost:8080).
   - Optional: Configure embeddings (e.g., via Ollama).

## Usage Examples

### Basic Memory Storage
```python
from memory.memory_solution_in_refrag.memory_store import InMemoryStore
import weaviate

client = weaviate.Client("http://localhost:8080")
memory = WeaviateMemory(client)

# Store an interaction
memory.store_interaction(
    question="What is AI?",
    answer="Artificial Intelligence is...",
    session_id="user_1",
    memory_type="reasoning"
)

# Retrieve similar
similar = memory.retrieve_similar("What is machine learning?", limit=3)
```

### Quick Facts
```python
from memory.memory_solution_in_refrag.low_latency_memory import QuickFactsRetrieval

quick_facts = QuickFactsRetrieval(memory)
result = quick_facts("Common question", use_cache=True)
```

## Best Practices

- **Memory Types**: Use specific types (e.g., "reasoning", "quick_facts") to organize data.
- **Scores**: Add relevance scores for filtering high-quality examples.
- **Session IDs**: Use for multi-user or session-based isolation.
- **Performance**: Monitor Weaviate for large-scale use; consider indexing strategies.

## Notes

- These are extension components; they assume core dspy_refrag is available.
- For testing, use the example in `memory_refrag/run.py`.
- Contributions: Extend these for custom memory strategies while maintaining core compatibility.

See `memory/README.md` for overall extension documentation.