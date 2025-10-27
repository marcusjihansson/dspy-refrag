# Memory in REFRAG (DB-agnostic)

This directory contains a generic, database-agnostic memory integration for REFRAG.
It demonstrates how prior QnA can be stored and reused for:

- A) Reasoning: previous QnA augment future reasoning paths
- B) Optimization: store high-quality QnA as examples for later improvement
- C) Low-latency QnA: quick answers using a lightweight local memory cache

## Key Components

- `memory_enhanced_refrag.py` — Example REFRAG module wired with a pluggable `MemoryStore`.
- `../memory_solution_in_refrag/memory_store.py` — Simple memory backends:
  - `InMemoryStore`: in-process, thread-safe memory with lexical search
  - `LocalJSONStore`: file-backed memory for local persistence
- `../memory_solution_in_refrag/reasoning_memory.py` — ReasoningWithMemory (stores QnA)
- `../memory_solution_in_refrag/optimization_memory.py` — Optimization memory examples
- `../memory_solution_in_refrag/low_latency_memory.py` — QuickFactsRetrieval (low-latency QnA)
- `../memory_solution_in_refrag/utility.py` — consolidation and decay helpers

## Quick start

1) (Optional) Configure DSPy with OpenRouter free model:

```bash
export OPENROUTER_API_KEY=YOUR_KEY
```

2) Run the example:

```bash
python3 memory/memory_refrag/memory_enhanced_refrag.py
```

You should see an initial answer based on simple retrieval, followed by a subsequent
call benefiting from low-latency memory.

## Using memory in your REFRAG pipelines

- Provide any `MemoryStore` instance to `MemoryEnhancedREFRAG` (or your own modules):

```python
from memory.memory_solution_in_refrag.memory_store import InMemoryStore
from memory.memory_refrag.memory_enhanced_refrag import MemoryEnhancedREFRAG
from dspy_refrag.retriever import SimpleRetriever

mem = InMemoryStore()
retriever = SimpleRetriever()
system = MemoryEnhancedREFRAG(memory=mem, retriever=retriever, session_id="demo")
result = system(question="What animal is loyal?")
print(result.answer)
```

- For optimization (B): save high-quality QnA as MemoryRecord(type="optimization") using the store.
- For low-latency (C): use `QuickFactsRetrieval` or similar to populate and check the memory
  before hitting your full retrieval/generation stack.

## Consolidation and decay

Use the utilities to keep memory clean and relevant:

```python
from memory.memory_solution_in_refrag.utility import consolidate_qna, apply_decay

# Consolidate duplicates across recent QnA
recent = mem.get_recent_qna(session_id="demo", limit=50)
merged = consolidate_qna(recent)
for m in merged:
    mem.save(m)

# Apply time-based decay to scores
apply_decay(mem, session_id="demo", half_life_days=30.0)
```

This setup showcases memory as a first-class concept in REFRAG without tying you to any specific database.
