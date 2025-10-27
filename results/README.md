# Results Directory

This directory contains benchmark results organized by model and timestamp.

## Structure

```
results/
├── {model_name}/
│   ├── {timestamp}_benchmark_results.json    # Raw benchmark data
│   ├── {timestamp}_comparison_plot.png       # Visualization
│   └── latest_results.json                   # Symlink to most recent
└── summary/
    ├── all_models_comparison.json            # Cross-model comparison
    └── latest_comparison.png                 # Latest visualization
```

## File Naming Convention

- **Model names**: Sanitized for filesystem (e.g., `openai_gpt-3.5-turbo`)
- **Timestamps**: ISO format `YYYY-MM-DD_HH-MM-SS`
- **Result files**: `{timestamp}_benchmark_results.json`
- **Plot files**: `{timestamp}_comparison_plot.png`

## Contents

### Benchmark Results JSON
Contains complete benchmark data including:
- Individual query results for both Simple RAG and REFRAG
- Aggregated statistics (latency, tokens, context metrics)
- Similarity analysis between approaches
- Configuration metadata (model, k, budget, etc.)

### Comparison Plots
Visual comparison showing:
- Latency differences
- Token usage efficiency  
- Context size impact
- Performance distributions