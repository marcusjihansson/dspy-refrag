# Comparative Analysis Directory

This directory contains tools and scripts for analyzing and comparing the performance of REFRAG (Retrieval-Augmented Fragmenting) against traditional Simple RAG across multiple language models.

## Overview

The tools in this directory process benchmark results from various models to extract and analyze key performance metrics, enabling comparisons between different retrieval approaches.

## Key Components

- **Scripts:**
  - `comparative_analysis.py`: Core script for processing and analyzing results from model directories.
  - `analyze.py`: Comprehensive command-line interface for running analysis with options for summary, detailed, and JSON outputs. Now integrates additional analyses like quality, cost, temporal, statistical, and categorization.
  - `generate_summary.py`: Script for generating quick summaries of the analysis, including insights from all integrated analyses.
  - `quality_analysis.py`: Analyzes answer quality using similarity metrics (exact, jaccard, len_ratio).
  - `cost_modeling.py`: Estimates costs based on token usage and latency.
  - `temporal_tracking.py`: Tracks performance metrics over time using timestamps.
  - `statistical_testing.py`: Performs statistical tests on metrics between Simple RAG and REFRAG.
  - `model_categorization.py`: Categorizes models by provider and compares performance within categories.
  - `viz.py`: Generates advanced visualizations for summary and detailed analysis results.
  - `viz.py`: Generates advanced visualizations for summary and detailed analysis results.

- **Output Files:**
  - `comparative_analysis_results.json`: Complete analysis results including aggregate statistics and integrated analyses.
  - `quality_analysis_results.json`: Quality metrics and insights.
  - `cost_modeling_results.json`: Cost estimates and savings.
  - `temporal_tracking_results.json`: Temporal trends in performance.
  - `statistical_testing_results.json`: Statistical test results.
  - `model_categorization_results.json`: Categorized model performance.
  - Timestamped reports (e.g., `summary_report_YYYY-MM-DD_HH-MM-SS.txt`, `detailed_analysis_YYYY-MM-DD_HH-MM-SS.txt`): Generated summaries and detailed analyses with integrated insights.
  - Custom output files when specified.

## Usage

Run the analysis tools using the following commands:

### Basic Analysis

```bash
python comparative_analysis.py
```

### Generate Summary

```bash
python generate_summary.py
python generate_summary.py --save  # Save to timestamped file
python generate_summary.py --output=filename.txt  # Save to custom file
```

### Complete Analysis Interface

```bash
python analyze.py  # Default summary
python analyze.py --detailed  # Detailed analysis
python analyze.py --json  # JSON export
python analyze.py --save  # Save output to timestamped file
python analyze.py --output=filename  # Save to custom file
```

### Additional Analyses

Run individual analysis scripts to generate specific insights:

```bash
python quality_analysis.py
python cost_modeling.py
python temporal_tracking.py
python statistical_testing.py
python model_categorization.py
```

For integrated results, run:

```bash
python analyze.py --detailed --save  # Includes all analyses
python generate_summary.py --save  # Summary with all insights
```

### Visualizations

Generate plots to visualize analysis results:

```bash
python viz.py --summary  # High-level summary plots
python viz.py --detailed  # Detailed plots
python viz.py --save  # Save plots to files
```

### Generate Summary

```bash
python generate_summary.py
python generate_summary.py --save  # Save to timestamped file
python generate_summary.py --output=filename.txt  # Save to custom file
```

### Complete Analysis Interface

```bash
python analyze.py  # Default summary
python analyze.py --detailed  # Detailed analysis
python analyze.py --json  # JSON export
python analyze.py --save  # Save output to timestamped file
python analyze.py --output=filename  # Save to custom file
```

### Additional Analyses

Run individual analysis scripts to generate specific insights:

```bash
python quality_analysis.py
python cost_modeling.py
python temporal_tracking.py
python statistical_testing.py
python model_categorization.py
```

For integrated results, run:

```bash
python analyze.py --detailed --save  # Includes all analyses
python generate_summary.py --save  # Summary with all insights
```

### Additional Analyses

Run individual analysis scripts to generate specific insights:

```bash
python quality_analysis.py
python cost_modeling.py
python temporal_tracking.py
python statistical_testing.py
python model_categorization.py
```

For integrated results, run:

```bash
python analyze.py --detailed --save  # Includes all analyses
python generate_summary.py --save  # Summary with all insights
```

### Generate Summary

```bash
python generate_summary.py
python generate_summary.py --save  # Save to timestamped file
python generate_summary.py --output=filename.txt  # Save to custom file
```

### Complete Analysis Interface

```bash
python analyze.py  # Default summary
python analyze.py --detailed  # Detailed analysis
python analyze.py --json  # JSON export
python analyze.py --save  # Save output to timestamped file
python analyze.py --detailed --save #Save detailed output to timestamped file§
python analyze.py --output=filename  # Save to custom file
```
