# Analysis Directory

This directory contains tools and results for analyzing and comparing the performance of REFRAG (Retrieval-Augmented Fragmenting) against traditional Simple RAG across multiple language models.

## Overview

The analysis suite processes benchmark results from various models to extract key performance metrics, enabling detailed comparisons between different retrieval approaches. It includes scripts for core analysis, additional insights (quality, cost, temporal, statistical, categorization), and visualizations.

## Subdirectories

- **comparative_analysis/**: Contains scripts for running the analysis.
  - Run `python analyze.py --detailed --save` for integrated results.
  - See comparative_analysis/README.md for detailed usage.

- **analysis_results/**: Contains output files from the analyses.
  - JSON files: Detailed results (e.g., comparative_analysis_results.json, quality_analysis_results.json).
  - Text reports: Timestamped summaries and detailed analyses.
  - Plots: PNG visualizations (e.g., performance comparisons, temporal trends).

## Usage

1. Navigate to `comparative_analysis/` and run the analysis scripts.
2. View results in `analysis_results/`.
3. Use `viz.py` for visualizations.

## Outputs

- Comprehensive performance comparisons.
- Quality metrics, cost estimates, temporal trends, statistical tests, and model categorizations.
- Visual plots for better interpretation.

For more details, see the README.md in comparative_analysis/.