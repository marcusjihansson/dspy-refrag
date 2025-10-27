#!/usr/bin/env python3
"""
Visualization script for comparative analysis results.
Generates plots for summary and detailed analysis insights.
"""

import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def generate_visualizations(summary_mode=False, save_plots=False):
    """
    Generate visualizations from analysis results.
    """
    results_dir = Path("../analysis_results")
    output_dir = Path("../analysis_results/plots")
    output_dir.mkdir(exist_ok=True)

    # Read main results
    main_file = results_dir / "comparative_analysis_results.json"
    if not main_file.exists():
        print("❌ Main analysis results not found. Run comparative_analysis.py first.")
        return

    with open(main_file, "r") as f:
        data = json.load(f)

    models = data.get("individual_model_data", [])
    metrics = data.get("comparative_metrics", {})
    integrated = data.get("integrated_analyses", {})

    if summary_mode:
        generate_summary_plots(models, metrics, integrated, output_dir, save_plots)
    else:
        generate_detailed_plots(models, metrics, integrated, output_dir, save_plots)

    print(f"✅ Visualizations generated in {output_dir}")

def generate_summary_plots(models, metrics, integrated, output_dir, save_plots):
    """Generate high-level summary plots."""
    # Performance improvements bar chart
    perf_comp = [m.get("performance_comparison", {}) for m in models]
    improvements = [(m.get("model_name", "Unknown").split("/")[-1],
                     p.get("latency_improvement_percent", 0),
                     p.get("token_reduction_percent", 0))
                    for m, p in zip(models, perf_comp) if p]

    if improvements:
        models, latency_imp, token_red = zip(*improvements)
        x = np.arange(len(models))
        width = 0.35

        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, latency_imp, width, label="Latency Improvement %")
        plt.bar(x + width/2, token_red, width, label="Token Reduction %")
        plt.xlabel("Model")
        plt.ylabel("Improvement %")
        plt.title("Summary: Performance Improvements by Model")
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.tight_layout()
        if save_plots:
            plt.savefig(output_dir / "summary_performance_improvements.png")
        plt.show()

    # Aggregate stats
    agg = metrics.get("aggregate_improvements", {})
    if agg:
        stats = []
        for metric, values in agg.items():
            if values:
                stats.append({"Metric": metric.replace("_", " ").title(), "Mean": values.get("mean", 0), "Min": values.get("min", 0), "Max": values.get("max", 0)})

        if stats:
            metrics_names = [s["Metric"] for s in stats]
            means = [s["Mean"] for s in stats]
            mins = [s["Min"] for s in stats]
            maxs = [s["Max"] for s in stats]

            x = np.arange(len(metrics_names))
            width = 0.25

            plt.figure(figsize=(10, 6))
            plt.bar(x - width, means, width, label="Mean")
            plt.bar(x, mins, width, label="Min")
            plt.bar(x + width, maxs, width, label="Max")
            plt.xlabel("Metric")
            plt.ylabel("Value")
            plt.title("Summary: Aggregate Statistics")
            plt.xticks(x, metrics_names)
            plt.legend()
            plt.tight_layout()
            if save_plots:
                plt.savefig(output_dir / "summary_aggregate_stats.png")
            plt.show()

def generate_detailed_plots(models, metrics, integrated, output_dir, save_plots):
    """Generate detailed plots."""
    # Individual model performance
    perf_data = []
    for m in models:
        name = m.get("model_name", "Unknown").split("/")[-1]
        simple = m.get("simple_rag", {})
        refrag = m.get("refrag", {})
        comp = m.get("performance_comparison", {})

        perf_data.append({
            "Model": name,
            "Simple Latency": simple.get("latency", {}).get("avg", 0),
            "Refrag Latency": refrag.get("latency", {}).get("avg", 0),
            "Simple Tokens": simple.get("tokens", {}).get("avg", 0),
            "Refrag Tokens": refrag.get("tokens", {}).get("avg", 0),
            "Latency Improvement %": comp.get("latency_improvement_percent", 0),
            "Token Reduction %": comp.get("token_reduction_percent", 0)
        })

    if perf_data:
        df = perf_data  # Already a list of dicts

        # Latency comparison
        models = [d["Model"] for d in df]
        simple_lat = [d["Simple Latency"] for d in df]
        refrag_lat = [d["Refrag Latency"] for d in df]

        x = np.arange(len(models))
        width = 0.35

        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, simple_lat, width, label="Simple Latency")
        plt.bar(x + width/2, refrag_lat, width, label="Refrag Latency")
        plt.xlabel("Model")
        plt.ylabel("Latency")
        plt.title("Detailed: Latency Comparison")
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.tight_layout()
        if save_plots:
            plt.savefig(output_dir / "detailed_latency_comparison.png")
        plt.show()

        # Token usage
        simple_tok = [d["Simple Tokens"] for d in df]
        refrag_tok = [d["Refrag Tokens"] for d in df]

        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, simple_tok, width, label="Simple Tokens")
        plt.bar(x + width/2, refrag_tok, width, label="Refrag Tokens")
        plt.xlabel("Model")
        plt.ylabel("Tokens")
        plt.title("Detailed: Token Usage Comparison")
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.tight_layout()
        if save_plots:
            plt.savefig(output_dir / "detailed_token_comparison.png")
        plt.show()

        # Token usage
        simple_tok = [d["Simple Tokens"] for d in df]
        refrag_tok = [d["Refrag Tokens"] for d in df]

        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, simple_tok, width, label="Simple Tokens")
        plt.bar(x + width/2, refrag_tok, width, label="Refrag Tokens")
        plt.xlabel("Model")
        plt.ylabel("Tokens")
        plt.title("Detailed: Token Usage Comparison")
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.tight_layout()
        if save_plots:
            plt.savefig(output_dir / "detailed_token_comparison.png")
        plt.show()

    # Temporal trends if available
    temporal = integrated.get("temporal_tracking", {}).get("temporal_summary", {})
    if temporal:
        dates = list(temporal.keys())
        models_count = [temporal[d].get("models_count", 0) for d in dates]

        plt.figure(figsize=(10, 6))
        plt.bar(dates, models_count)
        plt.xlabel("Date")
        plt.ylabel("Models Count")
        plt.title("Detailed: Models Analyzed Over Time")
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_plots:
            plt.savefig(output_dir / "detailed_temporal_trends.png")
        plt.show()

    # Model categorization
    cat = integrated.get("model_categorization", {}).get("categorization_summary", {})
    if cat:
        providers = list(cat.keys())
        counts = [cat[p].get("models_count", 0) for p in providers]

        plt.figure(figsize=(10, 6))
        plt.pie(counts, labels=providers, autopct='%1.1f%%')
        plt.title("Detailed: Model Distribution by Provider")
        plt.tight_layout()
        if save_plots:
            plt.savefig(output_dir / "detailed_model_categories.png")
        plt.show()

    # Model categorization
    cat = integrated.get("model_categorization", {}).get("categorization_summary", {})
    if cat:
        providers = list(cat.keys())
        counts = [cat[p].get("models_count", 0) for p in providers]

        plt.figure(figsize=(10, 6))
        plt.pie(counts, labels=providers, autopct='%1.1f%%')
        plt.title("Detailed: Model Distribution by Provider")
        plt.tight_layout()
        if save_plots:
            plt.savefig(output_dir / "detailed_model_categories.png")
        plt.show()

if __name__ == "__main__":
    summary_mode = "--summary" in sys.argv
    save_plots = "--save" in sys.argv

    generate_visualizations(summary_mode=summary_mode, save_plots=save_plots)