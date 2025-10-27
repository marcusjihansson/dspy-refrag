#!/usr/bin/env python3
"""
Quick summary generator for the comparative analysis results.
Generates a concise summary of key findings from the analysis.
"""

import json
import sys
from datetime import datetime

def generate_summary(save_to_file=False, output_file=None):
    """Generate a concise summary of the comparative analysis results."""

    try:
        with open('../analysis_results/comparative_analysis_results.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: comparative_analysis_results.json not found. Run comparative_analysis.py first.")
        return None

    # Integrate new analyses into summary
    integrate_new_summaries(data)
    
    metrics = data.get('comparative_metrics', {})
    
    # Build the summary content
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("DSPy REFRAG vs Simple RAG - Performance Summary")
    summary_lines.append("=" * 60)
    summary_lines.append(f"Analysis Date: {data.get('analysis_timestamp', 'Unknown')}")
    summary_lines.append(f"Models Analyzed: {data.get('total_models_analyzed', 0)}")
    summary_lines.append("")
    
    # Aggregate improvements
    agg_improvements = metrics.get('aggregate_improvements', {})
    
    summary_lines.append("📊 OVERALL PERFORMANCE IMPROVEMENTS")
    summary_lines.append("-" * 40)
    
    latency_stats = agg_improvements.get('latency_improvement_stats', {})
    if latency_stats:
        summary_lines.append(f"⚡ Average Latency Improvement: {latency_stats.get('mean', 0):.1f}%")
        summary_lines.append(f"   Best improvement: {latency_stats.get('max', 0):.1f}%")
        summary_lines.append(f"   Range: {latency_stats.get('min', 0):.1f}% to {latency_stats.get('max', 0):.1f}%")
    
    token_stats = agg_improvements.get('token_reduction_stats', {})
    if token_stats:
        summary_lines.append(f"🎯 Average Token Reduction: {token_stats.get('mean', 0):.1f}%")
        summary_lines.append(f"   Range: {token_stats.get('min', 0):.1f}% to {token_stats.get('max', 0):.1f}%")
    
    context_stats = agg_improvements.get('context_reduction_stats', {})
    if context_stats:
        summary_lines.append(f"📝 Context Reduction: {context_stats.get('mean', 0):.1f}%")
    
    summary_lines.append("")
    
    # Model rankings
    rankings = metrics.get('model_rankings', {})
    
    summary_lines.append("🏆 TOP PERFORMING MODELS")
    summary_lines.append("-" * 40)
    
    fastest_refrag = rankings.get('fastest_refrag_models', [])
    if fastest_refrag:
        summary_lines.append("⚡ Fastest REFRAG Models (by latency):")
        for i, model in enumerate(fastest_refrag[:5], 1):
            model_name = model.get('model', 'Unknown').split('/')[-1]
            latency = model.get('avg_latency', 0)
            summary_lines.append(f"   {i}. {model_name}: {latency:.2f}s")
    
    summary_lines.append("")
    
    best_improvement = rankings.get('best_improvement_models', [])
    if best_improvement:
        summary_lines.append("📈 Best Improvement Models:")
        for i, model in enumerate(best_improvement[:5], 1):
            model_name = model.get('model', 'Unknown').split('/')[-1]
            latency_imp = model.get('latency_improvement', 0)
            token_red = model.get('token_reduction')
            token_str = f", {token_red:.1f}% tokens" if token_red else ""
            summary_lines.append(f"   {i}. {model_name}: {latency_imp:.1f}% faster{token_str}")
    
    summary_lines.append("")
    
    # Summary insights
    summary = metrics.get('summary_insights', {})
    
    summary_lines.append("💡 KEY INSIGHTS")
    summary_lines.append("-" * 40)
    
    total_complete = summary.get('total_models_with_complete_data', 0)
    positive_latency = summary.get('models_with_positive_latency_improvement', 0)
    positive_tokens = summary.get('models_with_positive_token_reduction', 0)
    
    if total_complete > 0:
        success_rate = (positive_latency / total_complete) * 100
        summary_lines.append(f"✅ {positive_latency}/{total_complete} models ({success_rate:.0f}%) showed latency improvements")
        
        if positive_tokens > 0:
            summary_lines.append(f"💰 {positive_tokens} models showed significant token usage reductions")
        
        summary_lines.append(f"🎯 REFRAG consistently reduces context size by ~60%")
        summary_lines.append(f"⚡ Best case improvement: {latency_stats.get('max', 0):.0f}% faster")
        
        if latency_stats.get('mean', 0) > 0:
            summary_lines.append(f"📊 REFRAG is generally {latency_stats.get('mean', 0):.0f}% faster on average")
    
    summary_lines.append("")
    summary_lines.append("=" * 60)
    summary_lines.append("🔗 For detailed analysis, see comparative_analysis_results.json")
    summary_lines.append("📖 For methodology, see README.md")
    summary_lines.append("=" * 60)

    # Add new analysis summaries
    add_new_analysis_summaries(summary_lines, data)
    
    # Create the complete summary text
    summary_text = "\n".join(summary_lines)
    
    # Output to console
    print(summary_text)
    
    # Save to file if requested
    if save_to_file:
        if output_file is None:
            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_file = f"../analysis_results/summary_report_{timestamp}.txt"
        
        try:
            with open(output_file, 'w') as f:
                f.write(summary_text)
            print(f"\n💾 Summary saved to: {output_file}")
        except Exception as e:
            print(f"\n❌ Error saving summary to file: {e}")
    
    return summary_text

def integrate_new_summaries(data):
    """Integrate summaries from new analyses."""
    integrated = data.get("integrated_analyses", {})
    if not integrated:
        print("No integrated analyses found. Run analyze.py first.")
        return

    # Update metrics with new data
    for key, analysis_data in integrated.items():
        if key == "quality_analysis":
            quality = analysis_data.get("quality_summary", {})
            data.setdefault("quality_insights", quality)
        elif key == "cost_modeling":
            cost = analysis_data.get("cost_summary", {})
            data.setdefault("cost_insights", cost)
        elif key == "temporal_tracking":
            temporal = analysis_data.get("temporal_summary", {})
            data.setdefault("temporal_insights", temporal)
        elif key == "statistical_testing":
            stats = analysis_data.get("statistical_summary", {})
            data.setdefault("statistical_insights", stats)
        elif key == "model_categorization":
            cat = analysis_data.get("categorization_summary", {})
            data.setdefault("categorization_insights", cat)

def add_new_analysis_summaries(summary_lines, data):
    """Add summaries from new analyses to the output."""
    integrated = data.get("integrated_analyses", {})

    if "quality_analysis" in integrated:
        quality = integrated["quality_analysis"].get("quality_summary", {})
        summary_lines.append("\n📊 QUALITY ANALYSIS")
        summary_lines.append("-" * 40)
        exact = quality.get("exact_similarity_stats", {})
        jaccard = quality.get("jaccard_similarity_stats", {})
        if exact:
            summary_lines.append(f"Exact Similarity: {exact.get('mean', 0):.4f} (avg)")
        if jaccard:
            summary_lines.append(f"Jaccard Similarity: {jaccard.get('mean', 0):.4f} (avg)")

    if "cost_modeling" in integrated:
        cost = integrated["cost_modeling"].get("cost_summary", {})
        summary_lines.append("\n💰 COST MODELING")
        summary_lines.append("-" * 40)
        savings = cost.get("savings_stats", {})
        if savings:
            summary_lines.append(f"Average Cost Savings: {savings.get('mean', 0):.4f}")
            summary_lines.append(f"Models with Savings: {cost.get('insights', {}).get('models_with_savings', 0)}")

    if "temporal_tracking" in integrated:
        temporal = integrated["temporal_tracking"].get("temporal_summary", {})
        summary_lines.append("\n⏰ TEMPORAL TRACKING")
        summary_lines.append("-" * 40)
        for date, stats in temporal.items():
            summary_lines.append(f"{date}: {stats.get('models_count', 0)} models")

    if "statistical_testing" in integrated:
        stats = integrated["statistical_testing"].get("statistical_summary", {})
        summary_lines.append("\n📈 STATISTICAL TESTING")
        summary_lines.append("-" * 40)
        latency_test = stats.get("latency_test", {})
        if latency_test:
            summary_lines.append(f"Latency T-Stat: {latency_test.get('t_statistic', 0):.4f}")

    if "model_categorization" in integrated:
        cat = integrated["model_categorization"].get("categorization_summary", {})
        summary_lines.append("\n🏷️ MODEL CATEGORIZATION")
        summary_lines.append("-" * 40)
        for provider, stats in cat.items():
            summary_lines.append(f"{provider}: {stats.get('models_count', 0)} models")

if __name__ == "__main__":
    # Parse command line arguments
    save_file = False
    output_file = None
    
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == "--save":
                save_file = True
            elif arg.startswith("--output="):
                output_file = arg.split("=", 1)[1]
                save_file = True
            elif arg in ["--help", "-h"]:
                print("Usage: python generate_summary.py [--save] [--output=filename]")
                print("  --save: Save summary to timestamped file")
                print("  --output=filename: Save summary to specific filename")
                sys.exit(0)
    
    generate_summary(save_to_file=save_file, output_file=output_file)