import json
import os
from pathlib import Path
from datetime import datetime

def analyze_latest_results():
    """
    Analyze the latest_results.json files from each model directory in the results folder.
    
    This function extracts key performance metrics and comparison data to understand:
    1. Performance differences between Simple RAG and REFRAG approaches
    2. Model-specific performance characteristics
    3. Efficiency metrics (latency, tokens, context usage)
    4. Similarity and quality metrics
    """
    
    results_dir = Path("../../results")  # Adjusted path from analysis/comparative_analysis/
    output_file = "../analysis_results/comparative_analysis_results.json"
    
    models_data = []
    comparative_metrics = {
        "performance_summary": {},
        "refrag_vs_simple_rag": {},
        "model_rankings": {}
    }
    
    # Process each model directory
    for model_dir in os.listdir(results_dir):
        model_path = results_dir / model_dir
        if model_path.is_dir():
            latest_results_path = model_path / "latest_results.json"
            if latest_results_path.exists():
                try:
                    with open(latest_results_path, "r") as f:
                        data = json.load(f)
                    
                    # Extract metadata
                    metadata = data.get("metadata", {})
                    model_name = metadata.get("model_name", "unknown")
                    safe_model_name = metadata.get("safe_model_name", model_dir)
                    timestamp = metadata.get("timestamp", "unknown")
                    
                    # Extract benchmark data
                    benchmark_data = data.get("benchmark_data", {})
                    similarity = benchmark_data.get("similarity", {})
                    summary = benchmark_data.get("summary", {})
                    
                    # Process Simple RAG and REFRAG data
                    simple_rag = summary.get("Simple RAG", {})
                    refrag = summary.get("REFRAG", {})
                    
                    model_entry = {
                        "model_name": model_name,
                        "safe_model_name": safe_model_name,
                        "timestamp": timestamp,
                        "similarity_metrics": similarity,
                        "simple_rag": extract_metrics(simple_rag),
                        "refrag": extract_metrics(refrag),
                        "performance_comparison": {}
                    }
                    
                    # Calculate performance comparisons
                    if simple_rag and refrag:
                        model_entry["performance_comparison"] = calculate_performance_comparison(simple_rag, refrag)
                    
                    models_data.append(model_entry)
                    
                except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                    print(f"Error processing {latest_results_path}: {e}")
                    continue
    
    # Generate comparative analysis
    comparative_metrics = generate_comparative_analysis(models_data)
    
    # Prepare final output
    analysis_results = {
        "analysis_timestamp": datetime.now().isoformat(),
        "total_models_analyzed": len(models_data),
        "individual_model_data": models_data,
        "comparative_metrics": comparative_metrics
    }
    
    # Save results
    with open(output_file, "w") as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"Analysis complete! Results saved to {output_file}")
    print(f"Analyzed {len(models_data)} models")
    
    return analysis_results

def extract_metrics(rag_data):
    """Extract and structure metrics from RAG data."""
    if not rag_data:
        return {}
    
    return {
        "latency": rag_data.get("latency", {}),
        "tokens": rag_data.get("tokens", {}),
        "context_chars": rag_data.get("context_chars", {}),
        "prompt_chars": rag_data.get("prompt_chars", {}),
        "retrieved": rag_data.get("retrieved", {}),
        "selected": rag_data.get("selected", {})
    }

def calculate_performance_comparison(simple_rag, refrag):
    """Calculate performance improvements/differences between REFRAG and Simple RAG."""
    comparison = {}
    
    # Latency comparison
    simple_latency = simple_rag.get("latency", {}).get("avg")
    refrag_latency = refrag.get("latency", {}).get("avg")
    
    if simple_latency and refrag_latency:
        latency_improvement = ((simple_latency - refrag_latency) / simple_latency) * 100
        comparison["latency_improvement_percent"] = round(latency_improvement, 2)
        comparison["latency_ratio"] = round(refrag_latency / simple_latency, 3)
    
    # Token usage comparison
    simple_tokens = simple_rag.get("tokens", {}).get("avg")
    refrag_tokens = refrag.get("tokens", {}).get("avg")
    
    if simple_tokens and refrag_tokens:
        token_reduction = ((simple_tokens - refrag_tokens) / simple_tokens) * 100
        comparison["token_reduction_percent"] = round(token_reduction, 2)
        comparison["token_ratio"] = round(refrag_tokens / simple_tokens, 3)
    
    # Context efficiency
    simple_context = simple_rag.get("context_chars", {}).get("avg")
    refrag_context = refrag.get("context_chars", {}).get("avg")
    
    if simple_context and refrag_context:
        context_reduction = ((simple_context - refrag_context) / simple_context) * 100
        comparison["context_reduction_percent"] = round(context_reduction, 2)
    
    # Selection efficiency (documents selected vs retrieved)
    refrag_selected = refrag.get("selected", {}).get("avg", 0)
    refrag_retrieved = refrag.get("retrieved", {}).get("avg", 1)
    
    if refrag_retrieved > 0:
        selection_efficiency = (refrag_selected / refrag_retrieved) * 100
        comparison["selection_efficiency_percent"] = round(selection_efficiency, 2)
    
    return comparison

def generate_comparative_analysis(models_data):
    """Generate comprehensive comparative analysis across all models."""
    
    # Filter out models without both Simple RAG and REFRAG data
    complete_models = [m for m in models_data if m.get("simple_rag") and m.get("refrag")]
    
    if not complete_models:
        return {"error": "No models found with both Simple RAG and REFRAG data"}
    
    # Performance rankings
    refrag_latency_ranking = sorted(complete_models, 
                                   key=lambda x: x.get("refrag", {}).get("latency", {}).get("avg", float('inf')))
    
    simple_rag_latency_ranking = sorted(complete_models,
                                       key=lambda x: x.get("simple_rag", {}).get("latency", {}).get("avg", float('inf')))
    
    # Calculate aggregate statistics
    latency_improvements = []
    token_reductions = []
    context_reductions = []
    
    for model in complete_models:
        perf_comp = model.get("performance_comparison", {})
        
        if "latency_improvement_percent" in perf_comp:
            latency_improvements.append(perf_comp["latency_improvement_percent"])
        
        if "token_reduction_percent" in perf_comp:
            token_reductions.append(perf_comp["token_reduction_percent"])
        
        if "context_reduction_percent" in perf_comp:
            context_reductions.append(perf_comp["context_reduction_percent"])
    
    def calculate_stats(values):
        if not values:
            return {}
        return {
            "mean": round(sum(values) / len(values), 2),
            "min": min(values),
            "max": max(values),
            "count": len(values)
        }
    
    return {
        "aggregate_improvements": {
            "latency_improvement_stats": calculate_stats(latency_improvements),
            "token_reduction_stats": calculate_stats(token_reductions),
            "context_reduction_stats": calculate_stats(context_reductions)
        },
        "model_rankings": {
            "fastest_refrag_models": [
                {
                    "model": m["model_name"],
                    "avg_latency": m["refrag"]["latency"].get("avg"),
                    "rank": idx + 1
                }
                for idx, m in enumerate(refrag_latency_ranking[:10])
                if m["refrag"]["latency"].get("avg") is not None
            ],
            "fastest_simple_rag_models": [
                {
                    "model": m["model_name"], 
                    "avg_latency": m["simple_rag"]["latency"].get("avg"),
                    "rank": idx + 1
                }
                for idx, m in enumerate(simple_rag_latency_ranking[:10])
                if m["simple_rag"]["latency"].get("avg") is not None
            ],
            "best_improvement_models": sorted([
                {
                    "model": m["model_name"],
                    "latency_improvement": m["performance_comparison"].get("latency_improvement_percent"),
                    "token_reduction": m["performance_comparison"].get("token_reduction_percent")
                }
                for m in complete_models
                if m["performance_comparison"].get("latency_improvement_percent") is not None
            ], key=lambda x: x["latency_improvement"], reverse=True)[:10]
        },
        "summary_insights": {
            "total_models_with_complete_data": len(complete_models),
            "models_with_positive_latency_improvement": len([x for x in latency_improvements if x > 0]),
            "models_with_positive_token_reduction": len([x for x in token_reductions if x > 0]),
            "average_refrag_vs_simple_rag_improvement": calculate_stats(latency_improvements)
        }
    }

if __name__ == "__main__":
    analyze_latest_results()
