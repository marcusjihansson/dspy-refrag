import json
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def analyze_temporal():
    """
    Track performance metrics over time using timestamps from benchmark results.
    """
    results_dir = Path("../../results")
    output_file = "../analysis_results/temporal_tracking_results.json"
    
    temporal_data = defaultdict(list)
    models_data = []
    
    for model_dir in os.listdir(results_dir):
        model_path = results_dir / model_dir
        if model_path.is_dir():
            latest_results_path = model_path / "latest_results.json"
            if latest_results_path.exists():
                try:
                    with open(latest_results_path, "r") as f:
                        data = json.load(f)
                    
                    metadata = data.get("metadata", {})
                    model_name = metadata.get("model_name", "unknown")
                    safe_model_name = metadata.get("safe_model_name", model_dir)
                    timestamp_str = metadata.get("timestamp", "unknown")
                    
                    benchmark_data = data.get("benchmark_data", {})
                    summary = benchmark_data.get("summary", {})
                    
                    simple_rag = summary.get("Simple RAG", {})
                    refrag = summary.get("REFRAG", {})
                    
                    # Parse timestamp
                    try:
                        dt = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
                        date_key = dt.strftime("%Y-%m-%d")
                    except ValueError:
                        date_key = "unknown"
                    
                    model_entry = {
                        "model_name": model_name,
                        "safe_model_name": safe_model_name,
                        "timestamp": timestamp_str,
                        "date": date_key,
                        "simple_rag": extract_metrics(simple_rag),
                        "refrag": extract_metrics(refrag)
                    }
                    
                    models_data.append(model_entry)
                    temporal_data[date_key].append(model_entry)
                    
                except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                    print(f"Error processing {latest_results_path}: {e}")
                    continue
    
    # Generate temporal summary
    temporal_summary = generate_temporal_summary(temporal_data)
    
    analysis_results = {
        "analysis_timestamp": datetime.now().isoformat(),
        "total_models_analyzed": len(models_data),
        "individual_model_data": models_data,
        "temporal_summary": temporal_summary
    }
    
    # Save results
    with open(output_file, "w") as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"Temporal analysis complete! Results saved to {output_file}")
    print(f"Analyzed {len(models_data)} models")
    
    return analysis_results

def extract_metrics(rag_data):
    """Extract metrics from RAG data."""
    if not rag_data:
        return {}
    return {
        "latency_avg": rag_data.get("latency", {}).get("avg"),
        "tokens_avg": rag_data.get("tokens", {}).get("avg"),
        "context_chars_avg": rag_data.get("context_chars", {}).get("avg")
    }

def generate_temporal_summary(temporal_data):
    """Generate summary of metrics over time."""
    summary = {}
    
    for date, models in temporal_data.items():
        if date == "unknown":
            continue
        
        simple_latencies = [m["simple_rag"].get("latency_avg") for m in models if m["simple_rag"].get("latency_avg")]
        refrag_latencies = [m["refrag"].get("latency_avg") for m in models if m["refrag"].get("latency_avg")]
        simple_tokens = [m["simple_rag"].get("tokens_avg") for m in models if m["simple_rag"].get("tokens_avg")]
        refrag_tokens = [m["refrag"].get("tokens_avg") for m in models if m["refrag"].get("tokens_avg")]
        
        def stats(values):
            if not values:
                return {}
            return {
                "mean": round(sum(values) / len(values), 2),
                "min": min(values),
                "max": max(values),
                "count": len(values)
            }
        
        summary[date] = {
            "models_count": len(models),
            "simple_rag_stats": {
                "latency": stats(simple_latencies),
                "tokens": stats(simple_tokens)
            },
            "refrag_stats": {
                "latency": stats(refrag_latencies),
                "tokens": stats(refrag_tokens)
            }
        }
    
    return summary

if __name__ == "__main__":
    analyze_temporal()