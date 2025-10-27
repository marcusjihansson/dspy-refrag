import json
import os
from pathlib import Path
from datetime import datetime

# Hypothetical pricing (per 1000 tokens for input/output, per second for latency)
PRICING = {
    "input_tokens_per_1000": 0.002,
    "output_tokens_per_1000": 0.004,
    "latency_per_second": 0.001
}

def analyze_costs():
    """
    Estimate costs based on token usage and latency from benchmark results.
    Compares costs between Simple RAG and REFRAG approaches.
    """
    results_dir = Path("../../results")
    output_file = "../analysis_results/cost_modeling_results.json"
    
    models_data = []
    cost_summary = {
        "simple_rag_costs": {},
        "refrag_costs": {},
        "cost_savings": {}
    }
    
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
                    
                    benchmark_data = data.get("benchmark_data", {})
                    summary = benchmark_data.get("summary", {})
                    
                    simple_rag = summary.get("Simple RAG", {})
                    refrag = summary.get("REFRAG", {})
                    
                    model_entry = {
                        "model_name": model_name,
                        "safe_model_name": safe_model_name,
                        "simple_rag_cost": calculate_cost(simple_rag),
                        "refrag_cost": calculate_cost(refrag),
                        "cost_savings": {}
                    }
                    
                    if model_entry["simple_rag_cost"] and model_entry["refrag_cost"]:
                        savings = model_entry["simple_rag_cost"] - model_entry["refrag_cost"]
                        model_entry["cost_savings"] = {
                            "absolute": round(savings, 4),
                            "percentage": round((savings / model_entry["simple_rag_cost"]) * 100, 2) if model_entry["simple_rag_cost"] > 0 else 0
                        }
                    
                    models_data.append(model_entry)
                    
                except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                    print(f"Error processing {latest_results_path}: {e}")
                    continue
    
    # Generate cost summary
    cost_summary = generate_cost_summary(models_data)
    
    analysis_results = {
        "analysis_timestamp": datetime.now().isoformat(),
        "pricing_model": PRICING,
        "total_models_analyzed": len(models_data),
        "individual_model_data": models_data,
        "cost_summary": cost_summary
    }
    
    # Save results
    with open(output_file, "w") as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"Cost analysis complete! Results saved to {output_file}")
    print(f"Analyzed {len(models_data)} models")
    
    return analysis_results

def calculate_cost(rag_data):
    """Calculate estimated cost for a RAG approach."""
    if not rag_data:
        return None
    
    tokens = rag_data.get("tokens", {}).get("avg", 0)
    latency = rag_data.get("latency", {}).get("avg", 0)
    
    # Assume half tokens are input, half output
    input_tokens = tokens / 2
    output_tokens = tokens / 2
    
    token_cost = (input_tokens / 1000 * PRICING["input_tokens_per_1000"]) + (output_tokens / 1000 * PRICING["output_tokens_per_1000"])
    latency_cost = latency * PRICING["latency_per_second"]
    
    return round(token_cost + latency_cost, 4)

def generate_cost_summary(models_data):
    """Generate summary of cost metrics."""
    simple_costs = []
    refrag_costs = []
    savings = []
    
    for model in models_data:
        simple = model.get("simple_rag_cost")
        refrag = model.get("refrag_cost")
        save = model.get("cost_savings", {}).get("absolute")
        
        if simple:
            simple_costs.append(simple)
        if refrag:
            refrag_costs.append(refrag)
        if save is not None:
            savings.append(save)
    
    def calculate_stats(values):
        if not values:
            return {}
        return {
            "mean": round(sum(values) / len(values), 4),
            "min": min(values),
            "max": max(values),
            "count": len(values)
        }
    
    return {
        "simple_rag_cost_stats": calculate_stats(simple_costs),
        "refrag_cost_stats": calculate_stats(refrag_costs),
        "savings_stats": calculate_stats(savings),
        "insights": {
            "average_cost_savings": calculate_stats(savings).get("mean", 0),
            "models_with_savings": len([s for s in savings if s > 0])
        }
    }

if __name__ == "__main__":
    analyze_costs()