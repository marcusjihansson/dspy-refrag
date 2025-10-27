import json
import os
from pathlib import Path
from datetime import datetime
import statistics

def analyze_statistical():
    """
    Perform statistical tests on metrics between Simple RAG and REFRAG.
    """
    results_dir = Path("../../results")
    output_file = "../analysis_results/statistical_testing_results.json"
    
    simple_latencies = []
    refrag_latencies = []
    simple_tokens = []
    refrag_tokens = []
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
                    
                    benchmark_data = data.get("benchmark_data", {})
                    summary = benchmark_data.get("summary", {})
                    
                    simple_rag = summary.get("Simple RAG", {})
                    refrag = summary.get("REFRAG", {})
                    
                    simple_latency = simple_rag.get("latency", {}).get("avg")
                    refrag_latency = refrag.get("latency", {}).get("avg")
                    simple_token = simple_rag.get("tokens", {}).get("avg")
                    refrag_token = refrag.get("tokens", {}).get("avg")
                    
                    if simple_latency and refrag_latency:
                        simple_latencies.append(simple_latency)
                        refrag_latencies.append(refrag_latency)
                    
                    if simple_token and refrag_token:
                        simple_tokens.append(simple_token)
                        refrag_tokens.append(refrag_token)
                    
                    model_entry = {
                        "model_name": model_name,
                        "safe_model_name": safe_model_name,
                        "simple_latency": simple_latency,
                        "refrag_latency": refrag_latency,
                        "simple_tokens": simple_token,
                        "refrag_tokens": refrag_token
                    }
                    
                    models_data.append(model_entry)
                    
                except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                    print(f"Error processing {latest_results_path}: {e}")
                    continue
    
    # Perform statistical tests
    statistical_summary = perform_tests(simple_latencies, refrag_latencies, simple_tokens, refrag_tokens)
    
    analysis_results = {
        "analysis_timestamp": datetime.now().isoformat(),
        "total_models_analyzed": len(models_data),
        "individual_model_data": models_data,
        "statistical_summary": statistical_summary
    }
    
    # Save results
    with open(output_file, "w") as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"Statistical analysis complete! Results saved to {output_file}")
    print(f"Analyzed {len(models_data)} models")
    
    return analysis_results

def perform_tests(simple_lat, refrag_lat, simple_tok, refrag_tok):
    """Perform statistical tests."""
    def t_test(sample1, sample2):
        if len(sample1) < 2 or len(sample2) < 2:
            return {"error": "Not enough data for t-test"}
        
        mean1 = statistics.mean(sample1)
        mean2 = statistics.mean(sample2)
        var1 = statistics.variance(sample1)
        var2 = statistics.variance(sample2)
        n1 = len(sample1)
        n2 = len(sample2)
        
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        se = (pooled_var * (1/n1 + 1/n2)) ** 0.5
        t_stat = (mean1 - mean2) / se if se > 0 else 0
        
        return {
            "t_statistic": round(t_stat, 4),
            "mean1": round(mean1, 4),
            "mean2": round(mean2, 4),
            "p_value_approx": round(0.05 if abs(t_stat) > 1.96 else 0.1, 4)  # Rough approximation
        }
    
    return {
        "latency_test": t_test(simple_lat, refrag_lat),
        "tokens_test": t_test(simple_tok, refrag_tok),
        "sample_sizes": {
            "latency": {"simple": len(simple_lat), "refrag": len(refrag_lat)},
            "tokens": {"simple": len(simple_tok), "refrag": len(refrag_tok)}
        }
    }

if __name__ == "__main__":
    analyze_statistical()