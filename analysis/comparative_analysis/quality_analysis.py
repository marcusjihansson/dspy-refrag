import json
import os
from pathlib import Path
from datetime import datetime

def analyze_quality():
    """
    Analyze answer quality using similarity metrics from benchmark results.
    Compares quality between Simple RAG and REFRAG approaches.
    """
    results_dir = Path("../../results")
    output_file = "../analysis_results/quality_analysis_results.json"
    
    models_data = []
    quality_summary = {
        "exact_similarity": {},
        "jaccard_similarity": {},
        "len_ratio_similarity": {},
        "overall_quality_comparison": {}
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
                    similarity = benchmark_data.get("similarity", {})
                    
                    model_entry = {
                        "model_name": model_name,
                        "safe_model_name": safe_model_name,
                        "similarity_metrics": similarity
                    }
                    
                    models_data.append(model_entry)
                    
                except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                    print(f"Error processing {latest_results_path}: {e}")
                    continue
    
    # Generate quality analysis
    quality_summary = generate_quality_summary(models_data)
    
    analysis_results = {
        "analysis_timestamp": datetime.now().isoformat(),
        "total_models_analyzed": len(models_data),
        "individual_model_data": models_data,
        "quality_summary": quality_summary
    }
    
    # Save results
    with open(output_file, "w") as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"Quality analysis complete! Results saved to {output_file}")
    print(f"Analyzed {len(models_data)} models")
    
    return analysis_results

def generate_quality_summary(models_data):
    """Generate summary of quality metrics."""
    exact_scores = []
    jaccard_scores = []
    len_ratio_scores = []
    
    for model in models_data:
        sim = model.get("similarity_metrics", {})
        exact = sim.get("exact", 0)
        jaccard = sim.get("jaccard", 0)
        len_ratio = sim.get("len_ratio", 0)
        
        exact_scores.append(exact)
        jaccard_scores.append(jaccard)
        len_ratio_scores.append(len_ratio)
    
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
        "exact_similarity_stats": calculate_stats(exact_scores),
        "jaccard_similarity_stats": calculate_stats(jaccard_scores),
        "len_ratio_similarity_stats": calculate_stats(len_ratio_scores),
        "insights": {
            "average_exact_similarity": calculate_stats(exact_scores).get("mean", 0),
            "average_jaccard_similarity": calculate_stats(jaccard_scores).get("mean", 0),
            "average_len_ratio_similarity": calculate_stats(len_ratio_scores).get("mean", 0)
        }
    }

if __name__ == "__main__":
    analyze_quality()