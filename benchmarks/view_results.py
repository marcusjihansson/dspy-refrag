#!/usr/bin/env python3
"""
Results management utility for viewing and analyzing saved benchmark results.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def list_saved_results():
    """List all saved benchmark results organized by model."""
    results_dir = Path(__file__).parent.parent / "results"
    
    if not results_dir.exists():
        print("❌ No results directory found")
        return
    
    print("📊 Saved Benchmark Results:")
    print()
    
    model_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name != "summary"]
    
    if not model_dirs:
        print("   (No results found)")
        return
    
    for model_dir in sorted(model_dirs):
        print(f"🤖 {model_dir.name.replace('_', '/')}:")
        
        # Find all result files
        result_files = list(model_dir.glob("*_benchmark_results.json"))
        
        if not result_files:
            print("   (No result files)")
            continue
        
        # Show latest and count
        latest_link = model_dir / "latest_results.json"
        if latest_link.exists():
            print(f"   📄 Latest: {latest_link.readlink()}")
        
        print(f"   📁 Total runs: {len(result_files)}")
        
        # Show recent files
        sorted_files = sorted(result_files, key=lambda x: x.name, reverse=True)
        for f in sorted_files[:3]:  # Show 3 most recent
            timestamp = f.stem.split('_benchmark_results')[0]
            try:
                dt = datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S")
                formatted_time = dt.strftime("%Y-%m-%d %H:%M")
                print(f"     • {formatted_time}")
            except:
                print(f"     • {timestamp}")
        
        if len(sorted_files) > 3:
            print(f"     ... and {len(sorted_files) - 3} more")
        
        print()


def show_model_summary(model_name: str):
    """Show summary for a specific model."""
    results_dir = Path(__file__).parent.parent / "results"
    safe_name = model_name.replace('/', '_').replace(':', '_')
    model_dir = results_dir / safe_name
    
    if not model_dir.exists():
        print(f"❌ No results found for model: {model_name}")
        print(f"   Looked for: {model_dir}")
        return
    
    latest_file = model_dir / "latest_results.json"
    if not latest_file.exists():
        print(f"❌ No latest results found for {model_name}")
        return
    
    with open(latest_file) as f:
        data = json.load(f)
    
    metadata = data.get("metadata", {})
    benchmark_data = data.get("benchmark_data", {})
    summary = benchmark_data.get("summary", {})
    
    print(f"📊 Latest Results for {metadata.get('model_name', model_name)}")
    print(f"🕒 Run at: {metadata.get('timestamp', 'Unknown')}")
    print(f"⚙️  Parameters: k={metadata.get('parameters', {}).get('k')}, budget={metadata.get('parameters', {}).get('budget')}")
    print()
    
    # Show performance comparison
    if "Simple RAG" in summary and "REFRAG" in summary:
        rag = summary["Simple RAG"]
        refrag = summary["REFRAG"]
        
        print("🚀 Performance Comparison:")
        print(f"   Latency:  Simple RAG: {rag['latency']['avg']:.2f}s | REFRAG: {refrag['latency']['avg']:.2f}s")
        print(f"   Tokens:   Simple RAG: {rag['tokens']['avg']:.0f} | REFRAG: {refrag['tokens']['avg']:.0f}")
        print(f"   Context:  Simple RAG: {rag['context_chars']['avg']:.0f} | REFRAG: {refrag['context_chars']['avg']:.0f}")
        
        # Calculate improvements with zero-division protection
        latency_improvement = (rag['latency']['avg'] - refrag['latency']['avg']) / rag['latency']['avg'] * 100 if rag['latency']['avg'] > 0 else 0
        token_improvement = (rag['tokens']['avg'] - refrag['tokens']['avg']) / rag['tokens']['avg'] * 100 if rag['tokens']['avg'] > 0 else 0
        
        print()
        print("📈 REFRAG Improvements:")
        if rag['latency']['avg'] > 0:
            print(f"   Latency: {latency_improvement:+.1f}%")
        else:
            print(f"   Latency: Unable to calculate (no latency data)")
            
        if rag['tokens']['avg'] > 0:
            print(f"   Tokens:  {token_improvement:+.1f}%")
        else:
            print(f"   Tokens:  Unable to calculate (no token data)")
    
    # Show similarity metrics
    similarity = benchmark_data.get("similarity", {})
    if similarity:
        print()
        print("🔍 Answer Similarity:")
        print(f"   Exact match: {similarity.get('exact', 0)*100:.1f}%")
        print(f"   Jaccard:     {similarity.get('jaccard', 0)*100:.1f}%")
        print(f"   Length ratio: {similarity.get('len_ratio', 0)*100:.1f}%")


def compare_models():
    """Compare performance across different models."""
    results_dir = Path(__file__).parent.parent / "results"
    summary_file = results_dir / "summary" / "all_models_comparison.json"
    
    if not summary_file.exists():
        print("❌ No cross-model comparison available")
        return
    
    print("🔄 Cross-Model Performance Comparison:")
    print("   (Feature coming soon - check individual model results for now)")


def main():
    parser = argparse.ArgumentParser(description="View and analyze saved benchmark results")
    parser.add_argument(
        "--list", action="store_true",
        help="List all saved results"
    )
    parser.add_argument(
        "--model", type=str,
        help="Show summary for specific model"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare performance across models"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_saved_results()
    elif args.model:
        show_model_summary(args.model)
    elif args.compare:
        compare_models()
    else:
        parser.print_help()
        print()
        list_saved_results()


if __name__ == "__main__":
    main()