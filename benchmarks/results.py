import json
import os
from datetime import datetime
from pathlib import Path
from utils import sanitize_model_name


def save_results_to_directory(results: dict, model_name: str, args, plot_fig=None):
    """Save benchmark results to organized directory structure."""
    # Create results directory structure
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Sanitize model name for filesystem
    safe_model_name = sanitize_model_name(model_name)
    model_dir = results_dir / safe_model_name
    model_dir.mkdir(exist_ok=True)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Enhance results with metadata
    enhanced_results = {
        "metadata": {
            "timestamp": timestamp,
            "model_name": model_name,
            "safe_model_name": safe_model_name,
            "parameters": {
                "k": args.k,
                "budget": args.budget,
                "queries_count": len(results.get("queries", [])),
            },
            "environment": {
                "ollama_base": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                "ollama_model": os.getenv(
                    "OLLAMA_EMBED_MODEL", "nomic-embed-text:latest"
                ),
                "openrouter_base": os.getenv("OPENROUTER_BASE_URL"),
                "github_base": os.getenv("GITHUB_BASE_URL"),
                "github_model": os.getenv("GITHUB_MODEL"),
            },
        },
        "benchmark_data": results,
    }

    # Save JSON results
    json_filename = f"{timestamp}_benchmark_results.json"
    json_path = model_dir / json_filename

    with open(json_path, "w") as f:
        json.dump(enhanced_results, f, indent=2)

    # Create/update latest symlink
    latest_path = model_dir / "latest_results.json"
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()
    latest_path.symlink_to(json_filename)

    # Save plot if provided
    plot_path = None
    if plot_fig is not None:
        plot_filename = f"{timestamp}_comparison_plot.png"
        plot_path = model_dir / plot_filename
        plot_fig.savefig(plot_path, dpi=72, bbox_inches=None)

        # Create/update latest plot symlink
        latest_plot_path = model_dir / "latest_plot.png"
        if latest_plot_path.exists() or latest_plot_path.is_symlink():
            latest_plot_path.unlink()
        latest_plot_path.symlink_to(plot_filename)

    print(f"\n📁 Results saved to: {model_dir}")
    print(f"   📄 Data: {json_path.name}")
    if plot_path:
        print(f"   📊 Plot: {plot_path.name}")
    print(f"   🔗 Latest: latest_results.json -> {json_filename}")

    return {
        "model_dir": model_dir,
        "json_path": json_path,
        "plot_path": plot_path,
        "metadata": enhanced_results["metadata"],
    }


def update_summary_comparison(all_results: list):
    """Update cross-model comparison summary."""
    results_dir = Path(__file__).parent.parent / "results"
    summary_dir = results_dir / "summary"
    summary_dir.mkdir(exist_ok=True)

    # Save cross-model comparison
    summary_path = summary_dir / "all_models_comparison.json"

    summary_data = {"updated": datetime.now().isoformat(), "models": all_results}

    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"📊 Updated summary: {summary_path}")