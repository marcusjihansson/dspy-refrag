#!/usr/bin/env python3
"""
Command-line interface for running comparative analysis.

Usage:
    python analyze.py [--summary] [--detailed] [--json] [--save] [--output=filename]
    
Options:
    --summary    Show summary report only (default)
    --detailed   Show detailed analysis  
    --json       Output raw JSON results
    --save       Save output to file
    --output=filename  Save to specific filename (implies --save)
    --help       Show this help message
"""

import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

def save_output(content, output_type="summary", custom_filename=None):
    """Save output content to a file with appropriate naming."""
    if custom_filename:
        filename = custom_filename
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if output_type == "summary":
            filename = f"summary_report_{timestamp}.txt"
        elif output_type == "detailed":
            filename = f"detailed_analysis_{timestamp}.txt"
        elif output_type == "json":
            filename = f"analysis_results_{timestamp}.json"
        else:
            filename = f"analysis_output_{timestamp}.txt"
    
    try:
        with open("../analysis_results/" + filename, 'w') as f:
            f.write(content)
        print(f"\n💾 Output saved to: {filename}")
        return filename
    except Exception as e:
        print(f"\n❌ Error saving to file: {e}")
        return None

def main():
    args = sys.argv[1:]

    # Parse arguments
    save_output_flag = "--save" in args
    output_filename = None

    for arg in args:
        if arg.startswith("--output="):
            output_filename = arg.split("=", 1)[1]
            save_output_flag = True

    if "--help" in args or "-h" in args:
        print(__doc__)
        return

    # Run the analysis
    print("Running comparative analysis...")
    try:
        result = subprocess.run([sys.executable, "comparative_analysis.py"],
                               capture_output=True, text=True, check=True)
        print("✅ Analysis completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Analysis failed: {e}")
        print(f"Error output: {e.stderr}")
        return 1

    # Integrate new analyses
    print("Integrating additional analyses...")
    integrate_additional_analyses()
    
    # Determine output format
    if "--json" in args:
        # Output raw JSON
        try:
            with open("../analysis_results/comparative_analysis_results.json", "r") as f:
                data = json.load(f)
            json_output = json.dumps(data, indent=2)
            print(json_output)
            
            if save_output_flag:
                save_output(json_output, "json", output_filename)
                
        except FileNotFoundError:
            print("❌ Results file not found")
            return 1
            
    elif "--detailed" in args:
        # Show detailed analysis from JSON
        try:
            with open("../analysis_results/comparative_analysis_results.json", "r") as f:
                data = json.load(f)
            
            detailed_output = []
            detailed_output.append("=" * 70)
            detailed_output.append("DETAILED COMPARATIVE ANALYSIS")
            detailed_output.append("=" * 70)
            
            # Show some key detailed metrics
            metrics = data.get("comparative_metrics", {})
            individual_data = data.get("individual_model_data", [])
            
            detailed_output.append(f"\n📊 ANALYSIS OVERVIEW")
            detailed_output.append(f"Models analyzed: {len(individual_data)}")
            detailed_output.append(f"Analysis timestamp: {data.get('analysis_timestamp', 'Unknown')}")
            
            # Show individual model performance
            detailed_output.append(f"\n📈 INDIVIDUAL MODEL PERFORMANCE")
            detailed_output.append("-" * 50)
            
            for model_data in individual_data[:10]:  # Show first 10
                model_name = model_data.get("model_name", "Unknown").split("/")[-1]
                perf_comp = model_data.get("performance_comparison", {})
                latency_imp = perf_comp.get("latency_improvement_percent")
                token_red = perf_comp.get("token_reduction_percent")
                
                if latency_imp is not None:
                    token_str = f", {token_red:.1f}% tokens" if token_red else ""
                    detailed_output.append(f"  {model_name}: {latency_imp:+.1f}% latency{token_str}")
            
            # Show aggregate stats
            agg = metrics.get("aggregate_improvements", {})
            detailed_output.append(f"\n📊 AGGREGATE STATISTICS")
            detailed_output.append("-" * 30)
            
            for metric, stats in agg.items():
                if stats:
                    metric_name = metric.replace("_", " ").title().replace("Stats", "")
                    detailed_output.append(f"{metric_name}:")
                    detailed_output.append(f"  Mean: {stats.get('mean', 0):.1f}%")
                    detailed_output.append(f"  Range: {stats.get('min', 0):.1f}% to {stats.get('max', 0):.1f}%")
                    detailed_output.append(f"  Sample size: {stats.get('count', 0)}")
                    detailed_output.append("")
            
            detailed_text = "\n".join(detailed_output)
            print(detailed_text)
            
            if save_output_flag:
                save_output(detailed_text, "detailed", output_filename)
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"❌ Error reading detailed results: {e}")
            return 1
    
    else:
        # Default: show summary (same as --summary)
        try:
            if save_output_flag:
                # Call generate_summary with save options
                save_args = ["--save"]
                if output_filename:
                    save_args = [f"--output={output_filename}"]
                
                result = subprocess.run([sys.executable, "generate_summary.py"] + save_args, 
                                      check=True)
            else:
                result = subprocess.run([sys.executable, "generate_summary.py"], 
                                      check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Summary generation failed: {e}")
            return 1
    
    return 0

def integrate_additional_analyses():
    """Integrate results from new analysis scripts."""
    results_dir = Path("../analysis_results")
    main_results = Path("../analysis_results/comparative_analysis_results.json")

    if not main_results.exists():
        print("❌ Main analysis results not found")
        return

    try:
        with open(main_results, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("❌ Error reading main results")
        return

    # Read new analysis files
    new_analyses = {
        "quality_analysis": "quality_analysis_results.json",
        "cost_modeling": "cost_modeling_results.json",
        "temporal_tracking": "temporal_tracking_results.json",
        "statistical_testing": "statistical_testing_results.json",
        "model_categorization": "model_categorization_results.json"
    }

    integrated_data = {}
    for key, filename in new_analyses.items():
        file_path = results_dir / filename
        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    integrated_data[key] = json.load(f)
                print(f"✅ Integrated {key}")
            except json.JSONDecodeError:
                print(f"❌ Error reading {filename}")
        else:
            print(f"⚠️ {filename} not found")

    # Update main results
    data["integrated_analyses"] = integrated_data

    # Save updated results
    with open(main_results, "w") as f:
        json.dump(data, f, indent=2)

    print("✅ Integration complete")

if __name__ == "__main__":
    sys.exit(main())