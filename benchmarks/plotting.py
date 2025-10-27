import matplotlib.pyplot as plt
import numpy as np


def plot_results(summary, model_name: str | None = None):
    """Visualize benchmark metrics with enhanced formatting."""
    labels = list(summary.keys())
    # We'll plot avg latency and avg tokens
    latency = [summary[k]["latency"]["avg"] for k in labels]
    tokens = [summary[k]["tokens"]["avg"] for k in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))

    # Latency comparison
    bars1 = ax1.bar(labels, latency, alpha=0.7, color=["#3498db", "#e74c3c"])
    ax1.set_ylabel("Average Latency (seconds)")
    ax1.set_title("Latency Comparison")
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars1, latency):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + height * 0.01,
            f"{val:.2f}s",
            ha="center",
            va="bottom",
        )

    # Token usage comparison
    bars2 = ax2.bar(labels, tokens, alpha=0.7, color=["#3498db", "#e74c3c"])
    ax2.set_ylabel("Average Tokens")
    ax2.set_title("Token Usage Comparison")
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars2, tokens):
        height = bar.get_height()
        display_text = f"{int(val)}" if val > 0 else "No data"
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(height * 0.01, 10),
            display_text,
            ha="center",
            va="bottom",
        )

    # Calculate and show efficiency improvement
    if len(latency) == 2 and len(tokens) == 2:
        # Handle potential division by zero
        latency_improvement = (
            (latency[0] - latency[1]) / latency[0] * 100 if latency[0] > 0 else 0
        )
        token_improvement = (
            (tokens[0] - tokens[1]) / tokens[0] * 100 if tokens[0] > 0 else 0
        )

        # Create title with available metrics
        title_parts = [
            f"REFRAG vs Simple RAG Comparison",
            f"Model: {model_name or 'Current'}",
        ]

        if latency[0] > 0:
            title_parts.append(f"Latency: {latency_improvement:+.1f}%")

        if tokens[0] > 0:
            title_parts.append(f"Tokens: {token_improvement:+.1f}%")
        elif tokens[0] == 0 and tokens[1] == 0:
            title_parts.append("Tokens: No data available")

        fig.suptitle(
            "\n".join(title_parts[:3])
            + ("\n" + " | ".join(title_parts[3:]) if len(title_parts) > 3 else ""),
            fontsize=14,
            y=0.98,
        )

    plt.tight_layout()

    # Adjust layout to prevent warnings
    try:
        plt.subplots_adjust(top=0.85)  # Make room for the title
    except:
        pass  # Ignore if adjustment fails

    return fig