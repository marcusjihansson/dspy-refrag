import json
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from benchmark_simple_rag import SimpleRAGModule
from evaluation import benchmark_model, evaluate_accuracy
from plotting import plot_results
from refrag_benchmark import REFRAGBenchmarkModule
from request_lm import RequestLM

from dspy_refrag import SimpleRetriever
from dspy_refrag.common import make_ollama_embedder
from dspy_refrag.data_ingest import build_corpus_from_data
from results import save_results_to_directory, update_summary_comparison  # type: ignore


class BenchmarkRunner:
    def __init__(
        self,
        model: Optional[str],
        api_key: Optional[str],
        base_url: Optional[str],
        k: int = 5,
        budget: int = 2,
        queries_path: Optional[str] = None,
        data_dir: Optional[Path] = None,
        embedder_config: Optional[dict] = None,
        no_plot: bool = False,
        save_results: bool = True,
        wait_time: float = 0.0,
    ):
        if not model:
            raise ValueError("Model name is required")
        if not api_key:
            raise ValueError("API key is required")
        if not base_url:
            raise ValueError("Base URL is required")

        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.k = k
        self.budget = budget
        self.queries_path = queries_path
        self.data_dir = data_dir or Path(__file__).resolve().parents[1] / "data"
        self.embedder_config = embedder_config or {}
        self.no_plot = no_plot
        self.save_results = save_results
        self.wait_time = wait_time

        # Internal state
        self.model_name = None
        self.embedder = None
        self.corpus = None
        self.lm = None
        self.simple_rag = None
        self.refrag = None
        self.queries = None

    def _setup(self):
        """Setup embedder, corpus, and LM."""
        # Use provided model directly
        self.model_name = self.model

        # Configure Ollama embedder
        ollama_base = self.embedder_config.get(
            "base_url", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        ollama_model = self.embedder_config.get(
            "model", os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
        )
        self.embedder = make_ollama_embedder(
            api_endpoint=ollama_base, model=ollama_model
        )

        # Build corpus
        print(f"Building corpus from: {self.data_dir}")
        self.corpus = build_corpus_from_data(self.embedder, self.data_dir)
        print(
            f"Built {len(self.corpus)} chunks from {len({p.metadata['source'] for p in self.corpus})} PDFs"
        )

        # Initialize LM
        self.lm = RequestLM(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
        )

        # Initialize models
        self.simple_rag = SimpleRAGModule(
            retriever=SimpleRetriever(embedder=self.embedder, corpus=self.corpus),
            k=self.k,
            lm=self.lm,
        )
        self.refrag = REFRAGBenchmarkModule(
            retriever=SimpleRetriever(embedder=self.embedder, corpus=self.corpus),
            k=self.k,
            budget=self.budget,
            lm=self.lm,
        )

        # Load queries
        if self.queries_path:
            with open(self.queries_path, "r") as f:
                self.queries = json.load(f)
                if not isinstance(self.queries, list):
                    raise ValueError("Queries JSON must be a list of strings")
        else:
            self.queries = [
                "What is DSPy?",
                "What problem does 'programming, not prompting' aim to solve?",
                "List key components or modules in DSPy.",
                "How does DSPy integrate with LLMs?",
            ]

    def run(self) -> dict:
        """Run the benchmarking process."""
        self._setup()

        # Run benchmarks
        agg_rag, res_rag = benchmark_model(
            self.simple_rag, self.queries, self.wait_time
        )
        agg_refrag, res_refrag = benchmark_model(
            self.refrag, self.queries, self.wait_time
        )

        # Compute summary
        summary = {
            "Simple RAG": agg_rag,
            "REFRAG": agg_refrag,
        }

        # Multi-metric similarity
        sim = evaluate_accuracy(res_refrag, res_rag)
        out = {
            "similarity": sim,
            "summary": summary,
            "queries": self.queries,
            "individual_results": {"simple_rag": res_rag, "refrag": res_refrag},
        }

        # Print structured stats
        print(json.dumps(out, indent=2))

        # Generate plot
        plot_fig = None
        if not self.no_plot:
            try:
                plot_fig = plot_results(summary, self.model_name)
            except Exception as e:
                print(f"⚠️  Warning: Could not generate plot: {e}")
                plot_fig = None

        # Save results
        if self.save_results and self.model_name:
            save_info = save_results_to_directory(
                out,
                self.model_name,
                type("Args", (), {"k": self.k, "budget": self.budget})(),
                plot_fig,
            )
            try:
                update_summary_comparison([save_info["metadata"]])
            except Exception as e:
                print(f"⚠️  Warning: Could not update summary: {e}")

        # Show plot if not saved
        if plot_fig and not self.save_results:
            plt.show()

        return out


if __name__ == "__main__":
    model = os.getenv("MODEL_NAME")
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_MONEY_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL")
    if not model or not api_key or not base_url:
        raise ValueError(
            "Environment variables MODEL_NAME, OPENROUTER_API_KEY (or OPENROUTER_MONEY_KEY), and OPENROUTER_BASE_URL must be set."
        )
    runner = BenchmarkRunner(  # type: ignore
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    runner.run()
