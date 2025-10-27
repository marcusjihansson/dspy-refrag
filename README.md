# dspy-refrag — Enterprise REFRAG Framework

**A comprehensive, production-ready framework for Retrieval-Enhanced Fragmented Reasoning and Generation (REFRAG)** that revolutionizes how large language models process and reason with retrieved information. Built on DSPy, this enterprise-grade solution provides advanced benchmarking, memory-enhanced capabilities, and sophisticated analysis tools for next-generation AI applications.

[![Production Ready](https://img.shields.io/badge/Status-Enterprise%20Ready-green.svg)](./PRODUCTION_READINESS.md)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![DSPy Compatible](https://img.shields.io/badge/DSPy-3.0.3%2B-orange.svg)](https://github.com/stanfordnlp/dspy)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What is REFRAG: 

- For additional information about REFRAG, please view this part of the documentation: [refrag.md](https://github.com/marcusjihansson/dspy-refrag/blob/main/refrag.md)

## Enterprise Features

### **Core REFRAG Implementation**

- **Advanced Vector Retrieval**: Production-ready support for Weaviate, PostgreSQL, FAISS, Pinecone with enterprise-grade embedders
- **Intelligent Fragment Selection**: ML-powered sensor strategies including MMR, uncertainty sampling, adaptive selection, and ensemble methods
- **Memory-Enhanced Reasoning**: Persistent conversation memory, quick facts caching, and optimization memory for continuous learning
- **Multi-Model Integration**: Native support for 25+ language models across OpenAI, Anthropic, Google, Groq, X.ai, and local models

### **Enterprise Benchmarking Suite**

- **Comprehensive Model Evaluation**: Automated benchmarking across 25+ language models with performance profiling
- **Advanced Analytics Framework**: Statistical testing, cost analysis, temporal performance tracking, and model categorization
- **Professional Visualization**: Automated plotting, comparison charts, and enterprise-ready reporting
- **Research-Grade Methodology**: Reproducible benchmarking with timestamped results and comprehensive analysis

### **Production Infrastructure**

- **Scalable Architecture**: Modular design supporting horizontal scaling and enterprise deployment
- **Professional Testing**: 50+ test cases with comprehensive integration testing and service availability checks
- **Security & Compliance**: Enterprise-grade secret management, input validation, and secure API handling
- **Comprehensive Documentation**: Professional documentation with module-specific guides and deployment instructions

## 📁 Enterprise Project Structure

```
dspy-refrag/
├── src/dspy_refrag/           # Core REFRAG implementation
│   ├── __init__.py            # Public API exports
│   ├── refrag.py              # Main REFRAG module with DSPy integration
│   ├── retriever.py           # Retriever implementations and scaffolding
│   ├── sensor.py              # Advanced chunk selection strategies
│   ├── fragment.py            # Fragment data structures with validation
│   ├── serializer_*.py        # Multiple serialization options
│   └── weaviate_retriever.py  # Production Weaviate integration
├── benchmarks/                # Enterprise benchmarking suite
│   ├── benchmark_runner*.py   # 20+ model-specific benchmark runners
│   ├── evaluation.py          # Comprehensive evaluation metrics
│   ├── plotting.py            # Advanced visualization tools
│   └── utils.py               # Benchmarking utilities
├── results/                   # Organized benchmark results
│   ├── {model_name}/          # Model-specific results with timestamps
│   └── summary/               # Cross-model analysis and comparisons
├── analysis/                  # Advanced analysis framework
│   ├── comparative_analysis/  # Statistical testing and model comparison
│   └── analysis_results/      # Analysis outputs and visualizations
├── memory/                    # Memory-enhanced REFRAG extensions
│   ├── memory_refrag/         # Conversational AI capabilities
│   └── memory_solution_in_refrag/ # Memory storage implementations
├── tests/                     # Comprehensive test suite (50+ tests)
├── examples/                  # Professional examples and quickstarts
└── docs/                      # Enterprise documentation
```

## 🛠 Installation & Setup

### Standard Installation

Install the complete framework with all dependencies:

```bash
pip install -e .
```

### Development Installation

For development with all optional dependencies:

```bash
pip install -e '.[dev]'
```

### Prerequisites

- **Python 3.11+** (Enterprise Python version)
- **DSPy 3.0.3+** for framework integration
- **Optional**: Docker for Weaviate and memory extensions

### Quick Setup for Benchmarking

```bash
# Install with all benchmarking dependencies
pip install -e .

# Run comprehensive benchmarks (requires API keys)
cd benchmarks
python benchmark_runner.py --model gpt-4o-mini --queries 10

# View results and analysis
cd ../analysis/comparative_analysis
python analyze.py --detailed --save
```

## 🚀 Quick Start

### Basic REFRAG Usage

```python
from dspy_refrag import REFRAGModule, SimpleRetriever

# Initialize with simple retriever
module = REFRAGModule(retriever=SimpleRetriever(), lm=None, k=3, budget=2)
ctx = module.forward("How do I train my dog?")
print(f"Query: {ctx.query}")
print(f"Selected chunks: {len(ctx.chunk_vectors)}")
print(f"Answer: {ctx.answer}")
```

### Enterprise Weaviate Integration

```python
from dspy_refrag import REFRAGModule
from dspy_refrag.weaviate_retriever import WeaviateRetriever, make_ollama_embedder

# Production Weaviate setup
embedder = make_ollama_embedder(api_endpoint="http://localhost:11434")
retriever = WeaviateRetriever(embedder=embedder, collection_name="EnterpriseDocs")
module = REFRAGModule(retriever=retriever, k=5, budget=3)

# Process enterprise queries
ctx = module.forward("Explain our data governance policies")
print(f"Enterprise Answer: {ctx.answer}")
```

### Memory-Enhanced Conversational AI

```python
import sys
sys.path.append('src')

from dspy_refrag import REFRAGModule
from memory.memory_refrag.memory_enhanced_refrag import MemoryEnhancedREFRAG
from memory.memory_solution_in_refrag.memory_store import InMemoryStore

# Initialize memory-enhanced REFRAG
memory = InMemoryStore()
enhanced_refrag = MemoryEnhancedREFRAG(
    memory=memory,
    retriever=retriever,
    session_id="enterprise_session_1"
)

# Conversational AI with memory
result = enhanced_refrag("What did we discuss about data privacy?")
print(f"Contextual Answer: {result.answer}")
```

### Professional Benchmarking

```python
from benchmarks.benchmark_runner import BenchmarkRunner

# Enterprise model evaluation
runner = BenchmarkRunner(
    model_name="gpt-4o-mini",
    queries=["enterprise query 1", "enterprise query 2"],
    save_results=True
)

results = runner.run_benchmark()
print(f"Performance Metrics: {results.summary}")
```

## 🏢 Enterprise Use Cases

### **Customer Service Automation**

Deploy memory-enhanced REFRAG for intelligent customer support with conversation history and quick facts caching for common queries.

### **Knowledge Management Systems**

Implement enterprise-wide knowledge retrieval with advanced sensor strategies for accurate information discovery across large document repositories.

### **Research & Development**

Utilize comprehensive benchmarking infrastructure to evaluate and optimize RAG implementations across different domains and model configurations.

### **Conversational AI Platforms**

Build sophisticated chatbots with persistent memory, reasoning capabilities, and optimization memory for continuous improvement.

## 📊 Benchmarking & Analysis

### **Multi-Model Performance Evaluation**

The framework includes comprehensive benchmarking across 25+ language models:

```bash
# Run enterprise benchmarks
cd benchmarks
python benchmark_runner.py --model openai_gpt-4o-mini --queries data/enterprise_queries.json
python benchmark_runner_claude.py --model anthropic_claude-sonnet-4
python benchmark_runner_gemini.py --model google_gemini-2.5-flash

# Generate comparative analysis
cd ../analysis/comparative_analysis
python analyze.py --detailed --save --models all
```

### **Advanced Analysis Framework**

Access sophisticated analysis tools for performance optimization:

- **Statistical Testing**: Compare REFRAG vs traditional RAG with statistical significance
- **Cost Analysis**: Evaluate token efficiency and operational costs across models
- **Quality Metrics**: Assess response quality and relevance using multiple evaluation criteria
- **Temporal Analysis**: Track performance improvements over time

### **Professional Reporting**

Generate enterprise-ready reports and visualizations:

```python
from analysis.comparative_analysis.viz import generate_performance_report

# Generate comprehensive performance report
report = generate_performance_report(
    models=["gpt-4o-mini", "claude-sonnet-4", "gemini-2.5-flash"],
    output_format="enterprise"
)
```

## 📚 Comprehensive API Reference

### **Core REFRAG Classes**

#### `REFRAGModule`

Main enterprise module for REFRAG workflows.

```python
REFRAGModule(
    retriever: Retriever,
    lm: Optional[dspy.LM] = None,
    sensor: Optional[Sensor] = None,
    k: int = 3,
    budget: int = 2,
    lm_model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None
)
```

**Methods:**

- `forward(query: str) -> REFRAGContext`: Process query and return results
- `add_memory(memory_store: MemoryStore)`: Add memory capabilities

#### `REFRAGContext`

Enterprise result object with comprehensive metadata.

**Attributes:**

- `query: str`: Original query
- `chunk_vectors: List[np.ndarray]`: Selected vector chunks
- `metadata: Dict`: Retrieval and processing metadata
- `answer: Optional[str]`: Generated answer from LM
- `reasoning_path: List[Dict]`: Step-by-step reasoning trace

### **Advanced Retrievers**

#### `WeaviateRetriever`

Production-grade Weaviate integration.

```python
WeaviateRetriever(
    embedder: EmbedderProtocol,
    collection_name: str = "Documents",
    weaviate_url: str = "http://localhost:8080",
    timeout_config: Optional[dict] = None
)
```

#### `PostgreSQLRetriever`

Enterprise PostgreSQL with pgvector support.

#### `MemoryEnhancedREFRAG`

Advanced conversational AI with persistent memory.

```python
MemoryEnhancedREFRAG(
    memory: MemoryStore,
    retriever: Retriever,
    session_id: str,
    quick_facts_enabled: bool = True,
    optimization_memory: bool = True
)
```

### **Professional Sensor Strategies**

#### `AdvancedSensor`

ML-powered chunk selection with multiple algorithms.

**Selection Modes:**

- `"mmr"`: Maximal Marginal Relevance for diversity
- `"uncertainty"`: Uncertainty sampling for active learning
- `"adaptive"`: Dynamic strategy selection
- `"ensemble"`: Combination of multiple strategies

### **Benchmarking Infrastructure**

#### `BenchmarkRunner`

Enterprise benchmarking with comprehensive metrics.

```python
BenchmarkRunner(
    model_name: str,
    queries: List[str],
    save_results: bool = True,
    output_dir: str = "results/",
    metrics: List[str] = ["latency", "tokens", "quality"]
)
```

#### `AnalysisFramework`

Statistical analysis and model comparison.

```python
from analysis.comparative_analysis import AnalysisFramework

analyzer = AnalysisFramework()
results = analyzer.compare_models(
    models=["gpt-4o-mini", "claude-sonnet-4"],
    statistical_tests=True,
    cost_analysis=True
)
```

## 🔧 Enterprise Deployment

### **Docker Deployment**

```bash
# Core REFRAG with Weaviate
docker-compose up -d

# Memory-enhanced deployment
docker-compose -f memory/docker-compose.yml up -d
```

### **Production Environment Setup**

```bash
# Install production dependencies
pip install -e .

# Configure environment variables
export OPENAI_API_KEY="your-enterprise-key"
export WEAVIATE_URL="https://your-enterprise-weaviate.com"
export MEMORY_STORE_URL="your-memory-backend"

# Run production health checks
python -m pytest tests/ -k "production"
```

### **Kubernetes Deployment**

Enterprise Kubernetes manifests available in `deployment/k8s/` for scalable production deployment.

## 🛠 Troubleshooting & Support

### **Common Issues**

**Weaviate Connection**: Ensure Weaviate is running and accessible

```bash
docker run -p 8080:8080 -p 50051:50051 weaviate/weaviate:latest
```

**Memory Extension Setup**: Verify Docker Compose for memory services

```bash
cd memory && docker-compose up -d
```

**API Key Configuration**: Check environment variables for model access

```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

**Benchmark Failures**: Validate model accessibility and rate limits

```bash
python benchmarks/utils.py --test-connection --model gpt-4o-mini
```

### **Enterprise Support**

- **Documentation**: Comprehensive guides in each module's README
- **Examples**: Production-ready examples in `examples/`
- **Testing**: Run `pytest tests/` for validation
- **Community**: GitHub Issues for community support
- **Enterprise**: Contact for enterprise support and consulting

## Contributing to Enterprise REFRAG

We welcome contributions to advance the state of REFRAG technology. Priority areas include:

### **Core Development**

- **Advanced Retrievers**: Implement new vector database integrations
- **Sensor Strategies**: Develop ML-powered chunk selection algorithms
- **Memory Systems**: Enhance conversational AI and optimization capabilities
- **Performance Optimization**: Improve latency and throughput

### **Research & Benchmarking**

- **Model Evaluation**: Add support for new language models
- **Evaluation Metrics**: Develop domain-specific quality assessments
- **Analysis Tools**: Create advanced statistical analysis capabilities
- **Datasets**: Contribute benchmarking datasets and test cases

### **Enterprise Features**

- **Security**: Enhance enterprise security and compliance features
- **Monitoring**: Develop production monitoring and alerting
- **Scalability**: Optimize for high-throughput enterprise deployment
- **Integration**: Build connectors for enterprise systems

### **Development Guidelines**

1. **Follow Enterprise Standards**: Maintain high code quality and documentation standards
2. **Comprehensive Testing**: Add tests for all new features with >90% coverage
3. **Security First**: Ensure secure handling of API keys and sensitive data
4. **Performance Focused**: Benchmark and optimize all performance-critical changes
5. **Documentation**: Update all relevant documentation and examples

### **Contribution Process**

1. Fork the repository and create a feature branch
2. Implement changes following our coding standards
3. Add comprehensive tests and documentation
4. Run the full test suite and benchmarking validation
5. Submit a pull request with detailed description and performance analysis

## 📄 License & Enterprise Usage

This project is licensed under the MIT License, enabling both open-source and enterprise commercial usage. See [LICENSE](LICENSE) for full details.

### **Enterprise Licensing**

- ✅ **Commercial Use**: Full commercial usage rights
- ✅ **Modification**: Modify and extend for enterprise needs
- ✅ **Distribution**: Distribute as part of enterprise products
- ✅ **Private Use**: Use in private enterprise environments

## Acknowledgments

Built on the [DSPy](https://github.com/stanfordnlp/dspy) framework. Special thanks to the DSPy community and Stanford NLP Group for creating the foundation that makes REFRAG possible.

**Enterprise Framework**: Developed by Marcus Johansson  
**Production Readiness**: [Grade A Enterprise Ready](./PRODUCTION_READINESS.md)  
**Community**: Join us in advancing the future of retrieval-augmented generation

---

## Contact:

I am open to work. Let's work together. I am ready to start working as soon as possible!!

**Send me a message on either:** 

Twitter/X: [Visit Twitter/X](https://x.com/marcusjihansson) 

LinkedIn: [Visit LinkedIn](https://www.linkedin.com/in/marcus-frid-johansson/) 

## Additional projects:

Please take a look at my Github for my additional skills.
Coding languages: Python, Go, C++, SQL

[Visit GitHub](https://github.com/marcusjihansson) 
