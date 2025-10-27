# Contributing to dspy-refrag

Thank you for your interest in contributing to dspy-refrag! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/dspy-refrag.git`
3. Create a virtual environment: `python -m venv .venv && source .venv/bin/activate`
4. Install in development mode: `pip install -e .`
5. Run tests to ensure everything works: `pytest tests/`

## Development Setup

### Prerequisites
- Python 3.11 or higher
- Git

### Optional Services for Full Testing
- Docker (for Weaviate tests)
- PostgreSQL with pgvector extension (for PostgreSQL tests)

### Installing Development Dependencies
```bash
pip install pytest black isort mypy pytest-cov
```

## Contributing Guidelines

### Code Style
- Follow PEP 8 style guidelines
- Use `black` for code formatting: `black src/ tests/`
- Use `isort` for import sorting: `isort src/ tests/`
- Add type hints where appropriate

### Testing
- Write tests for new features
- Ensure all tests pass: `pytest tests/`
- Maintain or improve code coverage
- Tests should gracefully skip when external services are unavailable

### Documentation
- Update README.md if adding new features
- Add docstrings to new classes and methods
- Include usage examples for new retrievers or sensors

## Implementing New Retrievers

The project provides scaffolding for new retriever implementations. To add a new retriever:

1. Create a new file: `src/dspy_refrag/your_retriever.py`
2. Inherit from the `Retriever` base class
3. Implement required methods: `embed_query()` and `retrieve()`
4. Add comprehensive docstrings with implementation examples
5. Export in `__init__.py`
6. Add tests in `tests/test_your_retriever.py`

### Example Structure
```python
from .retriever import Retriever

class YourRetriever(Retriever):
    def __init__(self, ...):
        # Implementation or scaffolding with NotImplementedError
        pass
    
    def embed_query(self, query: str) -> np.ndarray:
        # Implementation
        pass
    
    def retrieve(self, query: str, k: int = 3) -> List[Passage]:
        # Implementation
        pass
```

## Implementing New Sensors

To add advanced selection strategies:

1. Extend the `AdvancedSensor` class or create a new sensor class
2. Add new `SelectionStrategy` enum values if needed
3. Implement the selection logic
4. Add comprehensive tests and examples

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes with appropriate tests
3. Ensure all tests pass and code is formatted
4. Update documentation as needed
5. Submit a pull request with a clear description

### PR Checklist
- [ ] Tests pass locally
- [ ] Code is formatted with black and isort
- [ ] New features have tests
- [ ] Documentation is updated
- [ ] No hardcoded secrets or credentials

## Issues and Bug Reports

When reporting issues:
- Use the issue template
- Provide minimal reproduction steps
- Include Python version and dependency versions
- For Weaviate/external service issues, include service versions

## Community Guidelines

- Be respectful and inclusive
- Help others learn and grow
- Focus on constructive feedback
- Follow the DSPy community standards

## Questions?

- Check existing issues and discussions
- Create a new issue for bugs or feature requests
- Join the DSPy community for general questions

Thank you for contributing to dspy-refrag!