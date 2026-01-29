# Contributing to CKD Prediction System

Thank you for your interest in contributing to the CKD Prediction System! This document provides guidelines and information for contributors.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Familiarity with machine learning concepts
- Basic understanding of healthcare data

### Development Setup

1. Fork the repository
2. Clone your fork locally:
```bash
git clone https://github.com/your-username/ckd.git
cd ckd
```

3. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

5. Install development dependencies:
```bash
pip install -e ".[dev]"
```

## Code Style and Standards

### Python Code Style

We follow PEP 8 guidelines with the following tools:

- **Black**: For code formatting
- **Flake8**: For linting
- **isort**: For import sorting

Run formatting before committing:
```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
```

### Documentation

- All public functions and classes must have docstrings
- Use Google-style docstrings
- Include type hints where applicable

### Testing

- Write unit tests for new functionality
- Ensure all tests pass before submitting PR
- Aim for high code coverage

Run tests:
```bash
pytest tests/
```

## Project Structure

```
ckd/
├── src/
│   ├── models/          # ML models and algorithms
│   ├── data/           # Data processing utilities
│   ├── features/       # Feature engineering
│   └── utils/          # Helper functions
├── tests/              # Unit tests
├── notebooks/          # Research notebooks
├── configs/            # Configuration files
├── docs/               # Documentation
└── scripts/            # Utility scripts
```

## Contribution Types

### Bug Reports

- Use the issue template for bug reports
- Include detailed steps to reproduce
- Provide environment details

### Feature Requests

- Open an issue with "Feature Request" label
- Describe the use case and expected behavior
- Discuss implementation approach

### Code Contributions

1. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes following the coding standards

3. Add tests for new functionality

4. Update documentation if needed

5. Commit your changes:
```bash
git commit -m "feat: add your feature description"
```

6. Push to your fork:
```bash
git push origin feature/your-feature-name
```

7. Create a Pull Request

## Pull Request Process

### PR Requirements

- Clear description of changes
- Link to relevant issues
- Tests pass
- Code follows style guidelines
- Documentation updated

### PR Template

Use the provided PR template and include:

- **Description**: What changes were made and why
- **Testing**: How changes were tested
- **Breaking Changes**: Any breaking changes
- **Documentation**: Documentation updates

## Review Process

1. Automated checks (CI/CD)
2. Code review by maintainers
3. Testing on different environments
4. Approval and merge

## Release Process

Releases follow semantic versioning:

- **Major**: Breaking changes
- **Minor**: New features
- **Patch**: Bug fixes

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Maintain professional communication

### Communication

- Use GitHub issues for bug reports and features
- Use discussions for general questions
- Join our community forum for extended discussions

## Recognition

Contributors are recognized in:

- README.md contributors section
- Release notes
- Annual contributor report

## Questions?

- Check existing issues and discussions
- Read the documentation
- Contact maintainers at research@ckd-project.org

Thank you for contributing to the CKD Prediction System!
