# Contributing to OpenWorld-Multimodal

We welcome contributions to OpenWorld-Multimodal! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Code Review Process](#code-review-process)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. We are committed to providing a welcoming and inclusive environment for all contributors.

### Our Standards

- **Be respectful**: Treat all community members with respect and kindness
- **Be inclusive**: Welcome newcomers and help them get started
- **Be constructive**: Provide helpful feedback and suggestions
- **Be professional**: Maintain a professional tone in all interactions

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Basic knowledge of PyTorch and transformers
- Familiarity with multimodal machine learning concepts

### Areas for Contribution

We welcome contributions in the following areas:

- **Bug fixes**: Help us identify and fix issues
- **Feature development**: Implement new features and capabilities
- **Documentation**: Improve documentation, tutorials, and examples
- **Testing**: Add tests and improve test coverage
- **Performance optimization**: Optimize code for speed and memory usage
- **Research integration**: Implement new research findings and techniques

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/OpenWorld-Multimodal.git
cd OpenWorld-Multimodal
```

### 2. Set Up Development Environment

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 3. Verify Setup

```bash
# Run tests to ensure everything works
pytest tests/ -v

# Run the demo
python demo.py

# Check code style
black --check openworld/
ruff check openworld/
```

## Making Changes

### Branch Naming Convention

Use descriptive branch names that follow this pattern:
- `feature/description` - for new features
- `fix/description` - for bug fixes
- `docs/description` - for documentation changes
- `refactor/description` - for code refactoring
- `test/description` - for test additions/improvements

### Commit Message Guidelines

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(models): add support for variable sequence lengths
fix(training): resolve memory leak in distributed training
docs(api): update API reference for new features
test(evaluation): add comprehensive metric tests
```

### Code Style Guidelines

#### Python Code Style

- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [Ruff](https://github.com/astral-sh/ruff) for linting
- Maximum line length: 100 characters

#### Type Hints

- Add type hints to all function signatures
- Use `from typing import` for type annotations
- Example:
```python
from typing import Dict, List, Optional, Tuple

def process_data(
    video: torch.Tensor,
    audio: torch.Tensor,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process multimodal data."""
    # Implementation
    return processed_video, processed_audio
```

#### Documentation

- Use Google-style docstrings
- Include parameter descriptions, return values, and examples
- Example:
```python
def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    epochs: int = 10
) -> Dict[str, float]:
    """Train a multimodal world model.
    
    Args:
        model: The model to train
        dataloader: Training data loader
        epochs: Number of training epochs
        
    Returns:
        Dictionary containing training metrics
        
    Example:
        >>> model = TransformerWorldModel()
        >>> metrics = train_model(model, train_loader, epochs=50)
        >>> print(f"Final loss: {metrics['loss']:.4f}")
    """
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models/test_transformer_world_model.py -v

# Run with coverage
pytest tests/ --cov=openworld --cov-report=html
```

### Writing Tests

- Write tests for all new functionality
- Aim for >90% test coverage
- Use descriptive test names
- Follow the Arrange-Act-Assert pattern

Example:
```python
def test_model_forward_pass():
    """Test that model forward pass produces expected output shapes."""
    # Arrange
    model = TransformerWorldModel(img_size=64, embed_dim=256)
    video = torch.randn(2, 8, 3, 64, 64)
    audio = torch.randn(2, 8, 128)
    
    # Act
    outputs = model(video=video, audio=audio)
    
    # Assert
    assert 'reconstruction' in outputs
    assert outputs['reconstruction']['video'].shape == video.shape
    assert outputs['reconstruction']['audio'].shape == audio.shape
```

### Test Categories

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete workflows
- **Performance tests**: Test speed and memory usage

## Documentation

### Types of Documentation

1. **Code documentation**: Docstrings and inline comments
2. **API documentation**: Comprehensive API reference
3. **Tutorials**: Step-by-step guides for common tasks
4. **Examples**: Code examples and notebooks

### Documentation Guidelines

- Keep documentation up-to-date with code changes
- Use clear, concise language
- Include code examples where appropriate
- Test all code examples to ensure they work

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs/
make html

# View documentation
open _build/html/index.html
```

## Submitting Changes

### Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write code following our guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**:
   ```bash
   pytest tests/ -v
   black openworld/
   ruff check openworld/
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a pull request**:
   - Go to GitHub and create a pull request
   - Use the pull request template
   - Provide a clear description of changes

### Pull Request Template

```markdown
## Description
Brief description of the changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
```

## Code Review Process

### For Contributors

- Be open to feedback and suggestions
- Respond to review comments promptly
- Make requested changes in a timely manner
- Ask questions if feedback is unclear

### Review Criteria

Reviews will focus on:

1. **Correctness**: Does the code work as intended?
2. **Style**: Does it follow our coding standards?
3. **Testing**: Are there adequate tests?
4. **Documentation**: Is it properly documented?
5. **Performance**: Are there any performance concerns?
6. **Maintainability**: Is the code easy to understand and maintain?

### Review Timeline

- Initial review: Within 2-3 business days
- Follow-up reviews: Within 1-2 business days
- Merge: After approval from at least one maintainer

## Performance Guidelines

### Code Performance

- Profile performance-critical code paths
- Use appropriate data structures and algorithms
- Consider memory usage and optimization
- Benchmark changes that might affect performance

### Testing Performance

```python
import pytest
import time

def test_model_inference_speed():
    """Test that model inference meets performance requirements."""
    model = TransformerWorldModel()
    video = torch.randn(1, 8, 3, 128, 128)
    audio = torch.randn(1, 8, 128)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model(video=video, audio=audio)
    inference_time = time.time() - start_time
    
    # Should complete inference in under 1 second
    assert inference_time < 1.0
```

## Security Guidelines

- Never commit sensitive information (API keys, passwords, etc.)
- Use environment variables for configuration
- Follow security best practices for dependencies
- Report security vulnerabilities privately

## Getting Help

### Communication Channels

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For general questions and discussions
- **Email**: nikjois@llamasearch.ai for sensitive matters

### Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Python Style Guide](https://pep8.org/)
- [Git Best Practices](https://git-scm.com/book/en/v2)

## Recognition

We appreciate all contributions and will recognize contributors in:

- Release notes
- Contributors file
- Project documentation

Thank you for contributing to OpenWorld-Multimodal! 