# Project Context: Dear

## Project Overview
`dear` is a Python project currently in its initial setup phase. Based on the dependencies, it appears intended for numerical computing or machine learning tasks, utilizing:
- **JAX**: For high-performance numerical computing and autograd.
- **Equinox**: A JAX-based library for neural networks.

## Architecture & Structure
- **`main.py`**: The current entry point for the application.
- **`pyproject.toml`**: Project configuration and dependency definition.
- **`uv.lock`**: Lock file for reproducible dependencies, managed by `uv`.

## Building and Running

### Prerequisites
- Python 3.12 or higher.
- `uv` package manager is used for this project.

### Setup
To install dependencies:
```bash
uv sync
```

### Running the Application
To run the main script:
```bash
uv run main.py
```
Or, if you have activated the virtual environment:
```bash
python main.py
```

## Development Conventions
- **Dependency Management**: Controlled via `pyproject.toml` and `uv`.
- **Code Style**: Standard Python conventions apply.
