# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `py.typed` marker file for PEP 561 compliance (typed package support)
- `ChatModelProtocol` and `StructuredOutputModelProtocol` in `react_usc.providers.base` for type checking
- Shared salvage utilities in `_internal/salvage.py` for non-JSON output recovery
- Separate test modules for better organization (`test_executors.py`, `test_plugins.py`, `test_decisions.py`, `test_validation.py`)
- Shared test fixtures in `tests/conftest.py`
- This CHANGELOG file

### Changed
- Version is now dynamically read from `__init__.py` (single source of truth)
- Imports in tests and examples now use `react_usc` instead of `src.react_usc`
- Executor classes now use shared salvage functions from `_internal/salvage.py`
- Added `httpx` to `a2a` optional dependencies (required for FastAPI TestClient)

### Fixed
- Provider tests mock patching for Python 3.12+ compatibility (patching lazy imports correctly)
- `MagicMock` protocol conformance test for Python 3.12+ behavior

### Removed
- `src/__init__.py` file (was causing incorrect import patterns)
- `react_usc.models` module (unnecessary re-export layer; import from `react_usc.types`, `react_usc.config`, or `react_usc.decisions` directly)
- `requirements.txt` (redundant; use `pip install -e ".[all]"` with pyproject.toml instead)

## [0.4.0] - 2026-01-27

### Added
- Initial public release
- LangGraph-based ReAct agent with Universal Self-Consistency (USC)
- Parallel reasoning with K paths
- Judge-based decision selection/synthesis
- Tool execution with validation
- Optional retry/reflection plugin for error recovery
- Comprehensive logging and tracing
- Vertex AI provider support
- A2A (Agent-to-Agent) integration

### Features
- `LangGraphReActUSCAgent` - Main agent class
- `LangGraphModels` - Container for LangChain chat models
- `AgentConfig`, `ModelConfig`, `RetryConfig` - Configuration classes
- `ToolSpec`, `ToolRegistry` - Tool management
- `ReflectAndRetryToolPlugin` - Error recovery plugin
- Custom exception hierarchy for detailed error handling

[Unreleased]: https://github.com/react-usc/react-usc/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/react-usc/react-usc/releases/tag/v0.4.0
