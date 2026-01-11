# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial harness infrastructure for KernelHeim v0.1
- Three-layer architecture (ops/kernels/ref) for kernel implementations (#5)
- Week 0: `copy_transpose` kernel with CuTe DSL implementation
- Kernel compilation caching in ops layer with symbolic compilation
- Pre-commit hooks configuration with ruff linting and formatting
- GitHub Actions CI workflow for automated code quality checks (#10)
- Benchmark framework with YAML-driven suite system (`bench/run.py`)
- Reference implementations for all Week 0-2 operations
- Comprehensive test suite with PyTorch reference validation
- Environment check utility (`forge_cute_py.env_check`)
- Documentation: README, CONTRIBUTING.md with contribution policy (#7)
- Pre-commit hooks installation instructions in CONTRIBUTING.md (#7)
- Support for float16, bfloat16, and float32 dtypes

### Changed
- **BREAKING:** Renamed `tile` parameter to `tile_size` in `copy_transpose` API (#5)
- Improved test structure with parametrized fixtures for dtypes, shapes, and tile sizes
- Enhanced error messages with device information for better debugging
- Set tolerance to 0 (exact comparison) for transpose correctness tests

### Infrastructure
- Package scaffolding with `pyproject.toml` and proper Python 3.13+ support
- PyTorch ops registration via `torch.library`
- Benchmark utilities with timing statistics (p50/p90/p99) and bandwidth estimation
- Test configuration with pytest
- Ruff configuration for linting and formatting (line-length=100)
- Scripts for architecture detection and profiling setup

## [0.1.0] - Unreleased

Initial development release targeting KernelHeim Weeks 0-2.

### Roadmap
- [x] Week 0: Harness infrastructure with copy/transpose kernel
- [ ] Week 1: Reduction operations (sum) with naive/improved/shuffle variants
- [ ] Week 2: Single-pass online softmax
- [ ] CI: GPU runners for correctness validation

[Unreleased]: https://github.com/Kernel-Heim/forge-cute-py/compare/main...init-harness
