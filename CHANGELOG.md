# Changelog

All notable changes to the Memori Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive end-to-end CLI tests covering help variations, command validation, and error handling
- Tests for CockroachDB cluster management commands (start, claim, delete)
- ASCII logo display verification test
- Parametrized tests for multiple help invocation methods (no args, --help, -h, help)

### Changed
- Refactored existing CLI tests to use behavior-based assertions instead of exact string matching
- Improved test assertions to validate both exit codes and error messages

[3.0.0]: https://github.com/MemoriLabs/Memori/releases/tag/v3.0.0
