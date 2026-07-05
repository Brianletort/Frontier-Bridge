# Security Policy

## Reporting a vulnerability

Please report security issues privately via [GitHub Security Advisories](https://github.com/Brianletort/Frontier-Bridge/security/advisories/new) rather than public issues. You should receive an acknowledgment within 7 days.

## Scope

Frontier Bridge is a planning and benchmarking layer. Security-relevant surfaces include:

- **`frontier detect`** runs local system tools (`system_profiler`, `sysctl`, `nvidia-smi`, `lsblk`, bounded disk-read benchmarks). It never uploads data; profiles are written locally and shared only when you commit them. Review profiles for sensitive details (hostnames, serial numbers) before submitting.
- **Model artifacts** referenced in model profiles are hash-pinned (`sha256`). Always verify hashes before loading third-party GGUF files; treat model files as untrusted input to your runtime.
- **Generated launch commands** are printed for review, not executed silently. Read the command before running it.
- **YAML parsing** uses `yaml.safe_load` exclusively. Contributions using unsafe loaders will be rejected.

## Supported versions

Pre-release (`0.1.0.dev`): fixes land on `main` only.
