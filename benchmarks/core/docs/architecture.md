# Architecture Overview

This repository is a simplified Rust workspace for the Memori engine.

## Layout

- Root `src/`: the main engine crate.
- `bindings/python`: PyO3 adapter for Python SDK calls.
- `bindings/node`: napi-rs adapter for Node SDK calls.
- `examples/`: small integration demos.

## Dependency Direction

Dependencies must remain one-way:

root engine crate -> adapters (`bindings/python`, `bindings/node`)

No reverse dependencies are allowed.

## Responsibility Split

### Root engine crate

- Owns the current Rust-side engine behavior.
- Handles deterministic command behavior and input validation.
- Accepts postprocess work and hands it off to a non-blocking background runtime.
- Keeps background runtime code as an internal module, not a separate package.

### `bindings/python` and `bindings/node`

- Expose thin language-native functions and exceptions/errors.
- Convert SDK inputs into engine calls.
- Must not bypass the engine crate and call runtime internals directly.

## Contribution Guardrails

- Keep the root engine crate simple and cohesive while the product shape is still evolving.
- Keep Python and Node adapters as thin conversion layers.
- Avoid reintroducing premature internal package boundaries.
