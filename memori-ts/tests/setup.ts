import { vi } from 'vitest';
import { createRequire } from 'node:module';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

// vi.hoisted ensures the mock objects exist before vi.mock factories and before imports run.
// This gives us a single shared MemoriEngine vi.fn() instance that both the ESM import
// (test file) and the createRequire path (engine.ts lazy load) refer to.
const { mockNativeModule } = vi.hoisted(() => {
  const stub = () => ({
    retrieve: vi.fn().mockResolvedValue([]),
    recall: vi.fn().mockResolvedValue(''),
    embedTexts: vi.fn().mockReturnValue([]),
    submitAugmentation: vi.fn().mockReturnValue('00000000-0000-0000-0000-000000000000'),
    waitForAugmentation: vi.fn().mockResolvedValue(true),
    shutdown: vi.fn(),
    resolveEmbeddingsCallback: vi.fn(),
    resolveFactsCallback: vi.fn(),
    resolveWriteCallback: vi.fn(),
  });

  return { mockNativeModule: { MemoriEngine: vi.fn().mockImplementation(stub) } };
});

/**
 * Unit tests must not load the real `.node` binary (Rust toolchain not required).
 * Integration tests that need the real engine should use a separate setup or unmock.
 */
vi.mock('../src/native/index.js', () => mockNativeModule);

// engine.ts uses createRequire() (CJS) to lazily load the native module, which bypasses
// vitest's ESM mock system. Pre-populate Node's require.cache at the resolved absolute path
// so that any createRequire(url)('../native/index.js') call returns the same mock.
const _nativePath = resolve(dirname(fileURLToPath(import.meta.url)), '../src/native/index.js');
const _req = createRequire(import.meta.url);
(_req.cache as Record<string, unknown>)[_nativePath] = {
  id: _nativePath,
  filename: _nativePath,
  loaded: true,
  exports: mockNativeModule,
  parent: null,
  children: [],
  paths: [],
};
