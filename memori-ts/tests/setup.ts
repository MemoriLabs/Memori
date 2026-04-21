import { vi } from 'vitest';

/**
 * Unit tests must not load the real `.node` binary (Rust toolchain not required).
 * Integration tests that need the real engine should use a separate setup or unmock.
 */
vi.mock('../src/native/index.js', () => {
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

  return {
    MemoriEngine: vi.fn().mockImplementation(stub),
  };
});
