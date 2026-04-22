import { describe, it, expect, vi } from 'vitest';
import { Memori } from '../src/memori.js';
import { SessionManager } from '../src/core/session.js';
import { Config } from '../src/core/config.js';

// Mock the storage manager so we don't need real DB adapters in unit tests
vi.mock('../src/storage/manager.js', () => ({
  StorageManager: vi
    .fn()
    .mockImplementation(() => ({ setEmbedder: vi.fn(), setEngineShutdown: vi.fn() })),
}));

describe('Memori SDK', () => {
  it('should instantiate with default components', () => {
    const memori = new Memori();

    expect(memori.config).toBeInstanceOf(Config);
    expect(memori.session).toBeInstanceOf(SessionManager);
    expect(memori.axon).toBeDefined();
    expect(memori.llm).toBeDefined();
    expect(memori.config.storage).toBeUndefined();
  });

  it('should instantiate StorageManager when a database connection is provided', () => {
    const mockDbConnection = { dummyDb: true };
    const memori = new Memori({ conn: mockDbConnection });

    expect(memori.config.storage).toBeDefined();
  });

  it('should update attribution config correctly', () => {
    const memori = new Memori();

    memori.attribution('user-123', 'process-xyz');

    expect(memori.config.entityId).toBe('user-123');
    expect(memori.config.processId).toBe('process-xyz');
  });

  it('should reset session correctly', () => {
    const memori = new Memori();
    const oldId = memori.session.id;

    memori.resetSession();

    expect(memori.session.id).not.toBe(oldId);
  });

  it('should set session correctly', () => {
    const memori = new Memori();
    const specificId = 'uuid-123-456';

    memori.setSession(specificId);

    expect(memori.session.id).toBe(specificId);
  });

  it('should register an LLM client via the llm helper', () => {
    const memori = new Memori();
    // Spy on the internal axon.llm.register method
    const registerSpy = vi.spyOn(memori.axon.llm, 'register').mockImplementation(() => ({}) as any);
    const mockClient = { name: 'mock-client' };

    const result = memori.llm.register(mockClient);

    expect(registerSpy).toHaveBeenCalledWith(mockClient);
    expect(result).toBe(memori); // Check chaining
  });

  it('should expose augmentation.wait() helper', async () => {
    const memori = new Memori();
    const waitSpy = vi.spyOn(memori.engine, 'waitForAugmentation').mockResolvedValue(true);

    const ok = await memori.augmentation.wait(100);

    expect(waitSpy).toHaveBeenCalledWith(100);
    expect(ok).toBe(true);
  });
});
