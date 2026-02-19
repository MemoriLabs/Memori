import { describe, it, expect, vi } from 'vitest';
import { Memori } from '../src/memori.js';
import { SessionManager } from '../src/core/session.js';
import { Config } from '../src/core/config.js';

describe('Memori SDK', () => {
  it('should instantiate with default components', () => {
    const memori = new Memori();

    expect(memori.config).toBeInstanceOf(Config);
    expect(memori.session).toBeInstanceOf(SessionManager);
    expect(memori.axon).toBeDefined();
    expect(memori.llm).toBeDefined();
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
    // Spy on the internal axon.llm.register method and mock implementation
    // to prevent the real registry from throwing validation errors on our mock object
    const registerSpy = vi.spyOn(memori.axon.llm, 'register').mockImplementation(() => ({}) as any);
    const mockClient = { name: 'mock-client' };

    const result = memori.llm.register(mockClient);

    expect(registerSpy).toHaveBeenCalledWith(mockClient);
    expect(result).toBe(memori); // Check chaining
  });
});
