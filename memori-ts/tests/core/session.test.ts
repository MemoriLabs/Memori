import { describe, it, expect, beforeEach } from 'vitest';
import { SessionManager } from '../../src/core/session.js';

describe('SessionManager', () => {
  let session: SessionManager;

  beforeEach(() => {
    session = new SessionManager();
  });

  it('should generate a valid UUID on initialization', () => {
    expect(session.id).toBeDefined();
    expect(typeof session.id).toBe('string');
    expect(session.id.length).toBeGreaterThan(0);
  });

  it('should reset the session ID to a new UUID', () => {
    const originalId = session.id;
    session.reset();
    expect(session.id).not.toBe(originalId);
    expect(session.id).toBeDefined();
  });

  it('should allow manually setting the session ID', () => {
    const customId = 'custom-session-id';
    session.set(customId);
    expect(session.id).toBe(customId);
  });

  it('should support chaining for methods', () => {
    const result = session.reset();
    expect(result).toBeInstanceOf(SessionManager);

    const result2 = session.set('chain-test');
    expect(result2).toBeInstanceOf(SessionManager);
  });
});
