import { describe, it, expect } from 'vitest';
import { extractContext } from '../../src/utils/context.js';
import type { OpenClawEvent, OpenClawContext } from '../../src/types.js';

describe('utils/context', () => {
  describe('extractContext', () => {
    const configuredEntityId = 'test-entity-123';

    it('should extract context successfully with all required fields', () => {
      const event: OpenClawEvent = {
        sessionId: 'event-session-456',
        messageProvider: 'event-provider',
      };
      const ctx: OpenClawContext = {
        sessionKey: 'ctx-session-789',
        messageProvider: 'ctx-provider',
      };

      const result = extractContext(event, ctx, configuredEntityId);

      expect(result).toEqual({
        entityId: 'test-entity-123',
        sessionId: 'ctx-session-789',
        provider: 'ctx-provider',
      });
    });

    it('should use event sessionId when ctx.sessionKey is missing', () => {
      const event: OpenClawEvent = {
        sessionId: 'event-session-456',
        messageProvider: 'event-provider',
      };
      const ctx: OpenClawContext = {
        messageProvider: 'ctx-provider',
      };

      const result = extractContext(event, ctx, configuredEntityId);

      expect(result.sessionId).toBe('event-session-456');
    });

    it('should use event messageProvider when ctx.messageProvider is missing', () => {
      const event: OpenClawEvent = {
        sessionId: 'event-session-456',
        messageProvider: 'event-provider',
      };
      const ctx: OpenClawContext = {
        sessionKey: 'ctx-session-789',
      };

      const result = extractContext(event, ctx, configuredEntityId);

      expect(result.provider).toBe('event-provider');
    });

    it('should throw error when sessionId cannot be determined', () => {
      const event: OpenClawEvent = {
        messageProvider: 'event-provider',
      };
      const ctx: OpenClawContext = {
        messageProvider: 'ctx-provider',
      };

      expect(() => extractContext(event, ctx, configuredEntityId)).toThrow(
        'Failed to extract context: Missing sessionId in OpenClaw context and event.'
      );
    });

    it('should throw error when provider cannot be determined', () => {
      const event: OpenClawEvent = {
        sessionId: 'event-session-456',
      };
      const ctx: OpenClawContext = {
        sessionKey: 'ctx-session-789',
      };

      expect(() => extractContext(event, ctx, configuredEntityId)).toThrow(
        'Failed to extract context: Missing message provider in OpenClaw context and event.'
      );
    });

    it('should throw error when both sessionId and provider are missing', () => {
      const event: OpenClawEvent = {};
      const ctx: OpenClawContext = {};

      expect(() => extractContext(event, ctx, configuredEntityId)).toThrow();
    });

    it('should handle empty string sessionId as missing', () => {
      const event: OpenClawEvent = {
        messageProvider: 'provider',
      };
      const ctx: OpenClawContext = {
        sessionKey: '',
        messageProvider: 'ctx-provider',
      };

      expect(() => extractContext(event, ctx, configuredEntityId)).toThrow(
        'Failed to extract context: Missing sessionId'
      );
    });

    it('should handle empty string provider as missing', () => {
      const event: OpenClawEvent = {
        sessionId: 'session-123',
      };
      const ctx: OpenClawContext = {
        sessionKey: 'ctx-session-789',
        messageProvider: '',
      };

      expect(() => extractContext(event, ctx, configuredEntityId)).toThrow(
        'Failed to extract context: Missing message provider'
      );
    });

    it('should always use configuredEntityId from plugin config', () => {
      const event: OpenClawEvent = {
        sessionId: 'session-123',
        messageProvider: 'provider',
        userId: 'different-user-id',
      };
      const ctx: OpenClawContext = {
        sessionKey: 'session-789',
        messageProvider: 'ctx-provider',
        agentId: 'agent-id',
      };

      const result = extractContext(event, ctx, 'my-configured-entity');

      expect(result.entityId).toBe('my-configured-entity');
    });
  });
});
