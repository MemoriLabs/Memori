import { describe, it, expect, vi, beforeEach } from 'vitest';
import { MemoriLogger } from '../../src/utils/logger.js';
import type { OpenClawPluginApi } from 'openclaw/plugin-sdk';

describe('utils/logger', () => {
  let mockApi: OpenClawPluginApi;
  let logger: MemoriLogger;

  beforeEach(() => {
    mockApi = {
      logger: {
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
      },
    } as unknown as OpenClawPluginApi;

    logger = new MemoriLogger(mockApi);
  });

  describe('info', () => {
    it('should call api.logger.info with prefixed message', () => {
      logger.info('Test message');
      expect(mockApi.logger.info).toHaveBeenCalledWith('[Memori] Test message');
    });

    it('should handle empty strings', () => {
      logger.info('');
      expect(mockApi.logger.info).toHaveBeenCalledWith('[Memori] ');
    });

    it('should preserve message content exactly', () => {
      logger.info('Message with    spaces\nand newlines');
      expect(mockApi.logger.info).toHaveBeenCalledWith(
        '[Memori] Message with    spaces\nand newlines'
      );
    });
  });

  describe('warn', () => {
    it('should call api.logger.warn with prefixed message', () => {
      logger.warn('Warning message');
      expect(mockApi.logger.warn).toHaveBeenCalledWith('[Memori] Warning message');
    });
  });

  describe('error', () => {
    it('should call api.logger.error with prefixed message', () => {
      logger.error('Error message');
      expect(mockApi.logger.error).toHaveBeenCalledWith('[Memori] Error message');
    });

    it('should handle error messages with special characters', () => {
      logger.error('Error: Failed to connect (code: 500)');
      expect(mockApi.logger.error).toHaveBeenCalledWith(
        '[Memori] Error: Failed to connect (code: 500)'
      );
    });
  });

  describe('section', () => {
    it('should log section start with formatting', () => {
      logger.section('HOOK START');
      expect(mockApi.logger.info).toHaveBeenCalledWith('\n=== [Memori] HOOK START ===');
    });

    it('should handle multi-word section titles', () => {
      logger.section('AUGMENTATION HOOK START');
      expect(mockApi.logger.info).toHaveBeenCalledWith(
        '\n=== [Memori] AUGMENTATION HOOK START ==='
      );
    });
  });

  describe('endSection', () => {
    it('should log section end with formatting', () => {
      logger.endSection('HOOK END');
      expect(mockApi.logger.info).toHaveBeenCalledWith('=== [Memori] HOOK END ===\n');
    });
  });

  describe('prefix consistency', () => {
    it('should use consistent prefix across all log levels', () => {
      logger.info('info');
      logger.warn('warn');
      logger.error('error');

      expect(mockApi.logger.info).toHaveBeenCalledWith('[Memori] info');
      expect(mockApi.logger.warn).toHaveBeenCalledWith('[Memori] warn');
      expect(mockApi.logger.error).toHaveBeenCalledWith('[Memori] error');
    });
  });

  describe('call count verification', () => {
    it('should call underlying logger exactly once per method call', () => {
      logger.info('test');
      expect(mockApi.logger.info).toHaveBeenCalledTimes(1);

      logger.warn('test');
      expect(mockApi.logger.warn).toHaveBeenCalledTimes(1);

      logger.error('test');
      expect(mockApi.logger.error).toHaveBeenCalledTimes(1);
    });

    it('should accumulate calls correctly', () => {
      logger.info('first');
      logger.info('second');
      logger.info('third');

      expect(mockApi.logger.info).toHaveBeenCalledTimes(3);
    });
  });
});
