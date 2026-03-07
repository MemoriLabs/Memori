import type { OpenClawPluginApi } from 'openclaw/plugin-sdk';
import { PLUGIN_CONFIG } from '../constants.js';

export class MemoriLogger {
  constructor(private api: OpenClawPluginApi) {}

  private prefix(msg: string): string {
    return `${PLUGIN_CONFIG.LOG_PREFIX} ${msg}`;
  }

  info(message: string): void {
    this.api.logger.info(this.prefix(message));
  }

  warn(message: string): void {
    this.api.logger.warn(this.prefix(message));
  }

  error(message: string): void {
    this.api.logger.error(this.prefix(message));
  }

  section(title: string): void {
    this.api.logger.info(`\n=== ${this.prefix(title)} ===`);
  }

  endSection(title: string): void {
    this.api.logger.info(`=== ${this.prefix(title)} ===\n`);
  }
}
