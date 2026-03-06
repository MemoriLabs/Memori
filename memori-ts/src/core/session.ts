import { randomUUID } from 'node:crypto';

/**
 * Manages the conversation session lifecycle.
 * Ensures consistent session IDs across requests to maintain conversation history.
 */
export class SessionManager {
  private _id: string;

  constructor() {
    this._id = randomUUID();
  }

  /**
   * The current active session UUID.
   */
  public get id(): string {
    return this._id;
  }

  /**
   * Generates a brand new random UUID for the session.
   * Use this to clear context and start fresh.
   */
  public reset(): this {
    this._id = randomUUID();
    return this;
  }

  /**
   * Manually sets the session ID to a specific value.
   * Useful for resuming a conversation from a database or frontend client.
   *
   * @param id - The UUID to reuse.
   */
  public set(id: string): this {
    this._id = id;
    return this;
  }
}
