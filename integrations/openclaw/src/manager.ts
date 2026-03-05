import { Memori } from '@memorilabs/memori';
import { SessionData } from './types.js';
import { SESSION_CONFIG } from './constants.js';

const activeSessions = new Map<string, SessionData>();
let lastGcTime = Date.now();

/**
 * Passive garbage collector to clean up stale sessions.
 * Removes sessions that haven't been accessed within the TTL period.
 * Throttled to only run at most once per GC_INTERVAL_MS to maintain high performance.
 */
function runGarbageCollection(): void {
  const now = Date.now();
  
  // Only run if the interval has passed since the last GC
  if (now - lastGcTime < SESSION_CONFIG.GC_INTERVAL_MS) {
    return;
  }

  for (const [entityId, data] of activeSessions.entries()) {
    if (now - data.lastAccessed > SESSION_CONFIG.TTL_MS) {
      activeSessions.delete(entityId);
    }
  }
  
  lastGcTime = now;
}

/**
 * Gets an existing session or creates a new one for the given entity.
 * Automatically updates the lastAccessed timestamp and passively triggers GC.
 *
 * @param entityId - Unique identifier for the entity (user/agent)
 * @param apiKey - Memori API key for initializing new sessions
 * @returns Session data containing Memori instance and metadata
 */
export function getOrCreateSession(entityId: string, apiKey: string): SessionData {
  // Passively trigger GC check before allocating/accessing
  runGarbageCollection();

  // Fetch or create the session
  let session = activeSessions.get(entityId);

  if (!session) {
    const memori = new Memori();
    memori.config.apiKey = apiKey;
    session = { memori, lastAccessed: Date.now() };
    activeSessions.set(entityId, session);
  } else {
    session.lastAccessed = Date.now();
  }

  return session;
}

/**
 * Clears the session for a specific entity.
 * Useful for testing or forcing a fresh session.
 *
 * @param entityId - Entity ID to clear
 */
export function clearLocalSession(entityId: string): void {
  activeSessions.delete(entityId);
}