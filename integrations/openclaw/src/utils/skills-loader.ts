import { readFileSync } from 'fs';

/**
 * Loads the Memori skills document at plugin registration time.
 * Returns an empty string if the file cannot be read so the plugin
 * degrades gracefully rather than failing to register.
 */
export function loadSkillsContent(resolvePath: (input: string) => string): string {
  try {
    return readFileSync(resolvePath('skills/memori/SKILL.md'), 'utf-8');
  } catch {
    return '';
  }
}
