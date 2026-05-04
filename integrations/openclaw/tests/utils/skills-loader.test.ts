import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('fs', () => ({
  readFileSync: vi.fn(),
}));

import { loadSkillsContent } from '../../src/utils/skills-loader.js';

describe('utils/skills-loader', () => {
  let mockResolvePath: (input: string) => string;

  beforeEach(() => {
    vi.clearAllMocks();
    mockResolvePath = vi.fn((input: string) => `/resolved/${input}`);
  });

  it('should resolve the skills file path correctly', async () => {
    const { readFileSync } = await import('fs');
    vi.mocked(readFileSync).mockReturnValue('# Skills content');

    loadSkillsContent(mockResolvePath);

    expect(mockResolvePath).toHaveBeenCalledWith('skills/memori/SKILL.md');
    expect(readFileSync).toHaveBeenCalledWith('/resolved/skills/memori/SKILL.md', 'utf-8');
  });

  it('should return the file contents when the file exists', async () => {
    const { readFileSync } = await import('fs');
    vi.mocked(readFileSync).mockReturnValue(
      '# Memori Skills\n\nUse memori_recall to fetch memories.'
    );

    const result = loadSkillsContent(mockResolvePath);

    expect(result).toBe('# Memori Skills\n\nUse memori_recall to fetch memories.');
  });

  it('should return an empty string when the file does not exist', async () => {
    const { readFileSync } = await import('fs');
    vi.mocked(readFileSync).mockImplementation(() => {
      throw new Error('ENOENT: no such file or directory');
    });

    const result = loadSkillsContent(mockResolvePath);

    expect(result).toBe('');
  });

  it('should return an empty string on any read error', async () => {
    const { readFileSync } = await import('fs');
    vi.mocked(readFileSync).mockImplementation(() => {
      throw new Error('Permission denied');
    });

    const result = loadSkillsContent(mockResolvePath);

    expect(result).toBe('');
  });
});
