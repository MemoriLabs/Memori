import { describe, it, expect, beforeEach } from 'vitest';
import { ProjectManager } from '../../src/core/project.js';

describe('ProjectManager', () => {
  let project: ProjectManager;

  beforeEach(() => {
    project = new ProjectManager();
  });

  it('should initialize with a null id', () => {
    expect(project.id).toBeNull();
  });

  it('should set the project id', () => {
    project.set('proj-123');
    expect(project.id).toBe('proj-123');
  });

  it('should overwrite a previously set id', () => {
    project.set('proj-123');
    project.set('proj-456');
    expect(project.id).toBe('proj-456');
  });

  it('should support chaining', () => {
    const result = project.set('proj-abc');
    expect(result).toBeInstanceOf(ProjectManager);
  });
});
