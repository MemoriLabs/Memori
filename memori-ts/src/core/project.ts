/**
 * Manages the project context for memory operations.
 * Ensures a consistent project ID is associated with all requests.
 */
export class ProjectManager {
  private _id: string | null = null;

  /**
   * The current active project ID, or null if not set.
   */
  public get id(): string | null {
    return this._id;
  }

  /**
   * Sets the project ID.
   *
   * @param id - The project identifier to associate with memory operations.
   */
  public set(id: string): this {
    this._id = id;
    return this;
  }
}
