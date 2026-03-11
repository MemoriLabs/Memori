import { quotaCommand } from './commands/quota.js';
import { helpCommand } from './commands/help.js';

const commands: Record<string, ((args: string[]) => Promise<void>) | undefined> = {
  quota: quotaCommand,
  help: helpCommand,
};

export async function main() {
  const args = process.argv.slice(2);
  const commandName = args[0] || 'help';
  const commandArgs = args.slice(1);

  const handler = commands[commandName];

  if (!handler) {
    console.error(`\nError: Unknown command '${commandName}'\n`);
    await helpCommand([]);
    process.exit(1);
  }

  try {
    await handler(commandArgs);
  } catch (error) {
    console.error(`\nUnexpected error executing command '${commandName}':`);
    console.error(error instanceof Error ? error.message : error);
    process.exit(1);
  }
}
