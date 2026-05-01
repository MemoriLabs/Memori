import { Config } from '../../core/config.js';
import { Api } from '../../core/network.js';
import { MemoriApiValidationError } from '../../core/errors.js';
import { printBanner } from '../utils.js';

interface SignUpResponse {
  content?: string;
}

export async function signupCommand(args: string[]): Promise<void> {
  printBanner();

  const email = args[0];

  if (!email) {
    console.log('Usage: memori sign-up <email_address>\n');
    process.exit(1);
  }

  const config = new Config();
  const api = new Api(config);

  try {
    const response = await api.post<SignUpResponse>('sdk/account', { email });
    console.log(response.content || "You're all set! We sent you an email.\n");
  } catch (error) {
    if (error instanceof MemoriApiValidationError) {
      console.log(`The email you provided "${email}" is not valid.\n`);
    } else {
      console.error('An unexpected error occurred during sign-up.');
      if (error instanceof Error) {
        console.error(`Error details: ${error.message}\n`);
      }
      process.exit(1);
    }
  }
}
