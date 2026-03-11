import { Config } from '../../core/config.js';
import { Api } from '../../core/network.js';
import { printBanner } from '../utils.js';

interface QuotaResponse {
  memories: {
    max: number;
    num: number;
  };
  message: string;
}

export async function quotaCommand(_args: string[]): Promise<void> {
  const config = new Config();
  const api = new Api(config);

  printBanner();

  try {
    const response = await api.get<QuotaResponse>('sdk/quota');

    console.log(`+ Maximum # of Memories: ${response.memories.max.toLocaleString()}`);
    console.log(`+ Current # of Memories: ${response.memories.num.toLocaleString()}\n`);
    console.log(`+ ${response.message}\n`);
  } catch (error) {
    console.error('Failed to fetch quota. Please check your MEMORI_API_KEY.');
    if (error instanceof Error) {
      console.error(`Error details: ${error.message}`);
    }
    process.exit(1);
  }
}
