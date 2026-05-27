#!/usr/bin/env node
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import process from 'node:process';

const envPath = resolve(process.cwd(), '.env');
if (existsSync(envPath)) {
  process.loadEnvFile(envPath);
}

import { main } from '../cli/router.js';

void main();
