import { defineConfig } from 'vitest/config';
import path from 'node:path';

const isIntegration = process.env.TEST_ENV === 'integration';

export default defineConfig({
  test: {
    setupFiles: ['dotenv/config', 'tests/setup.ts'],
    testTimeout: 30000,
    globals: true,
    environment: 'node',
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      include: ['src/**'],
      exclude: [
        'node_modules/',
        'dist/',
        'tests/',
        'examples/',
        '**/*.config.ts',
        '**/*.d.ts',
        '**/types/**',
        '**/index.ts',
        'src/bin/cli.ts',
      ],
    },
    include: isIntegration ? ['tests/integrations/**/*.test.ts'] : ['tests/**/*.test.ts'],
    exclude: isIntegration
      ? ['node_modules/', 'dist/']
      : ['node_modules/', 'dist/', 'tests/integrations/**'],
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
