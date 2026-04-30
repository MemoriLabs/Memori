import fs from 'node:fs';
import path from 'node:path';
import { execSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const RUST_BINDINGS_DIR = path.resolve(ROOT, '../core/bindings/node');
const SRC_NATIVE = path.resolve(ROOT, 'src/native');

// CHECK FOR FLAG: Did we run this with "node scripts/sync-native.js --dev"?
const isDev = process.argv.includes('--dev');

// Set target dynamically based on the flag
const DIST_NATIVE = path.resolve(ROOT, isDev ? 'dist/src/native' : 'dist/native');

function copyFolderSync(from, to) {
  if (!fs.existsSync(from)) return;
  if (fs.existsSync(to)) fs.rmSync(to, { recursive: true, force: true });
  fs.mkdirSync(to, { recursive: true });

  const files = fs.readdirSync(from);
  const extensions = ['.js', '.d.ts', '.node'];

  for (const file of files) {
    if (extensions.some((ext) => file.endsWith(ext))) {
      fs.copyFileSync(path.join(from, file), path.join(to, file));
    }
  }

  fs.writeFileSync(path.join(to, 'package.json'), JSON.stringify({ type: 'commonjs' }, null, 2));
}

function sync() {
  if (!fs.existsSync(path.join(RUST_BINDINGS_DIR, 'node_modules'))) {
    console.log('Installing dependencies in core/bindings/node (first time)...');
    execSync('npm ci', { cwd: RUST_BINDINGS_DIR, stdio: 'inherit' });
  }

  console.log('Building Rust N-API artifacts...');
  execSync('npm run build', { cwd: RUST_BINDINGS_DIR, stdio: 'inherit' });

  console.log('Syncing to src/native...');
  copyFolderSync(RUST_BINDINGS_DIR, SRC_NATIVE);

  // Copy to the single correct distribution folder
  console.log(`Syncing to ${isDev ? 'dist/src/native (Dev)' : 'dist/native (Prd)'}...`);
  copyFolderSync(SRC_NATIVE, DIST_NATIVE);

  console.log('Native sync complete.');
}

try {
  sync();
} catch (err) {
  console.error('sync-native failed:', err instanceof Error ? err.message : err);
  process.exit(1);
}
