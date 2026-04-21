import fs from 'node:fs';
import path from 'node:path';
import { execSync } from 'node:child_process';

const ROOT = process.cwd();
const RUST_BINDINGS_DIR = path.resolve(ROOT, '../core/bindings/node');
const SRC_NATIVE = path.resolve(ROOT, 'src/native');
const DIST_NATIVE = path.resolve(ROOT, 'dist/src/native');

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

  // Always ensure the CommonJS bridge config is present
  fs.writeFileSync(path.join(to, 'package.json'), JSON.stringify({ type: 'commonjs' }, null, 2));
}

function sync() {
  console.log('Building Rust artifacts...');
  execSync('npm run build', { cwd: RUST_BINDINGS_DIR, stdio: 'inherit' });

  console.log('Syncing to src/native...');
  copyFolderSync(RUST_BINDINGS_DIR, SRC_NATIVE);

  // If we've already built the TS, sync to dist too so examples run immediately
  if (fs.existsSync(path.resolve(ROOT, 'dist'))) {
    console.log('📦 Syncing to dist/src/native...');
    copyFolderSync(SRC_NATIVE, DIST_NATIVE);
  }

  console.log('✅ Native sync complete.');
}

try {
  sync();
} catch (err) {
  console.error('❌ sync-native failed:', err.message);
  process.exit(1);
}
