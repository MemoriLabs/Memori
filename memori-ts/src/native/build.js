const { execSync } = require('child_process');

const env = { ...process.env };
const npx = process.platform === 'win32' ? 'npx.cmd' : 'npx';
const args = ['napi', 'build', '--platform', '--release', ...process.argv.slice(2)];
const command = [npx, ...args].join(' ');

// NAPI-RS forces static C-Runtime on Windows by default.
// We must override it to dynamic (-crt-static) so that ort-sys and tokenizers
// can compile together without LNK2038 linker collisions.
if (process.platform === 'win32') {
  env.RUSTFLAGS = env.RUSTFLAGS
    ? `${env.RUSTFLAGS} -C target-feature=-crt-static`
    : '-C target-feature=-crt-static';
  console.log('Detected Windows: Injecting dynamic C-Runtime RUSTFLAGS...');
}

try {
  // Execute the standard NAPI build command with our modified environment
  execSync(command, { env, stdio: 'inherit' });
} catch (error) {
  process.exit(1);
}
