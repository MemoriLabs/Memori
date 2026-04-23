const { execSync } = require('child_process');

const env = { ...process.env };

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
  execSync('npx napi build --platform --release', { env, stdio: 'inherit' });
} catch (error) {
  process.exit(1);
}