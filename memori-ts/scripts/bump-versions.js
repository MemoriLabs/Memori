import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Paths to the two package.json files
const tsPackagePath = path.resolve(__dirname, '../package.json');
const nativePackagePath = path.resolve(__dirname, '../../core/bindings/node/package.json');

function bumpVersions() {
  console.log('[Memori Version Sync] Starting...');

  // 1. Read the master version from memori-ts
  const tsPkg = JSON.parse(fs.readFileSync(tsPackagePath, 'utf-8'));
  const masterVersion = tsPkg.version;
  console.log(`[Memori Version Sync] Master version is: ${masterVersion}`);

  // 2. Update optionalDependencies in memori-ts
  let depsUpdated = 0;
  if (tsPkg.optionalDependencies) {
    for (const [depName, depVersion] of Object.entries(tsPkg.optionalDependencies)) {
      if (depName.startsWith('@memori/native-')) {
        tsPkg.optionalDependencies[depName] = masterVersion;
        depsUpdated++;
      }
    }
  }
  
  fs.writeFileSync(tsPackagePath, JSON.stringify(tsPkg, null, 2) + '\n', 'utf-8');
  console.log(`[Memori Version Sync] Updated ${depsUpdated} optional dependencies in memori-ts.`);

  // 3. Update the native package version
  if (fs.existsSync(nativePackagePath)) {
    const nativePkg = JSON.parse(fs.readFileSync(nativePackagePath, 'utf-8'));
    nativePkg.version = masterVersion;
    fs.writeFileSync(nativePackagePath, JSON.stringify(nativePkg, null, 2) + '\n', 'utf-8');
    console.log(`[Memori Version Sync] Updated @memori/native version to ${masterVersion}.`);
  } else {
    console.warn(`[Memori Version Sync] Could not find native package.json at ${nativePackagePath}`);
  }

  console.log('[Memori Version Sync] Complete! \n');
}

bumpVersions();