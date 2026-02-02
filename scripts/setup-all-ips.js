#!/usr/bin/env node

/**
 * Master setup script - coordinates IP detection for all applications
 * Runs setup for mobile and web frontends
 */
const { execSync } = require("child_process");
const path = require("path");

const rootDir = path.join(__dirname, "..");

const setupTasks = [
  {
    name: "Mobile Frontend",
    cwd: path.join(rootDir, "apps/mobile/mobile-frontend"),
    script: "node setup-env.js",
  },
  {
    name: "Web Frontend",
    cwd: path.join(rootDir, "apps/web/web-frontend"),
    script: "node setup-env.js",
  },
];

console.log(`\n╔════════════════════════════════════════════════════╗`);
console.log(`║ Setting up local IP for all applications         ║`);
console.log(`╚════════════════════════════════════════════════════╝\n`);

let hasErrors = false;

for (const task of setupTasks) {
  try {
    console.log(`⏳ Setting up ${task.name}...`);
    execSync(task.script, { cwd: task.cwd, stdio: "inherit" });
    console.log(`✓ ${task.name} setup complete\n`);
  } catch (err) {
    console.error(`✗ ${task.name} setup failed`);
    hasErrors = true;
  }
}

if (hasErrors) {
  console.error(`\n⚠ Some setup tasks failed. Check the errors above.`);
  process.exit(1);
} else {
  console.log(`╔════════════════════════════════════════════════════╗`);
  console.log(`║ All applications ready!                           ║`);
  console.log(`╚════════════════════════════════════════════════════╝\n`);
}
