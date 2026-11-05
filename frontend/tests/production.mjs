import { spawn } from "node:child_process";
import assert from "node:assert/strict";
import { browserChecks } from "./browser-checks.mjs";
const base = "/CSE3141-Predictive-Analytics/";
const server = spawn(
  process.execPath,
  [
    "node_modules/vite/bin/vite.js",
    "preview",
    "--host",
    "127.0.0.1",
    "--port",
    "4173",
    "--strictPort",
  ],
  { env: { ...process.env, VITE_BASE_PATH: base }, stdio: "pipe" },
);
let output = "";
server.stdout.on("data", (b) => (output += b));
server.stderr.on("data", (b) => (output += b));
try {
  let ready = false;
  for (let i = 0; i < 100; i++) {
    try {
      if ((await fetch(`http://127.0.0.1:4173${base}`)).ok) {
        ready = true;
        break;
      }
    } catch {}
    if (server.exitCode !== null) throw new Error(output);
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  assert.ok(ready, output);
  await browserChecks(`http://127.0.0.1:4173${base}`);
} finally {
  server.kill();
}
