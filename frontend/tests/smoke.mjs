import { browserChecks } from "./browser-checks.mjs";
await browserChecks(process.env.TEST_URL || "http://127.0.0.1:5173/", {
  liveApi: true,
});
