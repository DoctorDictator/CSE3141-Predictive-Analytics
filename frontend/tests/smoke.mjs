import { chromium } from "playwright";
import assert from "node:assert/strict";
import fs from "node:fs";
const browser = await chromium.launch(
  process.platform === "win32"
    ? {
        executablePath:
          "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
        headless: true,
      }
    : { headless: true },
);
const page = await browser.newPage({ viewport: { width: 1440, height: 1050 } });
await page.emulateMedia({ reducedMotion: "reduce" });
const base =
  process.env.TEST_URL || "http://127.0.0.1:5173/CSE3141-Predictive-Analytics/";
const errors = [];
page.on("pageerror", (error) => errors.push(error.message));
const output = process.env.QA_SCREENSHOT_DIR;
if (output) fs.mkdirSync(output, { recursive: true });
await page.goto(base);
await page.getByRole("heading", { name: "Traffic, in perspective." }).waitFor();
assert.equal(
  await page.locator("body").evaluate((el) => el.scrollWidth > innerWidth),
  false,
);
if (output)
  await page.screenshot({
    path: `${output}/overview-desktop.png`,
    fullPage: true,
  });
for (const [route, title] of [
  ["analysis", "Patterns behind the numbers."],
  ["performance", "How well does it hold up?"],
  ["about", "A simple model. A transparent story."],
]) {
  await page.goto(`${base}#/${route}`);
  await page.getByRole("heading", { name: title, exact: true }).waitFor();
}
await page.goto(`${base}#/prediction`);
await page
  .getByRole("heading", { name: "What does this hour look like?" })
  .waitFor();
await page.locator('input[name="temp_c"]').fill("90");
await page
  .getByRole("button", { name: "Predict Traffic", exact: true })
  .click();
assert.equal(
  await page
    .locator('input[name="temp_c"]')
    .evaluate((el) => el.validity.valid),
  false,
);
await page.locator('input[name="temp_c"]').fill("18");
if (process.env.EXPECT_API !== "false") {
  const pending = page.waitForResponse(
    (response) =>
      response.url().endsWith("/api/predict") &&
      response.request().method() === "POST",
  );
  await page
    .getByRole("button", { name: "Predict Traffic", exact: true })
    .click();
  const response = await pending;
  assert.equal(response.status(), 200);
  const result = await response.json();
  await page.locator(".probability-value").waitFor();
  assert.ok(
    (await page.locator(".probability-value").textContent()).includes(
      `${(result.high_probability * 100).toFixed(1)}%`,
    ),
  );
  await page.getByText(result.category, { exact: true }).first().waitFor();
  if (output)
    await page.screenshot({
      path: `${output}/prediction-desktop.png`,
      fullPage: true,
    });
}
// A failed backend must yield an error, never a JavaScript probability fallback.
await page.route("**/api/predict", (route) => route.abort());
await page
  .getByRole("button", { name: "Predict Traffic", exact: true })
  .click();
await page.getByRole("alert").waitFor();
await page.setViewportSize({ width: 390, height: 844 });
await page.goto(base);
await page.getByRole("heading", { name: "Traffic, in perspective." }).waitFor();
assert.equal(
  await page.locator("body").evaluate((el) => el.scrollWidth > innerWidth),
  false,
);
await page.getByRole("button", { name: "Open navigation" }).click();
await page
  .getByRole("navigation", { name: "Main navigation" })
  .getByRole("link", { name: "Predict traffic", exact: true })
  .click();
await page
  .getByRole("heading", { name: "What does this hour look like?" })
  .waitFor();
assert.equal(
  await page.locator("body").evaluate((el) => el.scrollWidth > innerWidth),
  false,
);
if (output)
  await page.screenshot({
    path: `${output}/prediction-mobile.png`,
    fullPage: true,
  });
assert.deepEqual(errors, []);
await browser.close();
console.log(
  "Passed: all pages, real API prediction, input validation, API failure, mobile navigation and overflow.",
);
