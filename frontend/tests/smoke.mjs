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
const page = await browser.newPage({ viewport: { width: 1600, height: 1100 } });
const base = process.env.TEST_URL || "http://127.0.0.1:5173/";
const output = process.env.QA_SCREENSHOT_DIR || "test-results";
fs.mkdirSync(output, { recursive: true });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
let predictions = 0;
page.on("response", (r) => {
  if (r.url().endsWith("/api/predict-batch") && r.status() === 200)
    predictions++;
});
const noOverflow = async () =>
  assert.equal(
    await page.locator("body").evaluate((el) => el.scrollWidth > innerWidth),
    false,
  );
try {
  await page.goto(base);
  await page
    .getByRole("heading", { name: "Traffic, understood through motion." })
    .waitFor();
  await page.getByText("1,180,598", { exact: true }).waitFor();
  await noOverflow();
  await page.screenshot({
    path: `${output}/dashboard-desktop.png`,
    fullPage: true,
  });
  for (const [route, title] of [
    ["analysis", "Inside the trajectories"],
    ["performance", "Let the validation decide."],
    ["about", "A clear question. An honest experiment."],
  ]) {
    await page.goto(`${base}#/${route}`);
    await page.getByRole("heading", { name: title, exact: true }).waitFor();
    if (route === "performance")
      await page.getByText("87.1%", { exact: true }).first().waitFor();
    await noOverflow();
  }
  await page.goto(`${base}#/simulator`);
  await page.getByLabel("Number of Lanes", { exact: true }).selectOption("4");
  await page.getByLabel("Number of vehicles", { exact: true }).fill("12");
  await page
    .getByRole("button", { name: "Generate Random Traffic", exact: true })
    .click();
  assert.equal(
    await page.getByRole("button", { name: /^Vehicle \d+$/ }).count(),
    12,
  );
  assert.equal(
    await page.locator("svg.highway text").filter({ hasText: /^L\d/ }).count(),
    4,
  );
  const car = page.getByRole("button", { name: "Vehicle 1", exact: true });
  const xBefore = await car.locator("rect").first().getAttribute("x");
  await page
    .getByRole("button", { name: "Start Simulation", exact: true })
    .click();
  await page.waitForTimeout(1100);
  await page
    .getByRole("button", { name: "Pause Simulation", exact: true })
    .click();
  assert.notEqual(await car.locator("rect").first().getAttribute("x"), xBefore);
  assert.ok(
    Number(
      (await page.getByTestId("sim-time").textContent()).replace(" s", ""),
    ) > 0,
  );
  await car.click();
  await page.getByLabel("Vehicle speed", { exact: true }).fill("90");
  await page.getByLabel("Vehicle acceleration", { exact: true }).fill("2");
  await page.getByRole("button", { name: "Apply brakes", exact: true }).click();
  await page.getByRole("button", { name: "Right lane", exact: true }).click();
  await page
    .getByRole("button", { name: "Resume Simulation", exact: true })
    .click();
  await page.waitForTimeout(1800);
  await page
    .getByRole("button", { name: "Pause Simulation", exact: true })
    .click();
  assert.match(await page.locator(".pair-info").textContent(), /Lane|Position/);
  await page.getByRole("button", { name: "Reset", exact: true }).click();
  assert.equal(await page.getByTestId("sim-time").textContent(), "0.0 s");
  assert.equal(await car.locator("rect").first().getAttribute("x"), xBefore);
  await page.waitForTimeout(1300);
  assert.ok(predictions > 0, "Real batch requests must succeed");
  await page.screenshot({
    path: `${output}/simulator-desktop.png`,
    fullPage: true,
  });
  for (const scenario of [
    "Rear-End Collision Demo",
    "Unsafe Lane Change",
    "Multiple-Vehicle Incident",
  ]) {
    await page.getByLabel("Scenario", { exact: true }).selectOption(scenario);
    await page
      .getByRole("button", { name: "Load scenario", exact: true })
      .click();
    await page
      .getByLabel("Simulation speed", { exact: true })
      .selectOption("4");
    await page
      .getByRole("button", { name: "Start Simulation", exact: true })
      .click();
    await page.locator(".collision-alert").waitFor();
    await page
      .getByRole("button", { name: "Pause Simulation", exact: true })
      .click();
    assert.match(
      await page.locator(".collision-alert").textContent(),
      /Hypothetical Simulated Collision/,
    );
    if (scenario === "Rear-End Collision Demo") {
      await page.screenshot({
        path: `${output}/collision-desktop.png`,
        fullPage: true,
      });
      await page
        .getByRole("button", { name: "Rewind 2 s", exact: true })
        .click();
      assert.equal(await page.locator(".collision-alert").count(), 0);
      await page.getByRole("button", { name: "Replay", exact: true }).click();
      await page.locator(".collision-alert").waitFor();
      await page
        .getByRole("button", { name: "Pause Simulation", exact: true })
        .click();
    }
  }
  await page.getByLabel("Number of Lanes", { exact: true }).selectOption("6");
  await page.getByLabel("Number of vehicles", { exact: true }).fill("30");
  await page
    .getByRole("button", { name: "Generate Random Traffic", exact: true })
    .click();
  assert.equal(
    await page.getByRole("button", { name: /^Vehicle \d+$/ }).count(),
    30,
  );
  await page.getByRole("button", { name: "Zoom in", exact: true }).click();
  await page.getByRole("button", { name: "Fit all", exact: true }).click();
  await page
    .getByRole("button", { name: "Start Simulation", exact: true })
    .click();
  await page.waitForTimeout(500);
  await page
    .getByRole("button", { name: "Pause Simulation", exact: true })
    .click();
  await page.setViewportSize({ width: 390, height: 844 });
  await noOverflow();
  await page.screenshot({
    path: `${output}/simulator-mobile.png`,
    fullPage: true,
  });
  // Failure cannot invent model scores or stop motion.
  await page.route("**/api/**", (route) => route.abort());
  await page.reload();
  await page
    .getByRole("button", { name: "Start Simulation", exact: true })
    .click();
  await page
    .getByText("Predictions unavailable · retrying", { exact: true })
    .waitFor();
  const t = await page.getByTestId("sim-time").textContent();
  await page.waitForTimeout(600);
  assert.notEqual(await page.getByTestId("sim-time").textContent(), t);
  assert.equal(await page.locator(".risk-box strong").textContent(), "—");
  await page
    .getByRole("button", { name: "Pause Simulation", exact: true })
    .click();
  for (const route of ["", "analysis", "performance", "about"]) {
    await page.goto(`${base}#/${route}`);
    await page.waitForTimeout(400);
    await noOverflow();
  }
  // A delayed response from a discarded scene must be ignored.
  await page.unroute("**/api/**");
  let release;
  const gate = new Promise((resolve) => (release = resolve));
  await page.route("**/api/predict-batch", async (route) => {
    await gate;
    try {
      await route.fulfill({
        json: {
          predictions: [
            { id: "1", eligible: true, probability: 0.999, prediction: 1 },
          ],
        },
      });
    } catch {}
  });
  await page.goto(`${base}#/simulator`);
  await page.waitForTimeout(100);
  await page
    .getByRole("button", { name: "Generate Random Traffic", exact: true })
    .click();
  release();
  await page.waitForTimeout(300);
  assert.equal(await page.locator(".risk-box strong").textContent(), "—");
  assert.deepEqual(errors, []);
  console.log(
    "PASS: five routes, 4 lanes/12 cars, motion, controls, reset, real inference, 3 collision scenarios, replay, 30 cars, mobile, API outage, stale response protection.",
  );
  console.log("Successful actual inference batches:", predictions);
} finally {
  await browser.close();
}
