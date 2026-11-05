import { chromium } from "playwright";
import assert from "node:assert/strict";
import fs from "node:fs";
export async function browserChecks(base, { liveApi = false } = {}) {
  const browser = await chromium.launch(
    process.platform === "win32"
      ? {
          executablePath:
            "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
          headless: true,
        }
      : { headless: true },
  );
  try {
    const page = await browser.newPage({
        viewport: { width: 1500, height: 1100 },
      }),
      errors = [],
      missing = [];
    let batches = 0;
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("response", (r) => {
      if (r.url().includes("/assets/") && r.status() >= 400)
        missing.push(r.url());
      if (r.url().endsWith("/api/predict-batch") && r.status() === 200)
        batches++;
    });
    if (!liveApi)
      await page.route("**/api/**", (route) =>
        route.fulfill({
          json: {
            task: "lane-change-behavior",
            predictions: [{ id: "1", eligible: true, task: "lane-change-behavior", probability: 0.99, prediction: 1, label: "Lane change" }],
          },
        }),
      );
    fs.mkdirSync("test-results", { recursive: true });
    for (const [route, title] of [
      ["", "Predict the next lane change."],
      ["analysis", "How the recorded traffic behaves"],
      ["performance", "Compare actual behavior predictions"],
      ["about", "A question the recording can answer."],
      ["simulator", "Highway simulator"],
    ]) {
      await page.goto(`${base}#/${route}`);
      await page.getByRole("heading", { name: title, exact: true }).waitFor();
      if (route === "performance")
        await page.getByText("Always-stay baseline", { exact: true }).waitFor();
      await page.reload();
      await page.getByRole("heading", { name: title, exact: true }).waitFor();
      assert.ok((await page.locator("main").innerText()).length > 100);
      assert.equal(
        await page.locator("body").evaluate((e) => e.scrollWidth > innerWidth),
        false,
      );
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
    await page
      .getByText("New scene sampled from Automatum distributions", { exact: true })
      .waitFor();
    const before = await page
      .getByRole("button", { name: /^Vehicle \d+$/ })
      .evaluateAll((es) =>
        es.map((e) => e.querySelector("rect").getAttribute("x")).join(","),
      );
    await page
      .getByRole("button", { name: "Start Simulation", exact: true })
      .click();
    await page.waitForTimeout(1200);
    await page
      .getByRole("button", { name: "Pause Simulation", exact: true })
      .click();
    const after = await page
      .getByRole("button", { name: /^Vehicle \d+$/ })
      .evaluateAll((es) =>
        es.map((e) => e.querySelector("rect").getAttribute("x")).join(","),
      );
    assert.notEqual(after, before);
    await page.getByRole("button", { name: "Reset", exact: true }).click();
    assert.equal(await page.getByTestId("sim-time").textContent(), "0.0 s");
    await page
      .getByRole("button", { name: "Six-lane bypass demo", exact: true })
      .click();
    await page
      .getByLabel("Simulation speed", { exact: true })
      .selectOption("4");
    await page
      .getByRole("button", { name: "Start Simulation", exact: true })
      .click();
    await page
      .getByText(
        /Vehicle 2 completed change to Lane \d+ with no simulated contact/,
      )
      .waitFor();
    await page.waitForTimeout(500);
    await page
      .getByRole("button", { name: "Pause Simulation", exact: true })
      .click();
    assert.equal(await page.locator(".collision-alert").count(), 0);
    assert.match(
      await page
        .getByLabel("Selected vehicle", { exact: true })
        .locator("option:checked")
        .textContent(),
      /Lane 4/,
    );
    assert.match(
      await page.getByTestId("lane-assessment").textContent(),
      /The behavior prediction does not choose a lane/,
    );
    assert.doesNotMatch(
      await page.getByTestId("lane-assessment").textContent(),
      /Learned close-conflict/,
    );
    await page.screenshot({
      path: "test-results/behavior-independent-simulator.png",
      fullPage: true,
    });
    await page
      .getByLabel("Scenario", { exact: true })
      .selectOption("Blocked Lane - No Safe Gap");
    await page
      .getByRole("button", { name: "Load scenario", exact: true })
      .click();
    await page
      .getByRole("button", { name: "Start Simulation", exact: true })
      .click();
    await page.waitForTimeout(2200);
    await page
      .getByRole("button", { name: "Pause Simulation", exact: true })
      .click();
    assert.equal(await page.locator(".collision-alert").count(), 0);
    assert.match(
      await page
        .getByLabel("Selected vehicle", { exact: true })
        .locator("option:checked")
        .textContent(),
      /Lane 2/,
    );
    for (const scene of [
      "Rear-End Collision Demo",
      "Unsafe Lane Change",
      "Multiple-Vehicle Incident",
    ]) {
      await page.getByLabel("Scenario", { exact: true }).selectOption(scene);
      await page
        .getByRole("button", { name: "Load scenario", exact: true })
        .click();
      await page
        .getByRole("button", { name: "Start Simulation", exact: true })
        .click();
      await page.locator(".collision-alert").waitFor();
      await page
        .getByRole("button", { name: "Pause Simulation", exact: true })
        .click();
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
    await page.getByLabel("Number of Lanes", { exact: true }).selectOption("6");
    await page.getByLabel("Number of vehicles", { exact: true }).fill("30");
    await page
      .getByRole("button", { name: "Generate Random Traffic", exact: true })
      .click();
    assert.equal(
      await page.getByRole("button", { name: /^Vehicle \d+$/ }).count(),
      30,
    );
    await page.setViewportSize({ width: 390, height: 844 });
    assert.equal(
      await page.locator("body").evaluate((e) => e.scrollWidth > innerWidth),
      false,
    );
    await page.screenshot({
      path: "test-results/behavior-simulator-mobile.png",
      fullPage: true,
    });
    await page.goto(`${base}#/simulator`);
    await page.getByRole("heading", { name: "Highway simulator", exact: true }).waitFor();
    assert.equal(
      await page.locator("body").evaluate((e) => e.scrollWidth > innerWidth),
      false,
    );
    await page.screenshot({
      path: "test-results/behavior-simulator-mobile-final.png",
      fullPage: true,
    });
    assert.deepEqual(errors, []);
    assert.deepEqual(missing, []);
    if (liveApi)
      assert.ok(batches > 0, "Actual behavior API must receive requests");
    console.log(
      `PASS: 5 routes, repeated loads, aggregate dataset initialization, 4 lanes/12 cars, independent bypass, blocked lanes, 3 contacts/replays, 30 cars, mobile and task-schema protection. Behavior API batch responses: ${batches}.`,
    );
  } finally {
    await browser.close();
  }
}
