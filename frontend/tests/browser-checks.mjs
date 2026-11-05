import { chromium } from "playwright";
import assert from "node:assert/strict";
import fs from "node:fs";
const artifact = (name) =>
  JSON.parse(
    fs.readFileSync(new URL(`../public/data/${name}.json`, import.meta.url)),
  );
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
            task: "close-following",
            predictions: [{ id: "1", eligible: true, probability: 0.99 }],
          },
        }),
      );
    fs.mkdirSync("test-results", { recursive: true });
    const replay = artifact("replay");
    for (const [route, title] of [
      ["", "Predict the next lane change."],
      ["analysis", "How the recorded traffic behaves"],
      ["performance", "Compare actual behavior predictions"],
      ["about", "A question the recording can answer."],
    ]) {
      await page.goto(`${base}#/${route}`);
      await page.getByRole("heading", { name: title, exact: true }).waitFor();
      if (!route) await page.getByText("1,180,598", { exact: true }).waitFor();
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
    await page.goto(`${base}#/replay`);
    await page.getByLabel("Held-out example", { exact: true }).waitFor();
    for (let index = 0; index < replay.examples.length; index++) {
      const e = replay.examples[index];
      await page
        .getByLabel("Held-out example", { exact: true })
        .selectOption(String(index));
      assert.equal(
        await page.getByTestId("replay-outcome").textContent(),
        "Play to reveal",
      );
      const forecast = await page
        .getByTestId("replay-prediction")
        .textContent();
      assert.equal(forecast, e.predicted ? "Lane change" : "Stay in lane");
      await page.getByLabel("Recorded replay time", { exact: true }).fill("20");
      assert.equal(
        await page.getByTestId("replay-outcome").textContent(),
        e.actual ? "Lane change" : "Stay in lane",
      );
      assert.equal(
        await page.getByTestId("replay-prediction").textContent(),
        forecast,
      );
      assert.match(
        await page.locator(".result-card").nth(1).textContent(),
        e.actual === e.predicted ? /Correct/ : /Incorrect/,
      );
    }
    await page
      .getByLabel("Held-out example", { exact: true })
      .selectOption("0");
    await page.getByLabel("Replay speed", { exact: true }).selectOption("2");
    await page
      .getByRole("button", { name: "Play recorded future", exact: true })
      .click();
    await page.waitForFunction(
      () =>
        document.querySelector('[data-testid="replay-time"]')?.textContent ===
        "2.0 / 2.0 s",
    );
    await page.screenshot({
      path: "test-results/behavior-recorded-replay.png",
      fullPage: true,
    });
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
      .getByText("Adapted NGSIM training snapshot", { exact: true })
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
    await page.goto(`${base}#/replay`);
    await page.getByLabel("Held-out example", { exact: true }).waitFor();
    assert.equal(
      await page.locator("body").evaluate((e) => e.scrollWidth > innerWidth),
      false,
    );
    await page.screenshot({
      path: "test-results/behavior-replay-mobile.png",
      fullPage: true,
    });
    assert.deepEqual(errors, []);
    assert.deepEqual(missing, []);
    if (liveApi)
      assert.ok(batches > 0, "Actual behavior API must receive requests");
    console.log(
      `PASS: 6 routes, repeated loads, ${replay.examples.length} frozen forecasts versus observed outcomes, recorded playback, empirical initialization, 4 lanes/12 cars, independent bypass, blocked lanes, 3 contacts/replays, 30 cars, mobile and task-schema protection. Actual/legacy batch responses: ${batches}.`,
    );
  } finally {
    await browser.close();
  }
}
