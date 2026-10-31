const local = ["localhost", "127.0.0.1"].includes(window.location.hostname);
export const API_URL = (
  import.meta.env.VITE_API_URL || (local ? "http://127.0.0.1:8000" : "")
).replace(/\/$/, "");
async function request(url, options) {
  const response = await fetch(url, {
    ...options,
    signal: AbortSignal.timeout(15000),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    const detail = Array.isArray(data.detail)
      ? data.detail.map((item) => item.msg).join(" ")
      : data.detail;
    throw new Error(detail || `Request failed (${response.status}).`);
  }
  try {
    return await response.json();
  } catch {
    throw new Error(
      "The service returned an invalid response. If it is starting up, wait a moment and retry.",
    );
  }
}
export async function loadResearch() {
  const names = ["overview", "analytics", "metrics"];
  if (API_URL) {
    try {
      const values = await Promise.all(
        ["/api/overview", "/api/analytics", "/api/model/metrics"].map((path) =>
          request(API_URL + path),
        ),
      );
      return Object.fromEntries([
        ...names.map((name, i) => [name, values[i]]),
        ["source", "api"],
      ]);
    } catch {
      /* Saved data is an actual notebook export, never a prediction fallback. */
    }
  }
  const values = await Promise.all(
    names.map((name) =>
      request(`${import.meta.env.BASE_URL}data/${name}.json`),
    ),
  );
  return Object.fromEntries([
    ...names.map((name, i) => [name, values[i]]),
    ["source", "saved"],
  ]);
}
export async function predictTraffic(payload) {
  if (!API_URL)
    throw new Error(
      "Prediction service is not deployed. Run the FastAPI backend locally or configure VITE_API_URL with its HTTPS URL.",
    );
  return request(API_URL + "/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}
