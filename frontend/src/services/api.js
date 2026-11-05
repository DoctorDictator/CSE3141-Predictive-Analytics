import { validateResearch } from "./research-schema.js";
export const API_URL = (
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");
export async function request(path, options = {}) {
  const r = await fetch(`${API_URL}${path}`, {
    ...options,
    signal: options.signal || AbortSignal.timeout(10000),
  });
  if (!r.ok) throw new Error(`API ${r.status}`);
  return r.json();
}
async function saved(name) {
  const r = await fetch(
    `${import.meta.env.BASE_URL}data/${name}.json?v=behavior-1`,
  );
  if (!r.ok) throw new Error("Recorded research export is unavailable.");
  return {
    data: validateResearch(name, await r.json()),
    source: "Saved notebook export",
  };
}
export async function loadResearch(name) {
  // Matching, same-build snapshots start immediately, including during a Render cold start.
  try {
    return await saved(name);
  } catch {
    return {
      data: validateResearch(
        name,
        await request(
          {
            info: "/api/model/info",
            metrics: "/api/model/metrics",
            analytics: "/api/analytics",
            replay: "/api/replay",
          }[name],
        ),
      ),
      source: "Live API",
    };
  }
}
