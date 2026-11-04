import { validateResearch } from "./research-schema";

export const API_URL = (
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");
export async function request(path, options = {}) {
  const response = await fetch(`${API_URL}${path}`, {
    ...options,
    signal: options.signal || AbortSignal.timeout(12000),
  });
  if (!response.ok) throw new Error(`API ${response.status}`);
  return response.json();
}
export async function loadResearch(name) {
  try {
    return {
      data: validateResearch(
        name,
        await request(
          {
            info: "/api/model/info",
            metrics: "/api/model/metrics",
            analytics: "/api/analytics",
            "lane-info": "/api/lane-change/info",
            "lane-metrics": "/api/lane-change/metrics",
          }[name],
        ),
      ),
      source: "Live API",
    };
  } catch {
    const r = await fetch(
      `${import.meta.env.BASE_URL}data/${name}.json?v=nearmiss-1`,
    );
    if (!r.ok)
      throw new Error("Research export unavailable; execute the notebook.");
    return {
      data: validateResearch(name, await r.json()),
      source: "Saved notebook export",
    };
  }
}
