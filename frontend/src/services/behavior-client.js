import { inferBehavior } from "./behavior-inference.js";
import { validatePredictions, validateResearch } from "./research-schema.js";

// Render can be waking up or deploying a different model from the Pages bundle.
// Keep the saved classifier available immediately, and retry the API later.
export function createBehaviorClient(info, { request, now = Date.now, retryMs = 30000 }) {
  let connected = false;
  let retryAt = 0;
  const local = (rows) => ({
    source: "Saved fitted behavior model - local inference",
    predictions: rows.map(({ id, inputs }) => ({ id, ...inferBehavior(info, inputs) })),
  });
  return {
    async predict(rows, signal) {
      if (now() < retryAt) return local(rows);
      try {
        if (!connected) {
          const remote = validateResearch("info", await request("/api/model/info", { signal }));
          if (remote.model_id !== info.model_id)
            throw new Error("API and website model revisions differ.");
          connected = true;
        }
        const data = validatePredictions(
          await request("/api/predict-batch", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            signal,
            body: JSON.stringify({ observations: rows.map(({ id, inputs }) => ({ id, inputs })) }),
          }),
          info,
          rows.map(({ id }) => id),
        );
        return { ...data, source: "Model connected - behavior inference" };
      } catch (error) {
        connected = false;
        retryAt = now() + retryMs;
        return local(rows);
      }
    },
  };
}
