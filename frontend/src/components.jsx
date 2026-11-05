import { useEffect, useState } from "react";
import { loadResearch } from "./services/api.js";
export const pct = (x) => `${(x * 100).toFixed(1)}%`;
export const number = (x) => Number(x).toLocaleString("en-US");
export function Button({ children, icon: Icon, ...props }) {
  return (
    <button {...props}>
      {Icon && <Icon size={15} />} {children}
    </button>
  );
}
export function Field({ label, children }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
    </label>
  );
}
export function Stat({ label, value, note }) {
  return (
    <div className="stat">
      <span>{label}</span>
      <strong>{value}</strong>
      {note && <small>{note}</small>}
    </div>
  );
}
export function useResearch(name) {
  const [attempt, setAttempt] = useState(0);
  const [state, setState] = useState({
    data: null,
    source: "Loading recorded research…",
  });
  useEffect(() => {
    let active = true;
    setState({ data: null, source: "Loading recorded research...", error: false });
    loadResearch(name)
      .then((value) => active && setState(value))
      .catch(
        () =>
          active &&
          setState({
            data: null,
            error: true,
            source: "Research data could not load. Please try again.",
          }),
      );
    return () => {
      active = false;
    };
  }, [name, attempt]);
  return { ...state, retry: () => setAttempt((value) => value + 1) };
}
