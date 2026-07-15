export function money(value?: number | null) {
  if (value == null || !Number.isFinite(Number(value))) return "—";

  return new Intl.NumberFormat("en", {
    style: "currency",
    currency: "EUR",
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(Number(value));
}

export function parseDate(value: string) {
  return new Date(value.includes("T") ? value : `${value}T12:00:00Z`);
}

export function shortDate(value?: string | null) {
  if (!value) return "—";

  return new Intl.DateTimeFormat("en", {
    day: "numeric",
    month: "short",
  }).format(parseDate(value));
}

export function fullDate(value?: string | null) {
  if (!value) return "Date unavailable";

  return new Intl.DateTimeFormat("en", {
    day: "numeric",
    month: "short",
    year: "numeric",
  }).format(parseDate(value));
}

export function initials(name?: string) {
  return (name ?? "Player")
    .split(" ")
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0])
    .join("")
    .toUpperCase();
}

export function signedPercent(value?: number | null) {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)}%`;
}

export function directionCopy(direction?: string | null) {
  if (direction === "rising") return "Likely rising";
  if (direction === "falling") return "Likely falling";
  if (direction === "stable") return "Broadly stable";
  return "Not yet projected";
}

export function trendClass(value?: number | null) {
  if (value == null) return "neutral";
  if (value > 0.1) return "positive";
  if (value < -0.1) return "negative";
  return "neutral";
}

export function ratingClass(value?: number | null) {
  if (value == null) return "neutral";
  if (value > 6.25) return "positive";
  if (value < 5.75) return "negative";
  return "neutral";
}
