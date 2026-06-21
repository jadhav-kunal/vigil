// Stable, muted color per model family (dark-adjusted pastel chips, brighter text).
// One accent stays reserved for the app; these are quiet category hues, not neon.

interface Chip {
  bg: string;
  fg: string;
  dot: string;
}

const FAMILIES: { match: RegExp; chip: Chip }[] = [
  { match: /opus|gpt-4o(?!-mini)|gpt-4\.1(?!-mini|-nano)|o3(?!-mini)/i, chip: chip("#7c8cff") },
  { match: /sonnet|gpt-4/i, chip: chip("#4bb1c9") },
  { match: /haiku|mini|nano|3\.5-turbo|small/i, chip: chip("#5fae7a") },
  { match: /o3-mini|reason/i, chip: chip("#c79a55") },
];

function chip(hue: string): Chip {
  return { bg: hexA(hue, 0.12), fg: hue, dot: hue };
}

function hexA(hex: string, a: number): string {
  const n = parseInt(hex.slice(1), 16);
  const r = (n >> 16) & 255;
  const g = (n >> 8) & 255;
  const b = n & 255;
  return `rgba(${r}, ${g}, ${b}, ${a})`;
}

const NEUTRAL: Chip = { bg: "rgba(255,255,255,0.06)", fg: "#9b9ba3", dot: "#7a7a82" };

export function modelChip(model: string): Chip {
  for (const f of FAMILIES) if (f.match.test(model)) return f.chip;
  return NEUTRAL;
}

export function shortModel(model: string): string {
  return model.replace(/^(openai|anthropic)\//, "");
}
