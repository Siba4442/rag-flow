// Style tile behaviour: theme switch, baseline overlay, live swatches and contrast.

import "./styles/design.css";

type ThemeChoice = "light" | "dark" | "system";
const THEME_KEY = "ragflow.design.theme";

const SWATCHES: { token: string; name: string; role: string; against?: string }[] = [
  { token: "--bg", name: "Background", role: "Page" },
  { token: "--surface-2", name: "Surface 2", role: "Hover, current item" },
  { token: "--text", name: "Text", role: "Primary text" },
  { token: "--text-2", name: "Text 2", role: "Body, secondary" },
  { token: "--text-3", name: "Text 3", role: "Meta, captions" },
  { token: "--line-strong", name: "Line", role: "Borders, controls" },
  { token: "--accent", name: "Accent", role: "Accent marks and text" },
  { token: "--highlight", name: "Highlight", role: "Evidence, selection", against: "--on-highlight" },
  { token: "--danger", name: "Danger", role: "Errors only" },
];

function readTheme(): ThemeChoice {
  try {
    const v = localStorage.getItem(THEME_KEY);
    return v === "light" || v === "dark" ? v : "system";
  } catch {
    return "system";
  }
}

function applyTheme(choice: ThemeChoice): void {
  const root = document.documentElement;
  if (choice === "system") root.removeAttribute("data-theme");
  else root.dataset.theme = choice;
  for (const b of document.querySelectorAll<HTMLButtonElement>("[data-theme-choice]")) {
    b.setAttribute("aria-checked", String(b.dataset.themeChoice === choice));
  }
  try {
    localStorage.setItem(THEME_KEY, choice);
  } catch {
    /* preview still works without storage */
  }
  renderSwatches();
}

// ---------------------------------------------------------------- colour maths
function toRgb(color: string): [number, number, number] {
  const probe = document.createElement("span");
  probe.style.color = color;
  document.body.append(probe);
  const rgb = getComputedStyle(probe).color.match(/[\d.]+/g)?.map(Number) ?? [0, 0, 0];
  probe.remove();
  return [rgb[0] ?? 0, rgb[1] ?? 0, rgb[2] ?? 0];
}

function luminance([r, g, b]: [number, number, number]): number {
  const lin = (c: number): number => {
    const s = c / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
}

function contrast(a: string, b: string): number {
  const la = luminance(toRgb(a));
  const lb = luminance(toRgb(b));
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
}

function hex([r, g, b]: [number, number, number]): string {
  return `#${[r, g, b].map((v) => Math.round(v).toString(16).padStart(2, "0")).join("")}`.toUpperCase();
}

function grade(ratio: number): string {
  if (ratio >= 7) return "AAA";
  if (ratio >= 4.5) return "AA";
  if (ratio >= 3) return "AA large";
  return "decorative";
}

function renderSwatches(): void {
  const host = document.getElementById("swatches");
  if (!host) return;
  const css = getComputedStyle(document.documentElement);
  const bg = css.getPropertyValue("--bg").trim();
  host.replaceChildren();
  for (const s of SWATCHES) {
    const value = css.getPropertyValue(s.token).trim();
    const fg = s.against ? css.getPropertyValue(s.against).trim() : value;
    const base = s.against ? value : bg;
    const ratio = contrast(fg, base);
    const isSurface = s.token === "--bg" || s.token === "--surface-2";
    const card = document.createElement("div");
    card.className = "swatch";
    const chip = document.createElement("div");
    chip.className = "chip-color";
    chip.style.background = `var(${s.token})`;
    const name = document.createElement("p");
    name.className = "swatch-name";
    name.textContent = s.name;
    const role = document.createElement("p");
    role.className = "swatch-role";
    role.textContent = s.role;
    const token = document.createElement("p");
    token.className = "swatch-token";
    token.textContent = s.token;
    const code = document.createElement("p");
    code.className = "swatch-hex";
    code.textContent = hex(toRgb(value));
    const r = document.createElement("p");
    r.className = "ratio";
    if (isSurface) {
      r.textContent = "Surface";
    } else {
      const strong = document.createElement("b");
      strong.textContent = `${ratio.toFixed(2)}:1`;
      r.append(strong, ` ${grade(ratio)}${s.against ? " with text on it" : " on background"}`);
    }
    card.append(chip, name, role, token, code, r);
    host.append(card);
  }
}

function renderSpace(): void {
  const host = document.getElementById("space-scale");
  if (!host) return;
  for (const n of [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]) {
    const row = document.createElement("div");
    row.className = "space-item";
    const label = document.createElement("span");
    label.textContent = `${n * 4}px`;
    const bar = document.createElement("div");
    bar.className = "space-bar";
    bar.style.width = `var(--s-${n})`;
    row.append(label, bar);
    host.append(row);
  }
}

// ---------------------------------------------------------------- wire up
for (const b of document.querySelectorAll<HTMLButtonElement>("[data-theme-choice]")) {
  b.addEventListener("click", () => applyTheme((b.dataset.themeChoice as ThemeChoice) ?? "system"));
}
const baseline = document.getElementById("baseline-toggle") as HTMLInputElement | null;
baseline?.addEventListener("change", () => document.body.classList.toggle("show-baseline", baseline.checked));
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderSwatches);

renderSpace();
applyTheme(readTheme());
