// Screenshots and the demo GIF for the README, from the app's real pages (app/ui) under
// a fake __TAURI__ (fake-tauri.js) playing the made-up meeting in scenario.js. No model,
// no recording and no port: files are served through Playwright's request interception.
//
//   node app/demo-capture/capture.mjs            # writes assets/screenshots/*.png and
//                                                # assets/meetbutler-demo.gif (needs ffmpeg)
//
// Look at every image it writes before committing it (CLAUDE.md, "Grep does not read images").
import { createRequire } from "node:module";
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

// require, not import: it also finds a Playwright installed globally (NODE_PATH)
const { chromium } = createRequire(import.meta.url)("playwright");

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "../..");
const ORIGIN = "http://meetbutler.demo";
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".png": "image/png" };
const shots = join(root, "assets/screenshots");
const gifPath = join(root, "assets/meetbutler-demo.gif");
const frames = mkdtempSync(join(tmpdir(), "meetbutler-frames-"));
mkdirSync(shots, { recursive: true });

const browser = await chromium.launch();
const context = await browser.newContext({
  viewport: { width: 1280, height: 800 },
  deviceScaleFactor: 2,
  locale: "ko-KR",
  timezoneId: "Asia/Seoul",
  colorScheme: "light",
});
await context.route(`${ORIGIN}/**`, (route) => {
  const path = new URL(route.request().url()).pathname;
  const file = path.startsWith("/ui/") ? join(root, "app", path) : join(here, path.slice(1));
  try {
    route.fulfill({ body: readFileSync(file), contentType: TYPES[extname(file)] || "application/octet-stream" });
  } catch {
    route.fulfill({ status: 404, body: "" });
  }
});
await context.addInitScript({ path: join(here, "scenario.js") });
await context.addInitScript({ path: join(here, "fake-tauri.js") });

const page = await context.newPage();
page.on("pageerror", (e) => console.error("page error:", e.message));
await page.clock.install({ time: new Date("2026-11-04T14:00:00+09:00") });
await page.goto(`${ORIGIN}/stage.html`);
await page.waitForLoadState("networkidle");

const frame = (id) => page.frameLocator(`#${id}`);
const demo = (fn, arg) => page.evaluate(([f, a]) => window.__demo[f](a), [fn, arg]);
const settle = () => page.clock.runFor(300);

// the GIF: one frame per step, each held for its own time
const steps = [];
async function frameShot(hold) {
  await settle();
  const file = join(frames, `f${String(steps.length).padStart(3, "0")}.png`);
  await page.screenshot({ path: file, scale: "css", animations: "disabled" });
  steps.push({ file, hold });
}

// 1. before the meeting: the library holds past meetings
await frame("library").locator("#sessions li").first().click();
await frameShot(1.6);

// 2. Start from the mini bar; captions arrive as they are recognised
await frame("bar").locator("#toggle").click();
await settle();
const lines = await page.evaluate(() => window.SCENARIO.lines.map((l) => l[0]));
let now = 0;
for (let i = 0; i < lines.length; i++) {
  await page.clock.runFor((lines[i] + 2 - now) * 1000);
  now = lines[i] + 2;
  await demo("caption", i);
  await frameShot(i === lines.length - 1 ? 1.4 : 0.7);
  if (i === 5) {
    await page.locator("#bar-window").screenshot({ path: join(shots, "minibar.png"), animations: "disabled" });
    await page.locator("#panel-window").screenshot({ path: join(shots, "panel-captions.png"), animations: "disabled" });
  }
}
await page.clock.runFor((79 - now) * 1000);

// 3. Stop: the final pass replaces the captions, then the local summary
await frame("bar").locator("#toggle").click();
await settle();
for (const p of [0.2, 0.55, 0.9]) {
  await demo("finalProgress", p);
  await frameShot(0.6);
}
await demo("finalProgress", 1);
await frameShot(1.4);
await frame("panel").locator('[data-tab="summary"]').click();
for (const p of [0.3, 0.7]) {
  await demo("summaryProgress", p);
  await frameShot(0.6);
}
await demo("summaryProgress", 1);
await page.evaluate((t) => window.__demo.invoke("rename_session", { sessionId: window.__demo.latest(), title: t }),
  await page.evaluate(() => window.SCENARIO.title));
await settle();
await frameShot(1.6);
await page.locator("#panel-window").screenshot({ path: join(shots, "panel-summary.png"), animations: "disabled" });

// 4. the library: summary beside the final transcript; an item jumps to its line
await frame("library").locator("#sessions li").first().click();
await settle();
await frameShot(1.2);
await frame("library").locator("#summary-body .go").nth(1).click();
await page.clock.runFor(600);
await frameShot(2.4);
await page.locator("#library-window").screenshot({ path: join(shots, "library.png"), animations: "disabled" });

await browser.close();

// frames → GIF, 1000 px wide, one palette for the whole clip
const list = join(frames, "list.txt");
writeFileSync(list, steps.map((s) => `file '${s.file}'\nduration ${s.hold}`).join("\n") + `\nfile '${steps.at(-1).file}'\n`);
execFileSync("ffmpeg", [
  "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", list,
  "-vf", "scale=1000:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=full[p];[b][p]paletteuse=dither=none",
  "-loop", "0", gifPath,
]);
rmSync(frames, { recursive: true, force: true });
console.log(`wrote ${steps.length} frames to ${gifPath} and 4 screenshots to ${shots}`);
