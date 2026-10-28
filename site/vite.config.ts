import { execFileSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, "..");
const watched = ["studies", "docs", "ideas", "ROADMAP.md", "PROMPTS.md", "CONTRIBUTING.md", "CLAUDE.md"].map((p) =>
  path.join(repo, p),
);

/** In `npm run dev`, re-export the studies and reload the page whenever a study or doc changes. */
function studiesContent(): Plugin {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const exportContent = () => {
    try {
      execFileSync("python3", [path.join(repo, "tools/studies.py"), "export", path.join(here, "public/content"), "--force"],
        { stdio: "inherit" });
    } catch {
      // the tool prints its own errors
    }
  };
  return {
    name: "studies-content",
    apply: "serve",
    configureServer(server) {
      server.watcher.add(watched);
      const onChange = (file: string) => {
        if (!watched.some((w) => file === w || file.startsWith(w + path.sep))) return;
        clearTimeout(timer);
        timer = setTimeout(() => {
          exportContent();
          server.ws.send({ type: "full-reload" });
        }, 300);
      };
      for (const event of ["add", "change", "unlink"] as const) server.watcher.on(event, onChange);
    },
  };
}

// base "./" keeps every asset path relative, so the built site works from any GitHub Pages path.
export default defineConfig({
  base: "./",
  plugins: [react(), studiesContent()],
});
