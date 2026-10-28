# Website

A static, searchable website for the studies in this repository. For each study it shows the paper (`README.md`), the review, the notebook, and every log and results file; it also publishes the roadmap, backlog, protocol and compute pages. Built with React and Vite, and deployed to GitHub Pages.

## How it works

- `npm run dev` and `npm run build` first run `python3 ../tools/studies.py export public/content`. That checks every study, then writes `public/content/index.json` and copies each study's files (up to 5 MB each). The build stops if the check finds errors.
- The browser loads `content/index.json` and builds the full-text search index (MiniSearch) there. Math renders with KaTeX from ```` ```math ```` fences and `$…$`, the same syntax GitHub renders.
- Routes live in the URL hash (`#/study/0001/review`) and every asset path is relative, so the built `dist/` folder works from any path: a GitHub Pages project site, a user site or any static file server.
- During `npm run dev`, editing a study, a doc or the backlog re-exports the content and reloads the page.

## Commands

Run from this folder.

| Command | What it does |
|---|---|
| `npm install` | Install the dependencies (once) |
| `npm run dev` | Development server at http://localhost:5173 |
| `npm run build` | Check and export the studies, type-check, and build into `dist/` |
| `npm run preview` | Serve the built `dist/` |

## Deploying to GitHub Pages

1. Push the repository to GitHub.
2. In the repository's Settings → Pages, set Source to **GitHub Actions**.
3. Every push to `main` runs `.github/workflows/site.yml`, which builds the site and publishes it at https://maurerkrisztian.github.io/automated-AI-research/.

`repoUrl` in `src/config.ts` points links to code outside `studies/` at the GitHub repository. Everything in `studies/` gets published, and a GitHub Pages site is public.

## Code

| File | Contents |
|---|---|
| `src/App.tsx` | Layout, header, footer and route switch |
| `src/pages/` | Home, Studies (search and filters), StudyView (paper, review, notebook, files), FileView, DocPage |
| `src/Markdown.tsx` | Markdown and math rendering, heading anchors, table of contents |
| `src/search.ts` | Search index and result snippets |
| `src/links.ts` | Maps repository links inside Markdown to site routes and files |
| `src/data.ts` | Content types, labels and loading |
| `src/router.ts` | Hash routes |
| `src/styles.css` | Design tokens (light and dark) and styles |
