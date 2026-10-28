import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Loading, NotFound, SearchBox, ThemeToggle } from "./components";
import { site } from "./config";
import { loadContent, type Content } from "./data";
import { DocPage } from "./pages/DocPage";
import { FileView } from "./pages/FileView";
import { Home } from "./pages/Home";
import { Studies } from "./pages/Studies";
import { StudyView } from "./pages/StudyView";
import { href, useRoute, type Route } from "./router";
import { buildSearch, type Hit } from "./search";

const NAV: { label: string; href: string; active: (r: Route) => boolean }[] = [
  { label: "Studies", href: href.studies(), active: (r) => r.name === "studies" || r.name === "study" || r.name === "file" },
  ...(["prompts", "contribute", "roadmap", "backlog", "protocol"] as const).map((slug) => ({
    label: slug[0].toUpperCase() + slug.slice(1),
    href: href.page(slug),
    active: (r: Route) => r.name === "page" && r.slug === slug,
  })),
];

function titleFor(route: Route, content: Content | null): string {
  let part: string | undefined;
  if (route.name === "studies") part = "Studies";
  else if (route.name === "study") part = content?.studies.find((s) => s.id === route.id)?.title;
  else if (route.name === "file") part = route.path.split("/").pop();
  else if (route.name === "page") part = content?.pages.find((p) => p.slug === route.slug)?.title;
  return part ? `${part} · ${site.name}` : site.name;
}

function Logo() {
  return (
    <svg viewBox="0 0 32 32" className="logo" aria-hidden="true">
      <rect width="32" height="32" rx="7" fill="var(--accent)" />
      <rect x="7" y="18" width="4.5" height="7" rx="1.2" fill="var(--on-accent)" />
      <rect x="13.75" y="13" width="4.5" height="12" rx="1.2" fill="var(--on-accent)" />
      <rect x="20.5" y="7" width="4.5" height="18" rx="1.2" fill="var(--on-accent)" />
    </svg>
  );
}

function ErrorState({ message }: { message: string }) {
  return (
    <section className="state">
      <h1>Couldn't load the studies</h1>
      <p>{message}</p>
      <p className="muted">
        The site reads <code>content/index.json</code>, which <code>npm run dev</code> and <code>npm run build</code> generate
        from the repository with <code>tools/studies.py export</code>.
      </p>
    </section>
  );
}

export function App() {
  const route = useRoute();
  const [content, setContent] = useState<Content | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [headerQuery, setHeaderQuery] = useState("");

  useEffect(() => {
    loadContent().then(setContent, (e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, []);
  const search = useMemo<(q: string) => Hit[]>(() => (content ? buildSearch(content) : () => []), [content]);
  useEffect(() => {
    document.title = titleFor(route, content);
  }, [route, content]);

  let view: ReactNode;
  if (error) view = <ErrorState message={error} />;
  else if (!content) view = <Loading />;
  else if (route.name === "home") view = <Home content={content} />;
  else if (route.name === "studies") view = <Studies content={content} search={search} initialQuery={route.query} />;
  else if (route.name === "study") view = <StudyView content={content} id={route.id} tab={route.tab} anchor={route.anchor} />;
  else if (route.name === "file") view = <FileView content={content} id={route.id} path={route.path} />;
  else if (route.name === "page") view = <DocPage content={content} slug={route.slug} anchor={route.anchor} />;
  else view = <NotFound />;

  const showHeaderSearch = route.name !== "home" && route.name !== "studies";
  const generated = content ? new Date(content.generated) : null;
  return (
    <>
      <a
        className="skip"
        href="#main"
        onClick={(e) => {
          e.preventDefault();
          document.getElementById("main")?.focus();
        }}
      >
        Skip to content
      </a>
      <header className="site-header">
        <div className="container header-row">
          <a className="brand" href={href.home()}>
            <Logo />
            <span>{site.name}</span>
          </a>
          <nav className="nav" aria-label="Main">
            {NAV.map((n) => (
              <a key={n.label} href={n.href} aria-current={n.active(route) ? "page" : undefined}>
                {n.label}
              </a>
            ))}
          </nav>
          <div className="header-tools">
            {showHeaderSearch ? (
              <SearchBox
                value={headerQuery}
                onChange={setHeaderQuery}
                onSubmit={(q) => {
                  window.location.hash = href.studies(q);
                  setHeaderQuery("");
                }}
              />
            ) : null}
            <ThemeToggle />
          </div>
        </div>
      </header>
      <main id="main" tabIndex={-1} className="container main">
        {view}
      </main>
      <footer className="site-footer">
        <div className="container footer-row">
          <p>
            Every number on this site comes from a log or results file in its study.{" "}
            <a href={href.page("protocol")}>How studies are run</a> · <a href={href.page("compute")}>Compute budget</a>
          </p>
          <p className="muted">
            {generated && !Number.isNaN(generated.getTime())
              ? `Built ${generated.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" })}`
              : null}
            {site.repoUrl ? (
              <>
                {" · "}
                <a href={site.repoUrl} target="_blank" rel="noreferrer">
                  Source on GitHub
                </a>
              </>
            ) : null}
          </p>
        </div>
      </footer>
    </>
  );
}
