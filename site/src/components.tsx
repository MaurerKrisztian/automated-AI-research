import { useEffect, useRef, useState, type ReactNode } from "react";
import { EVIDENCE_INFO, STATUS_INFO, evidenceLevel, type Status, type Study } from "./data";
import { href } from "./router";

export function StatusBadge({ status }: { status: Status }) {
  const info = STATUS_INFO[status] ?? { label: status, hint: "" };
  return <span className={`badge status-${status}`} title={info.hint}>{info.label}</span>;
}

export function EvidenceMeter({ level, showName = false }: { level: string; showName?: boolean }) {
  const n = evidenceLevel(level);
  const info = EVIDENCE_INFO[level];
  return (
    <span className="evidence" title={info ? `${level} · ${info.name}: ${info.hint}` : level}>
      <span className="evidence-label">{level}</span>
      <span className="evidence-bar" aria-hidden="true">
        {[1, 2, 3, 4, 5].map((i) => <span key={i} className={i <= n ? "on" : undefined} />)}
      </span>
      {showName && info ? <span className="evidence-name">{info.name}</span> : null}
    </span>
  );
}

export function Tags({ tags }: { tags: string[] }) {
  if (!tags.length) return null;
  return (
    <ul className="tags" aria-label="Tags">
      {tags.map((t) => <li key={t}><a href={href.studies(t)}>{t}</a></li>)}
    </ul>
  );
}

export function StudyMeta({ study, tracks }: { study: Study; tracks: Record<string, string> }) {
  return (
    <div className="study-meta">
      <span className="study-id">{study.id}</span>
      <StatusBadge status={study.status} />
      <EvidenceMeter level={study.evidence} />
      <span className="track">{tracks[study.track] ?? `Track ${study.track}`}</span>
    </div>
  );
}

export function StudyCard({ study, tracks, children }: { study: Study; tracks: Record<string, string>; children?: ReactNode }) {
  return (
    <article className="card">
      <StudyMeta study={study} tracks={tracks} />
      <h3><a href={href.study(study.id)}>{study.title}</a></h3>
      <p className="summary">{study.summary}</p>
      {children}
      <div className="card-foot">
        <Tags tags={study.tags} />
        <span className="updated">Updated {study.updated}</span>
      </div>
    </article>
  );
}

/** Search input that submits on Enter; "/" focuses it from anywhere on the page. */
export function SearchBox({ value, onChange, onSubmit, large = false, autoFocus = false }: {
  value: string;
  onChange: (v: string) => void;
  onSubmit?: (v: string) => void;
  large?: boolean;
  autoFocus?: boolean;
}) {
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const typing = target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable);
      if (e.key === "/" && !typing && input.current?.offsetParent) {
        e.preventDefault();
        input.current.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  return (
    <form
      className={large ? "search search-large" : "search"}
      role="search"
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit?.(value.trim());
      }}
    >
      <svg viewBox="0 0 20 20" aria-hidden="true" className="search-icon">
        <circle cx="8.5" cy="8.5" r="5.5" fill="none" stroke="currentColor" strokeWidth="2" />
        <path d="M13 13l4.5 4.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      </svg>
      <input
        ref={input}
        type="search"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Search studies, results, reviews…"
        aria-label="Search studies"
        autoFocus={autoFocus}
      />
      <kbd className="search-key" aria-hidden="true">/</kbd>
    </form>
  );
}

/**
 * A markdown document with its table of contents: a sticky side column on wide screens,
 * a collapsible "Contents" box above the text on narrow ones.
 */
export function DocLayout({ items, hrefFor, children }: {
  items: { id: string; text: string }[];
  hrefFor: (id: string) => string;
  children: ReactNode;
}) {
  if (items.length < 3) return <>{children}</>;
  const list = (
    <ol>
      {items.map((it) => (
        <li key={it.id}>
          <a href={hrefFor(it.id)}>{it.text}</a>
        </li>
      ))}
    </ol>
  );
  return (
    <div className="doc-layout">
      <details className="outline-inline">
        <summary>Contents</summary>
        {list}
      </details>
      {children}
      <nav className="outline-side" aria-label="Contents">
        <p>Contents</p>
        {list}
      </nav>
    </div>
  );
}

export function Loading() {
  return <p className="state">Loading studies…</p>;
}

export function NotFound({ what = "page" }: { what?: string }) {
  return (
    <section className="state">
      <h1>Not found</h1>
      <p>This {what} doesn't exist, or it hasn't been published yet.</p>
      <p><a href={href.studies()}>Browse all studies</a></p>
    </section>
  );
}

/** Light/dark toggle; follows the system until the reader picks one. */
export function ThemeToggle() {
  const [theme, setTheme] = useState<string | null>(() => {
    try {
      return localStorage.getItem("theme");
    } catch {
      return null;
    }
  });
  useEffect(() => {
    if (theme) document.documentElement.dataset.theme = theme;
    else delete document.documentElement.dataset.theme;
  }, [theme]);
  const dark = theme ? theme === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
  const toggle = () => {
    const next = dark ? "light" : "dark";
    setTheme(next);
    try {
      localStorage.setItem("theme", next);
    } catch {
      // storage unavailable: the choice lasts for this visit only
    }
  };
  return (
    <button type="button" className="icon-button" onClick={toggle} aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}>
      {dark ? (
        <svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="4" fill="currentColor" /><g stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">{[0, 45, 90, 135, 180, 225, 270, 315].map((a) => <path key={a} d="M10 1.8v2.2" transform={`rotate(${a} 10 10)`} />)}</g></svg>
      ) : (
        <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M15.5 12.6A6.5 6.5 0 0 1 7.4 4.5a6.5 6.5 0 1 0 8.1 8.1z" fill="currentColor" /></svg>
      )}
    </button>
  );
}
