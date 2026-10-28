import { useEffect, useMemo, useState } from "react";
import { SearchBox, StudyCard } from "../components";
import { STATUS_INFO, evidenceLevel, type Content, type Status, type Study } from "../data";
import { href } from "../router";
import { snippet, type Hit } from "../search";

const KIND_LABEL = { paper: "Paper", review: "Review", notebook: "Notebook", page: "Page" } as const;

function Snippet({ hit, label = true }: { hit: Hit; label?: boolean }) {
  return (
    <p className="snippet">
      {label ? <span className="snippet-kind">{KIND_LABEL[hit.kind]}</span> : null}
      {snippet(hit.text, hit.terms).map((p, i) => (p.mark ? <mark key={i}>{p.text}</mark> : <span key={i}>{p.text}</span>))}
    </p>
  );
}

export function Studies({ content, search, initialQuery }: {
  content: Content;
  search: (query: string) => Hit[];
  initialQuery: string;
}) {
  const [query, setQuery] = useState(initialQuery);
  const [statuses, setStatuses] = useState<Set<Status>>(new Set());
  const [track, setTrack] = useState("");
  const [minEvidence, setMinEvidence] = useState(0);

  useEffect(() => setQuery(initialQuery), [initialQuery]);
  useEffect(() => {
    const target = href.studies(query.trim());
    if (window.location.hash !== target) window.history.replaceState(null, "", target);
  }, [query]);

  const q = query.trim();
  const hits = useMemo(() => search(q), [search, q]);
  const byId = useMemo(() => new Map(content.studies.map((s) => [s.id, s])), [content]);
  const counts = new Map<Status, number>();
  for (const s of content.studies) counts.set(s.status, (counts.get(s.status) ?? 0) + 1);

  const passes = (s: Study) =>
    (!statuses.size || statuses.has(s.status)) && (!track || s.track === track) && evidenceLevel(s.evidence) >= minEvidence;
  let rows: { study: Study; hits: Hit[] }[];
  if (q) {
    const grouped = new Map<string, Hit[]>();
    for (const h of hits) if (h.studyId) grouped.set(h.studyId, [...(grouped.get(h.studyId) ?? []), h]);
    rows = [...grouped]
      .map(([id, hs]) => ({ study: byId.get(id), hits: hs }))
      .filter((r): r is { study: Study; hits: Hit[] } => !!r.study && passes(r.study));
  } else {
    rows = content.studies
      .filter(passes)
      .sort((a, b) => b.id.localeCompare(a.id))
      .map((study) => ({ study, hits: [] }));
  }
  const pageHits = hits.filter((h) => h.kind === "page" && h.slug);
  const filtersOn = statuses.size > 0 || track !== "" || minEvidence > 0;
  const toggle = (s: Status) =>
    setStatuses((prev) => {
      const next = new Set(prev);
      if (next.has(s)) next.delete(s);
      else next.add(s);
      return next;
    });

  return (
    <>
      <header className="page-head">
        <h1>Studies</h1>
        <p className="muted">Every study in the lab, with its paper, review, notebook, logs and results.</p>
      </header>
      <SearchBox value={query} onChange={setQuery} large autoFocus />
      <div className="filters" role="group" aria-label="Filters">
        <div className="chips">
          {content.statuses
            .filter((s) => counts.get(s))
            .map((s) => (
              <button key={s} type="button" className={`chip status-${s}`} aria-pressed={statuses.has(s)} onClick={() => toggle(s)}>
                {STATUS_INFO[s].label} <span className="chip-count">{counts.get(s)}</span>
              </button>
            ))}
        </div>
        <label className="select">
          <span>Track</span>
          <select value={track} onChange={(e) => setTrack(e.target.value)}>
            <option value="">All tracks</option>
            {Object.entries(content.tracks).map(([key, name]) => (
              <option key={key} value={key}>
                {key}. {name}
              </option>
            ))}
          </select>
        </label>
        <label className="select">
          <span>Evidence</span>
          <select value={minEvidence} onChange={(e) => setMinEvidence(Number(e.target.value))}>
            <option value={0}>Any level</option>
            {[1, 2, 3, 4, 5].map((n) => (
              <option key={n} value={n}>
                E{n} or higher
              </option>
            ))}
          </select>
        </label>
        {filtersOn ? (
          <button
            type="button"
            className="link-button"
            onClick={() => {
              setStatuses(new Set());
              setTrack("");
              setMinEvidence(0);
            }}
          >
            Clear filters
          </button>
        ) : null}
      </div>

      <p className="result-count" aria-live="polite">
        {q
          ? `${rows.length} ${rows.length === 1 ? "study matches" : "studies match"} “${q}”`
          : `${rows.length} of ${content.studies.length} ${content.studies.length === 1 ? "study" : "studies"}`}
      </p>
      {rows.length ? (
        <div className="list">
          {rows.map(({ study, hits: studyHits }) => (
            <StudyCard key={study.id} study={study} tracks={content.tracks}>
              {studyHits.slice(0, 2).map((h) => <Snippet key={h.id} hit={h} />)}
            </StudyCard>
          ))}
        </div>
      ) : (
        <p className="state">
          No studies match{q ? " this search" : ""}
          {filtersOn ? " and these filters" : ""}. Try fewer or shorter words{filtersOn ? ", or clear the filters" : ""}.
        </p>
      )}

      {pageHits.length ? (
        <section className="section">
          <h2>Pages</h2>
          <ul className="page-hits">
            {pageHits.map((h) => (
              <li key={h.id}>
                <a href={href.page(h.slug!)}>{h.title}</a>
                <Snippet hit={h} label={false} />
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </>
  );
}
